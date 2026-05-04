"""
Chemprop encoder used as a deterministic molecular embedder.

We train a standard Chemprop v2 regressor on (SMILES, h298) where h298 is the
deterministic group-additivity target. Once training has converged, the FFN
head is discarded and only the message-passing encoder is used: every molecule
gets mapped to a fixed-dimensional vector whose neighbours in embedding space
are molecules with similar group-additivity h298.

That embedding space is then used by ``denoise`` for inverse-distance-weighted
k-NN denoising of a separate, noisy CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── data loading ──────────────────────────────────────────────────────────

def _csv_to_dataset(csv_path: str, smiles_col: str, target_col: str):
    """Build a chemprop MoleculeDataset from a CSV.

    Returns ``(dataset, valid_indices_into_df, df)``. ``valid_indices_into_df``
    lists the rows for which RDKit parsed the SMILES — anything else is
    silently skipped.
    """
    from rdkit import Chem
    from chemprop.data import MoleculeDatapoint, MoleculeDataset

    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column {smiles_col!r} not in {csv_path}")
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not in {csv_path}")

    smiles_list = df[smiles_col].astype(str).tolist()
    y_vals = df[target_col].astype(float).values

    dps, valid = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        dps.append(MoleculeDatapoint(mol=mol, y=np.array([y_vals[i]])))
        valid.append(i)
    return MoleculeDataset(dps), valid, df


def _smiles_to_dataset(smiles_list: list[str]):
    """Build a label-less MoleculeDataset for inference-only use."""
    from rdkit import Chem
    from chemprop.data import MoleculeDatapoint, MoleculeDataset

    dps, valid = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        dps.append(MoleculeDatapoint(mol=mol, y=np.array([0.0])))
        valid.append(i)
    return MoleculeDataset(dps), valid


# ── training ──────────────────────────────────────────────────────────────

DEFAULT_HPARAMS = {
    "d_h":           300,
    "depth":         3,
    "dropout":       0.0,
    "ffn_hidden":    300,
    "ffn_layers":    1,
    "init_lr":       1e-4,
    "max_lr":        1e-3,
    "final_lr":      1e-4,
    "warmup_epochs": 2,
}


def train_embedder(
    train_csv: str,
    val_csv: str,
    smiles_col: str,
    target_col: str,
    save_dir: str,
    epochs: int = 30,
    batch_size: int = 128,
    seed: int = 42,
    tuned_hparams: dict | None = None,
):
    """Train a Chemprop v2 regressor and return the trained model.

    The FFN head stays attached so that PyTorch-Lightning checkpointing
    round-trips work; downstream code calls ``model.fingerprint(...)`` to
    extract the encoder's output and never invokes the head.
    """
    import lightning as pl
    from chemprop.data import build_dataloader
    from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
    from chemprop.nn.transforms import UnscaleTransform
    from chemprop.models import MPNN
    from sklearn.preprocessing import StandardScaler
    from rich import print as rprint

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    rprint(f"  [cyan]Loading train ← {train_csv}[/cyan]")
    train_ds, _, _ = _csv_to_dataset(train_csv, smiles_col, target_col)
    rprint(f"  [cyan]Loading val   ← {val_csv}[/cyan]")
    val_ds, _, _ = _csv_to_dataset(val_csv, smiles_col, target_col)

    # Output scaling: standardise targets so the FFN sees a unit-scale signal,
    # then unscale on the way out.
    train_targets = np.array([dp.y for dp in train_ds]).reshape(-1)
    mean_val = float(np.nanmean(train_targets))
    std_val = float(np.nanstd(train_targets)) or 1.0
    scaler = StandardScaler()
    scaler.mean_ = np.array([mean_val])
    scaler.scale_ = np.array([std_val])
    scaler.var_ = np.array([std_val ** 2])
    scaler.n_features_in_ = 1
    output_transform = UnscaleTransform.from_standard_scaler(scaler)

    hp = {**DEFAULT_HPARAMS, **(tuned_hparams or {})}
    if "batch_size" in hp:
        batch_size = hp["batch_size"]

    mp = BondMessagePassing(
        d_v=72, d_e=14,
        d_h=hp["d_h"], depth=hp["depth"], dropout=hp["dropout"],
    )
    agg = MeanAggregation()
    ffn = RegressionFFN(
        input_dim=hp["d_h"],
        hidden_dim=hp["ffn_hidden"],
        n_layers=hp["ffn_layers"],
        n_tasks=1,
        output_transform=output_transform,
    )

    model = MPNN(
        message_passing=mp, agg=agg, predictor=ffn,
        batch_norm=True,
        warmup_epochs=hp["warmup_epochs"],
        init_lr=hp["init_lr"], max_lr=hp["max_lr"], final_lr=hp["final_lr"],
    )

    train_loader = build_dataloader(
        train_ds, batch_size=batch_size, num_workers=0, shuffle=True, seed=seed,
    )
    val_loader = build_dataloader(
        val_ds, batch_size=batch_size, num_workers=0, shuffle=False,
    )

    pl.seed_everything(seed)
    checkpoint_cb = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        filename="encoder-{epoch}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    early_stop_cb = pl.pytorch.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.005, patience=2, mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        default_root_dir=save_dir,
        enable_progress_bar=True,
        logger=pl.pytorch.loggers.CSVLogger(save_dir),
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)
    return model


# ── embedding extraction ──────────────────────────────────────────────────

def extract_embeddings(model, smiles_list: list[str], batch_size: int = 128):
    """Embed a list of SMILES with the trained encoder.

    Returns ``(z, valid_indices)`` where ``z`` has shape ``(len(valid), d)``
    and ``valid_indices`` are the positions in ``smiles_list`` whose SMILES
    parsed cleanly.
    """
    import torch
    from chemprop.data import build_dataloader

    dataset, valid = _smiles_to_dataset(smiles_list)
    if len(dataset) == 0:
        return np.zeros((0, 0), dtype=np.float32), valid

    model.eval()
    device = next(model.parameters()).device
    loader = build_dataloader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False,
    )

    chunks = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            if hasattr(bmg, "to"):
                bmg = bmg.to(device)
            V_d = getattr(batch, "V_d", None)
            X_d = getattr(batch, "X_d", None)
            if V_d is not None and hasattr(V_d, "to"):
                V_d = V_d.to(device)
            if X_d is not None and hasattr(X_d, "to"):
                X_d = X_d.to(device)
            z = model.fingerprint(bmg, V_d=V_d, X_d=X_d)
            chunks.append(z.detach().cpu().numpy())

    return np.concatenate(chunks, axis=0), valid


# ── checkpoint helpers ────────────────────────────────────────────────────

def save_model(model, path: str | Path) -> None:
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def load_model(path: str | Path, hparams: dict | None = None):
    """Recreate the architecture, load weights, return the model in eval mode."""
    import torch
    from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
    from chemprop.models import MPNN

    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    mp = BondMessagePassing(
        d_v=72, d_e=14,
        d_h=hp["d_h"], depth=hp["depth"], dropout=hp["dropout"],
    )
    agg = MeanAggregation()
    ffn = RegressionFFN(
        input_dim=hp["d_h"],
        hidden_dim=hp["ffn_hidden"],
        n_layers=hp["ffn_layers"],
        n_tasks=1,
    )
    model = MPNN(
        message_passing=mp, agg=agg, predictor=ffn,
        batch_norm=True,
        warmup_epochs=hp["warmup_epochs"],
        init_lr=hp["init_lr"], max_lr=hp["max_lr"], final_lr=hp["final_lr"],
    )
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    model.eval()
    return model
