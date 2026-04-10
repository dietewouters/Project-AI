"""
Hyperparameter optimisation for Chemprop v2 via Optuna.

Uses the Python-API training path with TPE sampling and Median Pruning
to efficiently search the hyperparameter space, using validation loss
as the objective.

The search space covers the parameters that matter most for GNN
regression quality:
    - message passing hidden dim  (d_h)
    - message passing depth
    - dropout rate
    - FFN hidden dim & n_layers
    - learning-rate schedule  (init, max, final)
    - batch size
    - warmup epochs
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Default search-space bounds  (can be overridden from CLI state)
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE: dict[str, Any] = {
    "d_h":           {"low": 100,  "high": 600,  "step": 50},
    "depth":         {"low": 2,    "high": 5},
    "dropout":       {"low": 0.0,  "high": 0.4},
    "ffn_hidden":    {"low": 100,  "high": 600,  "step": 50},
    "ffn_layers":    {"low": 1,    "high": 3},
    "init_lr":       {"low": 1e-5, "high": 1e-3, "log": True},
    "max_lr":        {"low": 5e-4, "high": 5e-2, "log": True},
    "final_lr":      {"low": 1e-6, "high": 1e-3, "log": True},
    "batch_size":    {"choices": [32, 64, 128, 256]},
    "warmup_epochs": {"low": 1,    "high": 5},
}


# ---------------------------------------------------------------------------
#  Objective function
# ---------------------------------------------------------------------------

def _build_objective(
    train_csv: str,
    val_csv: str,
    target_col: str,
    smiles_col: str,
    descriptor_columns: list[str] | None,
    node_noise_config: dict | None,
    seed: int,
    hpo_epochs: int,
    search_space: dict,
    disable_output_scaling: bool = False,
    disable_input_scaling: bool = False,
):
    """Return an Optuna objective closure that trains and returns val_loss."""

    import torch
    import lightning as pl
    from rdkit import Chem
    from sklearn.preprocessing import StandardScaler

    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
    from chemprop.nn.transforms import ScaleTransform, UnscaleTransform
    from chemprop.models import MPNN
    from node_noise import NoisyMessagePassing

    # ── pre-load data once (shared across all trials) ──────────────────
    def _csv_to_dataset(csv_path):
        df = pd.read_csv(csv_path)
        smiles_list = df[smiles_col].tolist()
        targets = df[[target_col]].values.astype(float)
        x_d = df[descriptor_columns].values.astype(float) if descriptor_columns else None
        dps = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            dps.append(MoleculeDatapoint(
                mol=mol, y=targets[i],
                x_d=x_d[i] if x_d is not None else None,
            ))
        return MoleculeDataset(dps)

    train_ds = _csv_to_dataset(train_csv)
    val_ds   = _csv_to_dataset(val_csv)

    # Target scaling
    output_scaler = None
    if not disable_output_scaling:
        train_targets = np.array([dp.y for dp in train_ds])
        mean_val = float(np.nanmean(train_targets))
        std_val  = float(np.nanstd(train_targets))
        if std_val == 0:
            std_val = 1.0

        scaler = StandardScaler()
        scaler.mean_  = np.array([mean_val])
        scaler.scale_ = np.array([std_val])
        scaler.var_   = np.array([std_val ** 2])
        scaler.n_features_in_ = 1
        output_scaler = scaler

    # Input descriptor scaling
    X_d_transform = None
    if descriptor_columns and not disable_input_scaling:
        xd_scaler = train_ds.normalize_inputs("X_d")
        val_ds.normalize_inputs("X_d", scaler=xd_scaler)
        X_d_transform = ScaleTransform.from_standard_scaler(xd_scaler)

    d_xd = len(descriptor_columns) if descriptor_columns else 0
    sp = search_space  # alias

    # ── Optuna objective ───────────────────────────────────────────────
    def objective(trial):
        # Sample hyperparameters
        d_h      = trial.suggest_int("d_h",        sp["d_h"]["low"],        sp["d_h"]["high"],        step=sp["d_h"].get("step", 50))
        depth    = trial.suggest_int("depth",      sp["depth"]["low"],      sp["depth"]["high"])
        dropout  = trial.suggest_float("dropout",  sp["dropout"]["low"],    sp["dropout"]["high"])
        ffn_h    = trial.suggest_int("ffn_hidden",  sp["ffn_hidden"]["low"], sp["ffn_hidden"]["high"], step=sp["ffn_hidden"].get("step", 50))
        ffn_n    = trial.suggest_int("ffn_layers",  sp["ffn_layers"]["low"], sp["ffn_layers"]["high"])
        init_lr  = trial.suggest_float("init_lr",  sp["init_lr"]["low"],    sp["init_lr"]["high"],    log=sp["init_lr"].get("log", True))
        max_lr   = trial.suggest_float("max_lr",   sp["max_lr"]["low"],     sp["max_lr"]["high"],     log=sp["max_lr"].get("log", True))
        final_lr = trial.suggest_float("final_lr", sp["final_lr"]["low"],   sp["final_lr"]["high"],   log=sp["final_lr"].get("log", True))
        bs       = trial.suggest_categorical("batch_size", sp["batch_size"]["choices"])
        warmup   = trial.suggest_int("warmup_epochs", sp["warmup_epochs"]["low"], sp["warmup_epochs"]["high"])

        # Ensure lr ordering makes sense
        if init_lr > max_lr:
            init_lr, max_lr = max_lr, init_lr
        if final_lr > max_lr:
            final_lr = init_lr

        output_transform = (
            UnscaleTransform.from_standard_scaler(output_scaler)
            if output_scaler is not None
            else None
        )

        # Build model
        mp = BondMessagePassing(d_v=72, d_e=14, d_h=d_h, depth=depth, dropout=dropout)

        if node_noise_config and node_noise_config.get("layers"):
            nf = node_noise_config.get("node_fraction", 0.0)
            df_frac = node_noise_config.get("dim_fraction", 0.0)
            layers = node_noise_config.get("layers", [])
            if nf > 0 and df_frac > 0 and layers:
                mp = NoisyMessagePassing(mp, nf, df_frac, layers)

        agg = MeanAggregation()
        ffn = RegressionFFN(
            input_dim=d_h + d_xd,
            hidden_dim=ffn_h,
            n_layers=ffn_n,
            n_tasks=1,
            output_transform=output_transform,
        )

        model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=ffn,
            batch_norm=True,
            warmup_epochs=warmup,
            init_lr=init_lr,
            max_lr=max_lr,
            final_lr=final_lr,
            X_d_transform=X_d_transform,
        )

        train_loader = build_dataloader(train_ds, batch_size=bs, num_workers=0, shuffle=True, seed=seed)
        val_loader   = build_dataloader(val_ds, batch_size=bs, num_workers=0, shuffle=False)

        # Pruning callback for Optuna
        from optuna.integration import PyTorchLightningPruningCallback
        pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_loss")

        pl.seed_everything(seed)
        trainer = pl.Trainer(
            max_epochs=hpo_epochs,
            callbacks=[pruning_cb],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
        )

        try:
            trainer.fit(model, train_loader, val_loader)
        except Exception:
            return float("inf")

        # Return best validation loss
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return float("inf")
        return val_loss.item()

    return objective


# ---------------------------------------------------------------------------
#  Main HPO runner
# ---------------------------------------------------------------------------

def run_hpo(
    train_csv: str,
    val_csv: str,
    target_col: str = "h298",
    smiles_col: str = "smiles",
    descriptor_columns: list[str] | None = None,
    node_noise_config: dict | None = None,
    seed: int = 42,
    n_trials: int = 20,
    hpo_epochs: int = 10,
    search_space: dict | None = None,
    save_dir: str | None = None,
    disable_output_scaling: bool = False,
    disable_input_scaling: bool = False,
) -> dict:
    """
    Run Optuna hyperparameter search and return the best parameters.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials to evaluate.
    hpo_epochs : int
        Epochs per trial (shorter = faster search; 10-15 is a good default).
    search_space : dict | None
        Override DEFAULT_SEARCH_SPACE with custom bounds.
    save_dir : str | None
        If given, saves the Optuna study database and results JSON here.

    Returns
    -------
    dict : the best hyperparameters found.
    """
    import optuna
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich import print as rprint

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sp = {**DEFAULT_SEARCH_SPACE, **(search_space or {})}

    rprint(Panel(
        f"Trials: {n_trials}  |  Epochs/trial: {hpo_epochs}  |  Seed: {seed}",
        title="[bold cyan]Hyperparameter Optimisation[/bold cyan]",
        border_style="magenta",
    ))

    objective = _build_objective(
        train_csv=train_csv,
        val_csv=val_csv,
        target_col=target_col,
        smiles_col=smiles_col,
        descriptor_columns=descriptor_columns,
        node_noise_config=node_noise_config,
        seed=seed,
        hpo_epochs=hpo_epochs,
        search_space=sp,
        disable_output_scaling=disable_output_scaling,
        disable_input_scaling=disable_input_scaling,
    )

    # Storage
    storage = None
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        db_path = Path(save_dir) / "hpo_study.db"
        storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        storage=storage,
        study_name="chemprop_hpo",
        load_if_exists=True,
    )

    # Progress tracking with rich
    from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[yellow]{task.fields[best_val]}"),
    ) as progress:
        task = progress.add_task("HPO Trials", total=n_trials, best_val="best: ---")

        def trial_callback(study, trial):
            best = study.best_value if study.best_trial else float("inf")
            progress.update(task, advance=1, best_val=f"best val_loss: {best:.4f}")

        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

    best = study.best_params
    best_val = study.best_value

    # Show results
    tbl = Table(box=box.ROUNDED, border_style="green", expand=False,
                title="[bold green]Best Hyperparameters[/bold green]")
    tbl.add_column("Parameter", style="accent", width=18)
    tbl.add_column("Value", style="white")
    for k, v in sorted(best.items()):
        tbl.add_row(k, f"{v:.6g}" if isinstance(v, float) else str(v))
    tbl.add_row("─" * 18, "─" * 12)
    tbl.add_row("Best val_loss", f"{best_val:.6f}")
    rprint(tbl)

    # Save results
    if save_dir:
        results_path = Path(save_dir) / "hpo_best_params.json"
        with open(results_path, "w") as f:
            json.dump({"best_params": best, "best_val_loss": best_val}, f, indent=2)
        rprint(f"  [green]HPO results saved → {results_path}[/green]")

    return best
