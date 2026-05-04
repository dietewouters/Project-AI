"""
Optional hyperparameter optimisation for the embedder.

Runs Optuna with TPE sampling over the same hyperparameters that
``embedder.train_embedder`` accepts. Each trial trains a stripped-down
version of that model for ``hpo_epochs`` epochs and reports validation
loss; the best parameters are written to ``hpo_best_params.json`` and
loaded automatically when the embedder is trained next.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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


def _build_objective(
    train_csv: str,
    val_csv: str,
    smiles_col: str,
    target_col: str,
    seed: int,
    hpo_epochs: int,
    search_space: dict,
):
    """Return an Optuna objective closure that trains and returns val_loss."""
    import lightning as pl
    from sklearn.preprocessing import StandardScaler

    from chemprop.data import build_dataloader
    from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
    from chemprop.nn.transforms import UnscaleTransform
    from chemprop.models import MPNN

    from embedder import _csv_to_dataset

    # Load datasets once and share across trials.
    train_ds, _, _ = _csv_to_dataset(train_csv, smiles_col, target_col)
    val_ds, _, _ = _csv_to_dataset(val_csv, smiles_col, target_col)

    train_targets = np.array([dp.y for dp in train_ds]).reshape(-1)
    mean_val = float(np.nanmean(train_targets))
    std_val = float(np.nanstd(train_targets)) or 1.0
    scaler = StandardScaler()
    scaler.mean_ = np.array([mean_val])
    scaler.scale_ = np.array([std_val])
    scaler.var_ = np.array([std_val ** 2])
    scaler.n_features_in_ = 1
    output_transform = UnscaleTransform.from_standard_scaler(scaler)

    sp = search_space

    def objective(trial):
        d_h = trial.suggest_int(
            "d_h", sp["d_h"]["low"], sp["d_h"]["high"], step=sp["d_h"].get("step", 50),
        )
        depth = trial.suggest_int("depth", sp["depth"]["low"], sp["depth"]["high"])
        dropout = trial.suggest_float("dropout", sp["dropout"]["low"], sp["dropout"]["high"])
        ffn_h = trial.suggest_int(
            "ffn_hidden", sp["ffn_hidden"]["low"], sp["ffn_hidden"]["high"],
            step=sp["ffn_hidden"].get("step", 50),
        )
        ffn_n = trial.suggest_int(
            "ffn_layers", sp["ffn_layers"]["low"], sp["ffn_layers"]["high"],
        )
        init_lr = trial.suggest_float(
            "init_lr", sp["init_lr"]["low"], sp["init_lr"]["high"],
            log=sp["init_lr"].get("log", True),
        )
        max_lr = trial.suggest_float(
            "max_lr", sp["max_lr"]["low"], sp["max_lr"]["high"],
            log=sp["max_lr"].get("log", True),
        )
        final_lr = trial.suggest_float(
            "final_lr", sp["final_lr"]["low"], sp["final_lr"]["high"],
            log=sp["final_lr"].get("log", True),
        )
        bs = trial.suggest_categorical("batch_size", sp["batch_size"]["choices"])
        warmup = trial.suggest_int(
            "warmup_epochs", sp["warmup_epochs"]["low"], sp["warmup_epochs"]["high"],
        )

        # Keep the LR schedule monotone.
        if init_lr > max_lr:
            init_lr, max_lr = max_lr, init_lr
        if final_lr > max_lr:
            final_lr = init_lr

        mp = BondMessagePassing(d_v=72, d_e=14, d_h=d_h, depth=depth, dropout=dropout)
        agg = MeanAggregation()
        ffn = RegressionFFN(
            input_dim=d_h, hidden_dim=ffn_h, n_layers=ffn_n, n_tasks=1,
            output_transform=output_transform,
        )
        model = MPNN(
            message_passing=mp, agg=agg, predictor=ffn, batch_norm=True,
            warmup_epochs=warmup, init_lr=init_lr, max_lr=max_lr, final_lr=final_lr,
        )

        train_loader = build_dataloader(
            train_ds, batch_size=bs, num_workers=0, shuffle=True, seed=seed,
        )
        val_loader = build_dataloader(
            val_ds, batch_size=bs, num_workers=0, shuffle=False,
        )

        pl.seed_everything(seed)
        trainer = pl.Trainer(
            max_epochs=hpo_epochs,
            enable_progress_bar=False, enable_model_summary=False,
            logger=False, num_sanity_val_steps=0,
        )
        try:
            trainer.fit(model, train_loader, val_loader)
        except Exception:
            return float("inf")

        val_loss = trainer.callback_metrics.get("val_loss")
        return val_loss.item() if val_loss is not None else float("inf")

    return objective


def run_hpo(
    train_csv: str,
    val_csv: str,
    smiles_col: str = "smiles",
    target_col: str = "h298",
    seed: int = 42,
    n_trials: int = 20,
    hpo_epochs: int = 10,
    search_space: dict | None = None,
    save_dir: str | None = None,
) -> dict:
    """Run Optuna HPO and return the best hyperparameter dict.

    If ``save_dir`` is given, writes the study database and a
    ``hpo_best_params.json`` file there for later reuse.
    """
    import optuna
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn,
    )
    from rich import box, print as rprint

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sp = {**DEFAULT_SEARCH_SPACE, **(search_space or {})}

    rprint(Panel(
        f"Trials: {n_trials}  |  Epochs/trial: {hpo_epochs}  |  Seed: {seed}",
        title="[bold cyan]Hyperparameter Optimisation[/bold cyan]",
        border_style="magenta",
    ))

    objective = _build_objective(
        train_csv=train_csv, val_csv=val_csv,
        smiles_col=smiles_col, target_col=target_col,
        seed=seed, hpo_epochs=hpo_epochs, search_space=sp,
    )

    storage = None
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        db_path = Path(save_dir) / "hpo_study.db"
        storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        storage=storage, study_name="embedder_hpo", load_if_exists=True,
    )

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[yellow]{task.fields[best_val]}"),
    ) as progress:
        task = progress.add_task("HPO Trials", total=n_trials, best_val="best: ---")

        def trial_callback(_study, _trial):
            best = _study.best_value if _study.best_trial else float("inf")
            progress.update(task, advance=1, best_val=f"best val_loss: {best:.4f}")

        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

    best = dict(study.best_params)
    best_val = float(study.best_value)

    tbl = Table(
        box=box.ROUNDED, border_style="green", expand=False,
        title="[bold green]Best Hyperparameters[/bold green]",
    )
    tbl.add_column("Parameter", style="bold magenta", width=18)
    tbl.add_column("Value", style="white")
    for k, v in sorted(best.items()):
        tbl.add_row(k, f"{v:.6g}" if isinstance(v, float) else str(v))
    tbl.add_row("─" * 18, "─" * 12)
    tbl.add_row("Best val_loss", f"{best_val:.6f}")
    rprint(tbl)

    if save_dir:
        results_path = Path(save_dir) / "hpo_best_params.json"
        with open(results_path, "w") as f:
            json.dump({"best_params": best, "best_val_loss": best_val}, f, indent=2)
        rprint(f"  [green]HPO results saved → {results_path}[/green]")

    return best
