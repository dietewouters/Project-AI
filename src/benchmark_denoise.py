#!/usr/bin/env python3
"""
Benchmark iterative k-NN denoising against group-additivity ground truth.

For each noise spec (a list of noise layers from ``noise.generate_layer``), the
script:
  1. Takes y_truth = GA-computed h298 for a held-out subsample of SMILES.
  2. Applies the noise spec to produce y_noisy.
  3. Runs ``iterative_denoise(z, y_noisy, k, rounds)`` ONCE and captures the
     full per-round history.
  4. Computes metrics (MAE, RMSE, R², bias, residual std, p50 |err|) for every
     round in the history versus y_truth.

The output is a long-format results.csv plus seven thesis-quality figures.
The embedder is reused as-is (it was trained on GA truth elsewhere). The
held-out pool comes from ``groupadditivity_test.csv`` which the embedder has
not seen at training time.

Standalone usage:
    python -m src.benchmark_denoise                    # Gaussian sweep default
    python -m src.benchmark_denoise --preset smoke
    python -m src.benchmark_denoise --preset noise_types
    python -m src.benchmark_denoise --preset full_grid
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import denoise as denoise_mod
import embedder as embedder_mod
import ga_compute as ga_mod
import noise as noise_mod


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GA_TEST_CSV = REPO_ROOT / "groupadditivity_h298" / "dataset" / "groupadditivity_test.csv"
# The non-test split of the GA dataset — safe for embedder training (no test
# leakage). Falls back to groupadditivity.csv if `_1.csv` doesn't exist.
DEFAULT_GA_TRAIN_CSV = REPO_ROOT / "groupadditivity_h298" / "dataset" / "groupadditivity_1.csv"
DEFAULT_GA_FIT_CSV = REPO_ROOT / "groupadditivity_h298" / "dataset" / "groupadditivity.csv"
DEFAULT_EMBEDDER_DIR = Path(__file__).resolve().parent / "cli_workspace" / "embedder"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "cli_workspace" / "benchmark"


# ── presets ───────────────────────────────────────────────────────────────

def _gaussian_layers(sigma: float) -> list[dict]:
    return [{"type": "normal", "scale": float(sigma)}]


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "label": "Smoke test (~1 min)",
        "description": "1k molecules, 5 rounds, 1 seed, σ=5 — verifies the pipeline.",
        "n_sample": 1000,
        "rounds": 5,
        "n_seeds": 1,
        "k": 10,
        "noise_specs": [
            {"label": "normal σ=5", "type": "normal", "scale": 5.0,
             "layers": _gaussian_layers(5.0)},
        ],
    },
    "gaussian_sweep": {
        "label": "Gaussian sweep  (σ ∈ {1,2,5,10,20})",
        "description": "100k mols, 30 rounds, 3 seeds — the canonical thesis sweep.",
        "n_sample": 100000,
        "rounds": 30,
        "n_seeds": 3,
        "k": 10,
        "noise_specs": [
            {"label": f"normal σ={s:g}", "type": "normal", "scale": float(s),
             "layers": _gaussian_layers(s)}
            for s in [1, 2, 5, 10, 20]
        ],
    },
    "noise_types": {
        "label": "Noise type comparison  (scale=5)",
        "description": "Same scale, different distributions: normal/uniform/cosh/bimodal/outlier.",
        "n_sample": 50000,
        "rounds": 20,
        "n_seeds": 3,
        "k": 10,
        "noise_specs": [
            {"label": "normal s=5", "type": "normal", "scale": 5.0,
             "layers": [{"type": "normal", "scale": 5.0}]},
            {"label": "uniform s=5", "type": "uniform", "scale": 5.0,
             "layers": [{"type": "uniform", "scale": 5.0}]},
            {"label": "cosh s=5", "type": "cosh", "scale": 5.0,
             "layers": [{"type": "cosh", "scale": 5.0}]},
            {"label": "bimodal s=5", "type": "bimodal", "scale": 5.0,
             "layers": [{"type": "bimodal", "scale": 5.0}]},
            {"label": "outlier 10% σ=20", "type": "outlier", "scale": 20.0,
             "layers": [{"type": "outlier", "scale": 20.0, "fraction": 0.1}]},
        ],
    },
    "full_grid": {
        "label": "Full grid  (types × levels)",
        "description": "All shaped types at scales {2, 5, 10} + outlier fractions — long run.",
        "n_sample": 50000,
        "rounds": 20,
        "n_seeds": 2,
        "k": 10,
        "noise_specs": (
            [
                {"label": f"{t} s={s:g}", "type": t, "scale": float(s),
                 "layers": [{"type": t, "scale": float(s)}]}
                for t in ["normal", "uniform", "cosh", "bimodal"]
                for s in [2, 5, 10]
            ]
            + [
                {"label": f"outlier {int(f * 100)}% σ=20",
                 "type": "outlier", "scale": 20.0,
                 "layers": [{"type": "outlier", "scale": 20.0, "fraction": float(f)}]}
                for f in [0.05, 0.1, 0.2]
            ]
        ),
    },
}


# ── data prep ─────────────────────────────────────────────────────────────

def load_test_pool(
    ga_test_csv: Path,
    n_sample: int,
    seed: int,
    stratify_bins: int = 20,
) -> pd.DataFrame:
    """Read the held-out GA test CSV and return ``n_sample`` rows.

    Sampling is **quantile-stratified by h298** so the returned subset spans
    the full property range — small / medium / large molecules and the whole
    enthalpy spectrum are all represented, instead of risking an isolated
    cluster of similar molecules. Each h298 quantile bin contributes ~equal
    numbers of rows; bins smaller than the per-bin quota contribute all their
    rows and the shortfall is filled with random draws.

    Set ``stratify_bins<=1`` to fall back to pure random sampling.
    """
    df = pd.read_csv(ga_test_csv, usecols=["smiles", "h298"])
    if len(df) <= n_sample:
        return df.reset_index(drop=True)

    if stratify_bins <= 1:
        return df.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    try:
        binned = pd.qcut(df["h298"], q=stratify_bins,
                         labels=False, duplicates="drop")
    except Exception:
        return df.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    df = df.assign(__bin=binned.astype("Int64"))
    df = df.dropna(subset=["__bin"])
    n_bins = int(df["__bin"].nunique())
    if n_bins <= 1:
        return df.drop(columns="__bin").sample(
            n=n_sample, random_state=seed,
        ).reset_index(drop=True)

    per_bin = n_sample // n_bins
    rng = np.random.default_rng(seed)
    parts = []
    for _, sub in df.groupby("__bin", observed=True):
        take = min(per_bin, len(sub))
        parts.append(sub.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(parts)

    # Fill the remainder from rows not yet picked, preserving uniqueness.
    if len(out) < n_sample:
        remaining = df.drop(out.index)
        extra = min(n_sample - len(out), len(remaining))
        if extra > 0:
            out = pd.concat([
                out,
                remaining.sample(n=extra, random_state=int(rng.integers(0, 2**31 - 1))),
            ])
    return out.drop(columns="__bin").reset_index(drop=True)


# ── embedder training (benchmark-side) ────────────────────────────────────

def _train_config_path(embedder_dir: Path) -> Path:
    return embedder_dir / "train_config.json"


def load_train_config(embedder_dir: Path | str) -> dict | None:
    """Return the provenance dict for the saved embedder, or None if missing."""
    p = _train_config_path(Path(embedder_dir))
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def needs_retraining(
    embedder_dir: Path | str,
    *,
    n_train: int,
    epochs: int,
    batch_size: int,
    ga_train_csv: Path | str,
) -> bool:
    """Return True iff a fresh ``train_embedder_for_benchmark`` is needed
    (no weights, no provenance, or provenance mismatches the requested config)."""
    embedder_dir = Path(embedder_dir)
    if not (embedder_dir / "embedder.pt").exists():
        return True
    saved = load_train_config(embedder_dir)
    if saved is None:
        return True
    return not (
        int(saved.get("n_train", -1)) == int(n_train)
        and int(saved.get("epochs", -1)) == int(epochs)
        and int(saved.get("batch_size", -1)) == int(batch_size)
        and str(saved.get("ga_train_csv")) == str(ga_train_csv)
    )


def train_embedder_for_benchmark(
    *,
    ga_train_csv: Path | str = DEFAULT_GA_TRAIN_CSV,
    embedder_dir: Path | str = DEFAULT_EMBEDDER_DIR,
    n_train: int = 100_000,
    epochs: int = 30,
    batch_size: int = 128,
    seed: int = 42,
    val_frac: float = 0.15,
    stratify_bins: int = 20,
) -> Path:
    """Train a fresh embedder on a stratified subsample of ``ga_train_csv``.

    Saves weights to ``embedder_dir/embedder.pt`` and provenance to
    ``embedder_dir/train_config.json``. Returns the weights path.
    """
    ga_train_csv = Path(ga_train_csv)
    embedder_dir = Path(embedder_dir)
    embedder_dir.mkdir(parents=True, exist_ok=True)

    if not ga_train_csv.exists():
        raise FileNotFoundError(
            f"GA train CSV not found: {ga_train_csv}. Did you mean groupadditivity.csv?"
        )

    _console.print(
        f"[cyan]embedder:[/cyan] subsampling [bold]{n_train:,}[/bold] mols "
        f"(stratified by h298) from {ga_train_csv.name}…"
    )
    df = load_test_pool(
        ga_train_csv, n_sample=n_train, seed=seed, stratify_bins=stratify_bins,
    )
    n_actual = len(df)

    # Split into train / val deterministically.
    n_val = max(1, int(round(n_actual * val_frac)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_actual)
    val_df = df.iloc[perm[:n_val]].reset_index(drop=True)
    train_df = df.iloc[perm[n_val:]].reset_index(drop=True)

    train_csv = embedder_dir / "_bench_train.csv"
    val_csv = embedder_dir / "_bench_val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    batches_per_epoch = max(1, (len(train_df) + batch_size - 1) // batch_size)
    _console.print(
        Panel(
            Text.from_markup(
                f"  train mols       [bold]{len(train_df):>10,}[/bold]\n"
                f"  val mols         [bold]{len(val_df):>10,}[/bold]\n"
                f"  batch size       [bold]{batch_size:>10,}[/bold]\n"
                f"  batches/epoch    [bold]{batches_per_epoch:>10,}[/bold]   "
                f"[bright_black](≈ train/batch_size)[/bright_black]\n"
                f"  max epochs       [bold]{epochs:>10,}[/bold]\n"
                f"  early stopping   [bold]val_loss patience=2, min_Δ=0.005[/bold]\n"
                f"  source           [bright_black]{ga_train_csv.name}[/bright_black]"
            ),
            title="[bold cyan]Embedder training[/bold cyan]",
            border_style="cyan",
            padding=(0, 2),
        )
    )

    model = embedder_mod.train_embedder(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        smiles_col="smiles",
        target_col="h298",
        save_dir=str(embedder_dir),
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    weights_path = embedder_dir / "embedder.pt"
    embedder_mod.save_model(model, weights_path)

    # Save provenance so future runs can detect a config mismatch.
    cfg = {
        "n_train": int(n_train),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "val_frac": float(val_frac),
        "stratify_bins": int(stratify_bins),
        "ga_train_csv": str(ga_train_csv),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train_actual": int(len(train_df)),
        "n_val_actual": int(len(val_df)),
    }
    with open(_train_config_path(embedder_dir), "w") as f:
        json.dump(cfg, f, indent=2)

    for tmp in (train_csv, val_csv):
        try:
            tmp.unlink()
        except Exception:
            pass

    _console.print(
        f"[green]✓[/green] embedder saved → [bright_black]{weights_path}[/bright_black]"
    )
    return weights_path


def compute_truth_and_embed(
    df: pd.DataFrame,
    model,
    ga_model: dict,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    smiles_all = df["smiles"].astype(str).tolist()
    y_ga, ga_mask = ga_mod.predict_h298(smiles_all, ga_model)
    ga_keep_idx = np.where(ga_mask)[0]
    smiles_ga = [smiles_all[i] for i in ga_keep_idx]
    y_ga_kept = y_ga[ga_keep_idx]

    z, emb_valid = embedder_mod.extract_embeddings(
        model, smiles_ga, batch_size=batch_size,
    )
    emb_valid = np.asarray(emb_valid, dtype=int)
    y_truth = y_ga_kept[emb_valid]
    smiles_final = [smiles_ga[i] for i in emb_valid]
    return z, y_truth, smiles_final


def apply_noise_spec(
    y_truth: np.ndarray,
    smiles: list[str],
    layers: list[dict],
    seed: int,
) -> np.ndarray:
    """Inject one full noise spec (sum of layers) onto y_truth."""
    noise_vec = noise_mod.apply_layers(y_truth, smiles, layers, base_seed=seed)
    return y_truth + noise_vec


# ── metrics ───────────────────────────────────────────────────────────────

def metrics_for_round(y_pred: np.ndarray, y_truth: np.ndarray) -> dict:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_truth, dtype=float)
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    truth_var = float(np.var(y_truth))
    r2 = 1.0 - mse / truth_var if truth_var > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": float(np.mean(err)),
        "residual_std": float(np.std(err)),
        "p50_abs_err": float(np.median(np.abs(err))),
    }


# ── live UI ───────────────────────────────────────────────────────────────

_console = Console()


class _SuiteUI:
    """Live status table + nested progress bars for benchmark runs.

    Layout:
        ┌── status table (one row per preset) ───────────────────┐
        │ Preset │ Status │ Mols │ Specs │ Best r │ MAE │ Elapsed │
        ├─── progress bars ──────────────────────────────────────┤
        │  Suite           [████░░░░] 1/2  preset complete         │
        │  Specs (preset)  [██░░░░░░] 1/5  normal σ=2               │
        │  Rounds          [██████░░] 18/30 mean=… std=…            │
        └────────────────────────────────────────────────────────┘
    """

    def __init__(self, console: Console | None = None, title: str = "Benchmark"):
        self.console = console or _console
        self.title = title
        self.progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]{task.description:<22s}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[bright_black]{task.fields[note]}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        self.rows: list[dict] = []
        self._live: Live | None = None
        self._suite_task: int | None = None
        self._spec_task: int | None = None
        self._round_task: int | None = None

    # ── table ──
    def _make_panel(self):
        t = Table(
            title=self.title, title_style="bold magenta",
            show_lines=False, expand=False,
        )
        t.add_column("Preset", style="cyan", no_wrap=True)
        t.add_column("Status")
        t.add_column("Mols", justify="right")
        t.add_column("Specs", justify="right")
        t.add_column("Best r", justify="right")
        t.add_column("Best MAE", justify="right")
        t.add_column("Elapsed", justify="right")
        for r in self.rows:
            mae = r.get("best_mae")
            t.add_row(
                r["label"],
                r["status"],
                str(r.get("mols") or "—"),
                str(r.get("specs") or "—"),
                str(r.get("best_r")) if r.get("best_r") is not None else "—",
                f"{mae:.3f}" if mae is not None else "—",
                r.get("elapsed") or "—",
            )
        return Group(t, self.progress)

    def _refresh(self):
        if self._live is not None:
            self._live.update(self._make_panel())

    def __enter__(self):
        self._live = Live(
            self._make_panel(),
            console=self.console,
            refresh_per_second=8,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    # ── presets ──
    def init_presets(self, presets: list[tuple[str, str]]):
        self.rows = [
            {"name": n, "label": lab, "status": "[bright_black]pending[/bright_black]"}
            for n, lab in presets
        ]
        if self._suite_task is None:
            self._suite_task = self.progress.add_task(
                "Suite", total=len(presets), note="",
            )
        else:
            self.progress.update(
                self._suite_task, total=len(presets), completed=0, note="",
            )
        self._refresh()

    def start_preset(self, name: str, n_specs: int):
        for r in self.rows:
            if r["name"] == name:
                r["status"] = "[yellow]running[/yellow]"
                r["specs"] = f"0/{n_specs}"
        if self._spec_task is not None:
            self.progress.remove_task(self._spec_task)
        self._spec_task = self.progress.add_task(
            f"Specs ({name})", total=n_specs, note="",
        )
        self._refresh()

    def status(self, name: str, msg: str, *, mols: int | None = None):
        for r in self.rows:
            if r["name"] == name:
                r["status"] = msg
                if mols is not None:
                    r["mols"] = mols
        self._refresh()

    def advance_spec(self, name: str, current_idx: int, n_specs: int, spec_label: str):
        if self._spec_task is not None:
            self.progress.update(self._spec_task, completed=current_idx, note=spec_label)
        for r in self.rows:
            if r["name"] == name:
                r["specs"] = f"{current_idx}/{n_specs}"
        self._refresh()

    def start_rounds(self, total_steps: int, note: str):
        if self._round_task is not None:
            self.progress.remove_task(self._round_task)
        self._round_task = self.progress.add_task(
            "Rounds", total=total_steps, note=note,
        )

    def tick_round(self, mean: float, std: float):
        if self._round_task is not None:
            self.progress.update(
                self._round_task, advance=1,
                note=f"μ={mean:+.3f}  σ={std:.3f}",
            )

    def set_round_note(self, note: str):
        if self._round_task is not None:
            self.progress.update(self._round_task, note=note)

    def finish_preset(
        self, name: str, *, best_r: int, best_mae: float, elapsed: str,
    ):
        for r in self.rows:
            if r["name"] == name:
                r["status"] = "[green]done[/green]"
                r["best_r"] = best_r
                r["best_mae"] = best_mae
                r["elapsed"] = elapsed
        if self._suite_task is not None:
            self.progress.advance(self._suite_task)
        if self._spec_task is not None:
            self.progress.remove_task(self._spec_task)
            self._spec_task = None
        if self._round_task is not None:
            self.progress.remove_task(self._round_task)
            self._round_task = None
        self._refresh()

    def mark_failed(self, name: str, reason: str):
        for r in self.rows:
            if r["name"] == name:
                r["status"] = f"[red]failed[/red]"
        self.log(f"[red]{name}: {reason}[/red]")
        if self._suite_task is not None:
            self.progress.advance(self._suite_task)
        self._refresh()

    def log(self, msg: str):
        self.console.log(msg)


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:4.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m:d}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m"


# ── per-spec runner ───────────────────────────────────────────────────────

def run_one_spec(
    z: np.ndarray,
    y_truth: np.ndarray,
    smiles: list[str],
    spec: dict,
    spec_idx: int,
    seed_idx: int,
    base_seed: int,
    k: int,
    rounds: int,
    out_dir: Path,
    progress_cb=None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run one (spec, seed). Returns (per-round metrics df, history stack with
    shape (rounds+1, n_molecules); row 0 is y_noisy)."""
    label = spec["label"]
    noise_seed = base_seed + 1000 * spec_idx + seed_idx
    y_noisy = apply_noise_spec(y_truth, smiles, spec["layers"], noise_seed)

    safe_label = label.replace("/", "_").replace(" ", "_")
    log_dir = out_dir / "rounds" / f"{safe_label}_seed{seed_idx}"
    _, history = denoise_mod.iterative_denoise(
        z, y_noisy, k=k, rounds=rounds, log_dir=log_dir, smiles=smiles,
        progress_cb=progress_cb,
    )

    full = [y_noisy] + history
    full_stack = np.stack([np.asarray(y, dtype=float) for y in full])

    rows = []
    for r in range(full_stack.shape[0]):
        m = metrics_for_round(full_stack[r], y_truth)
        step = (
            float("nan") if r == 0
            else float(np.mean(np.abs(full_stack[r] - full_stack[r - 1])))
        )
        rows.append({
            "noise_label": label,
            "noise_type": spec.get("type", ""),
            "noise_scale": spec.get("scale", float("nan")),
            "seed_idx": int(seed_idx),
            "round": int(r),
            **m,
            "mean_step": step,
        })
    return pd.DataFrame(rows), full_stack


# ── plotting ──────────────────────────────────────────────────────────────

def _bold_title(ax, text: str, size: int = 13) -> None:
    ax.set_title(text, fontweight="bold", fontsize=size)


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _agg_by_label_round(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    return results.groupby(["noise_label", "round"])[metric].agg(["mean", "std"]).reset_index()


def _label_colors(labels: list[str]):
    cmap = matplotlib.colormaps["viridis"].resampled(max(len(labels), 2))
    return {lab: cmap(i / max(len(labels) - 1, 1)) for i, lab in enumerate(labels)}


def _optimal_round_table(results: pd.DataFrame) -> pd.DataFrame:
    agg = _agg_by_label_round(results, "mae")
    rows = []
    for label, sub in agg.groupby("noise_label"):
        sub = sub.sort_values("round").reset_index(drop=True)
        i = int(sub["mean"].idxmin())
        rows.append({
            "noise_label": label,
            "optimal_round": int(sub.loc[i, "round"]),
            "mae_at_optimum": float(sub.loc[i, "mean"]),
            "mae_at_round_0": float(sub.iloc[0]["mean"]),
            "mae_at_last_round": float(sub.iloc[-1]["mean"]),
        })
    return pd.DataFrame(rows)


def plot_mae_vs_round(results: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    agg = _agg_by_label_round(results, "mae")
    colors = _label_colors(labels)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for label in labels:
        sub = agg[agg["noise_label"] == label].sort_values("round")
        c = colors[label]
        ax.plot(sub["round"], sub["mean"], "-o", color=c, ms=4, label=label)
        if sub["std"].notna().any():
            ax.fill_between(sub["round"],
                            sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                            color=c, alpha=0.15, linewidth=0)
        # baseline at round 0 (faint reference)
        r0 = float(sub.iloc[0]["mean"])
        ax.axhline(r0, color=c, linestyle=":", linewidth=0.8, alpha=0.45)
        opt = int(sub["mean"].idxmin())
        ax.plot(sub.loc[opt, "round"], sub.loc[opt, "mean"],
                marker="*", color=c, ms=14, mec="black", mew=0.6)

    ax.set_xlabel("Denoising round")
    ax.set_ylabel("MAE vs. ground truth  (kcal/mol)")
    _bold_title(ax, "MAE vs. denoising round")
    ax.legend(title="Noise spec", frameon=False, loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "01_mae_vs_round")


def plot_r2_vs_round(results: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    agg = _agg_by_label_round(results, "r2")
    colors = _label_colors(labels)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for label in labels:
        sub = agg[agg["noise_label"] == label].sort_values("round")
        c = colors[label]
        ax.plot(sub["round"], sub["mean"], "-o", color=c, ms=4, label=label)
        if sub["std"].notna().any():
            ax.fill_between(sub["round"],
                            sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                            color=c, alpha=0.15, linewidth=0)

    ax.set_xlabel("Denoising round")
    ax.set_ylabel("R² vs. ground truth")
    _bold_title(ax, "R² vs. denoising round")
    ax.legend(title="Noise spec", frameon=False, loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "02_r2_vs_round")


def plot_bias_variance(results: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    bias_agg = _agg_by_label_round(
        results.assign(absbias=results["bias"].abs()), "absbias",
    )
    var_agg = _agg_by_label_round(results, "residual_std")

    n = len(labels)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.8 * rows),
                             sharex=True, squeeze=False)
    flat = axes.flatten()
    for ax in flat[n:]:
        ax.axis("off")

    for ax, label in zip(flat, labels):
        sub_b = bias_agg[bias_agg["noise_label"] == label].sort_values("round")
        sub_v = var_agg[var_agg["noise_label"] == label].sort_values("round")
        ln1 = ax.plot(sub_b["round"], sub_b["mean"], "-",
                      color="#c0392b", label="|bias|")
        ax.set_ylabel("|bias|", color="#c0392b")
        ax.tick_params(axis="y", labelcolor="#c0392b")
        ax2 = ax.twinx()
        ln2 = ax2.plot(sub_v["round"], sub_v["mean"], "--",
                       color="#2c3e50", label="residual std")
        ax2.set_ylabel("residual std", color="#2c3e50")
        ax2.tick_params(axis="y", labelcolor="#2c3e50")
        ax.set_xlabel("Round")
        _bold_title(ax, label, size=11)
        ax.grid(alpha=0.3)
        if ax is flat[0]:
            lns = ln1 + ln2
            ax.legend(lns, [l.get_label() for l in lns],
                      frameon=False, loc="upper right", fontsize=8)

    fig.suptitle("Bias–variance trade-off across denoising rounds",
                 fontweight="bold", fontsize=14, y=1.01)
    _save(fig, out_dir, "03_bias_variance")


def plot_residual_violins(
    history_by_label: dict[str, np.ndarray],
    y_truth: np.ndarray,
    results: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
) -> None:
    # Pick the "middle" spec for visual clarity.
    label = labels[len(labels) // 2]
    history = history_by_label[label]
    R = history.shape[0] - 1
    sub = _agg_by_label_round(results[results["noise_label"] == label], "mae")
    optimal_round = int(sub.sort_values("round").reset_index(drop=True)["mean"].idxmin())

    rounds_to_show = sorted({0, 1, optimal_round, R})
    data = [history[r] - y_truth for r in rounds_to_show]
    xlabels = [f"r={r}" for r in rounds_to_show]

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for body in parts["bodies"]:
        body.set_facecolor("#3498db")
        body.set_alpha(0.65)
        body.set_edgecolor("black")
    ax.set_xticks(range(1, len(xlabels) + 1))
    ax.set_xticklabels(xlabels)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Denoising round")
    ax.set_ylabel("Residual: ŷ − y_truth  (kcal/mol)")
    _bold_title(ax, f"Residual distribution across rounds  ({label})")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, out_dir, "04_residual_violins")


def plot_truth_vs_pred_facets(
    history_by_label: dict[str, np.ndarray],
    y_truth: np.ndarray,
    results: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
) -> None:
    n_rows = len(labels)
    col_specs = ["round_0", "optimal", "last"]
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.2 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    lim_lo = float(np.min(y_truth))
    lim_hi = float(np.max(y_truth))
    pad = 0.05 * (lim_hi - lim_lo)
    lim_lo -= pad
    lim_hi += pad

    for i, label in enumerate(labels):
        history = history_by_label[label]
        R = history.shape[0] - 1
        sub = _agg_by_label_round(results[results["noise_label"] == label], "mae")
        sub = sub.sort_values("round").reset_index(drop=True)
        optimal_round = int(sub["mean"].idxmin())
        round_for_col = {"round_0": 0, "optimal": optimal_round, "last": R}

        for j, spec in enumerate(col_specs):
            ax = axes[i, j]
            r = round_for_col[spec]
            y_pred = history[r]
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                    color="black", linewidth=0.7, alpha=0.6)
            ax.scatter(y_truth, y_pred, s=3, alpha=0.25, color="#2980b9",
                       edgecolors="none")
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.grid(alpha=0.2)
            if i == 0:
                titles = {
                    "round_0": "Round 0 (noisy)",
                    "optimal": "Optimal round",
                    "last": f"Round {R} (last)",
                }
                _bold_title(ax, titles[spec], size=11)
            if j == 0:
                ax.set_ylabel(f"{label}\nŷ", fontweight="bold", fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("y_truth")
            if spec == "optimal":
                ax.text(0.04, 0.96, f"r={optimal_round}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=9, color="black",
                        bbox=dict(facecolor="white", alpha=0.7,
                                  edgecolor="none", boxstyle="round,pad=0.2"))

    fig.suptitle("Truth vs. prediction across rounds and noise specs",
                 fontweight="bold", fontsize=14, y=1.00)
    _save(fig, out_dir, "05_truth_vs_pred_facets")


def plot_optimal_round_table(results: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    table = _optimal_round_table(results).set_index("noise_label").loc[labels].reset_index()
    fig, ax1 = plt.subplots(figsize=(max(7.5, 1.0 * len(labels) + 3), 5.0))
    x = np.arange(len(labels))

    color1 = "#1f77b4"
    color2 = "#d62728"
    ax1.bar(x, table["optimal_round"], width=0.65, color=color1, alpha=0.75,
            label="Optimal round")
    ax1.set_xticks(x)
    ax1.set_xticklabels(table["noise_label"], rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Optimal round (argmin MAE)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(x, table["mae_at_optimum"], "o-", color=color2,
             ms=7, label="MAE at optimum")
    ax2.set_ylabel("MAE at optimum  (kcal/mol)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    _bold_title(ax1, "Optimal round and MAE per noise spec")
    ax1.grid(alpha=0.3, axis="y")
    _save(fig, out_dir, "06_optimal_round_per_spec")


def plot_convergence_step(results: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    agg = _agg_by_label_round(results, "mean_step")
    agg = agg[agg["round"] >= 1]
    colors = _label_colors(labels)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for label in labels:
        sub = agg[agg["noise_label"] == label].sort_values("round")
        ax.plot(sub["round"], sub["mean"], "-o", color=colors[label], ms=4, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Denoising round")
    ax.set_ylabel("Mean |y_r − y_{r−1}|  (kcal/mol, log scale)")
    _bold_title(ax, "Per-round movement (convergence vs. over-smoothing)")
    ax.legend(title="Noise spec", frameon=False, loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    _save(fig, out_dir, "07_convergence_step")


def make_plots(
    results: pd.DataFrame,
    history_by_label: dict[str, np.ndarray],
    y_truth: np.ndarray,
    labels: list[str],
    out_dir: Path,
) -> None:
    plot_mae_vs_round(results, labels, out_dir)
    plot_r2_vs_round(results, labels, out_dir)
    plot_bias_variance(results, labels, out_dir)
    plot_residual_violins(history_by_label, y_truth, results, labels, out_dir)
    plot_truth_vs_pred_facets(history_by_label, y_truth, results, labels, out_dir)
    plot_optimal_round_table(results, labels, out_dir)
    plot_convergence_step(results, labels, out_dir)


# ── orchestrator ──────────────────────────────────────────────────────────

def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _sanity_table(results: pd.DataFrame, labels: list[str]) -> Table:
    t = Table(title="Per-spec summary", title_style="bold magenta", show_lines=False)
    t.add_column("Noise spec", style="cyan", no_wrap=True)
    t.add_column("r0 MAE", justify="right")
    t.add_column("Opt r", justify="right")
    t.add_column("Opt MAE", justify="right")
    t.add_column("Last MAE", justify="right")
    t.add_column("Bias@0", justify="right")
    t.add_column("Trend")
    for label in labels:
        sub = results[results["noise_label"] == label]
        agg_mae = sub.groupby("round")["mae"].mean()
        agg_bias = sub.groupby("round")["bias"].mean()
        rounds_sorted = sorted(agg_mae.index)
        agg_mae = agg_mae.loc[rounds_sorted]
        agg_bias = agg_bias.loc[rounds_sorted]
        opt_round = int(agg_mae.idxmin())
        opt_mae = float(agg_mae.loc[opt_round])
        r0_mae = float(agg_mae.iloc[0])
        last_mae = float(agg_mae.iloc[-1])
        trend = (
            "[red]over-denoise[/red]" if last_mae > opt_mae * 1.05
            else "[green]stable[/green]"
        )
        t.add_row(
            label,
            f"{r0_mae:.3f}",
            str(opt_round),
            f"{opt_mae:.3f}",
            f"{last_mae:.3f}",
            f"{float(agg_bias.iloc[0]):+.3f}",
            trend,
        )
    return t


def run_benchmark(
    *,
    noise_specs: list[dict] | None = None,
    sigmas: list[float] | None = None,  # legacy convenience
    ga_test_csv: Path | str = DEFAULT_GA_TEST_CSV,
    ga_fit_csv: Path | str = DEFAULT_GA_FIT_CSV,
    embedder_dir: Path | str = DEFAULT_EMBEDDER_DIR,
    rounds: int = 30,
    k: int = 10,
    n_sample: int = 100000,
    n_seeds: int = 3,
    seed: int = 42,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    batch_size: int = 128,
    run_label: str | None = None,
    make_plots_flag: bool = True,
    cached_zyt: tuple[np.ndarray, np.ndarray, list[str]] | None = None,
    ui: "_SuiteUI | None" = None,
) -> dict[str, Any]:
    """Run one benchmark over a set of noise specs.

    ``noise_specs`` is a list of dicts each shaped like::
        {"label": "normal σ=5", "type": "normal", "scale": 5.0,
         "layers": [{"type": "normal", "scale": 5.0}]}

    If ``noise_specs`` is None, build one from ``sigmas`` (default Gaussian sweep).

    ``cached_zyt`` is an optional pre-computed ``(z, y_truth, smiles)`` triple
    so the suite runner can avoid re-embedding when consecutive presets share
    the same n_sample.
    """
    if noise_specs is None:
        if sigmas is None:
            sigmas = [1.0, 2.0, 5.0, 10.0, 20.0]
        noise_specs = [
            {"label": f"normal σ={s:g}", "type": "normal", "scale": float(s),
             "layers": [{"type": "normal", "scale": float(s)}]}
            for s in sigmas
        ]

    ga_test_csv = Path(ga_test_csv)
    ga_fit_csv = Path(ga_fit_csv)
    embedder_dir = Path(embedder_dir)
    base_out_dir = Path(out_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_label}" if run_label else ""
    run_dir = base_out_dir / f"run{tag}_{timestamp}"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)

    weights_path = embedder_dir / "embedder.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Embedder weights not found at {weights_path}. Train the embedder first."
        )
    hparams_path = embedder_dir / "embedder_hparams.json"
    hparams = None
    if hparams_path.exists():
        try:
            with open(hparams_path) as f:
                hparams = json.load(f)
        except Exception:
            hparams = None

    if not ga_test_csv.exists():
        raise FileNotFoundError(f"GA test CSV not found: {ga_test_csv}")
    if not ga_fit_csv.exists():
        raise FileNotFoundError(f"GA fit CSV not found: {ga_fit_csv}")

    config = {
        "ga_test_csv": str(ga_test_csv),
        "ga_fit_csv": str(ga_fit_csv),
        "embedder_dir": str(embedder_dir),
        "noise_specs": [
            {kk: vv for kk, vv in s.items() if kk != "layers"} | {"layers": s["layers"]}
            for s in noise_specs
        ],
        "rounds": rounds,
        "k": k,
        "n_sample": n_sample,
        "n_seeds": n_seeds,
        "seed": seed,
        "batch_size": batch_size,
        "timestamp": timestamp,
        "git_sha": _git_sha(),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # If called outside a suite, spin up a one-row UI for visual consistency.
    owns_ui = ui is None
    preset_name = run_label or "benchmark"
    if owns_ui:
        ui = _SuiteUI(title="Benchmark")
        ui.__enter__()
        ui.init_presets([(preset_name, run_label or "(manual)")])
        ui.log(f"[bright_black]run dir:[/bright_black] {run_dir}")

    t0 = time.time()
    try:
        if cached_zyt is not None:
            z, y_truth, smiles = cached_zyt
            ui.status(preset_name, "[yellow]reusing embeddings[/yellow]",
                      mols=len(smiles))
        else:
            ui.status(preset_name, "[yellow]fitting GA model[/yellow]")
            ga_model = ga_mod.get_or_fit_model(
                ga_fit_csv, cache_dir=ga_fit_csv.parent,
            )
            ui.status(preset_name, "[yellow]subsampling truth pool[/yellow]")
            df = load_test_pool(ga_test_csv, n_sample=n_sample, seed=seed)
            ui.status(preset_name, "[yellow]loading embedder[/yellow]")
            model = embedder_mod.load_model(weights_path, hparams=hparams)
            ui.status(preset_name, f"[yellow]embedding {len(df)} mols[/yellow]")
            z, y_truth, smiles = compute_truth_and_embed(
                df, model, ga_model, batch_size,
            )
            ui.status(preset_name, "[yellow]running[/yellow]", mols=len(smiles))

        if len(smiles) < 10:
            raise RuntimeError("Too few molecules survive masking; raise n_sample.")

        pd.DataFrame({"smiles": smiles, "h298": y_truth}).to_csv(
            run_dir / "y_truth.csv", index=False,
        )
        np.save(run_dir / "z.npy", z)

        all_rows: list[pd.DataFrame] = []
        history_by_label: dict[str, np.ndarray] = {}
        labels = [s["label"] for s in noise_specs]

        ui.start_preset(preset_name, len(noise_specs))
        ui.status(preset_name, "[yellow]running[/yellow]", mols=len(smiles))

        for spec_idx, spec in enumerate(noise_specs):
            ui.advance_spec(preset_name, spec_idx, len(noise_specs), spec["label"])
            ui.start_rounds(n_seeds * rounds, f"seed 1/{n_seeds}  {spec['label']}")
            per_seed_histories: list[np.ndarray] = []
            for seed_idx in range(n_seeds):
                ui.set_round_note(f"seed {seed_idx + 1}/{n_seeds}  {spec['label']}")
                df_rows, full_stack = run_one_spec(
                    z=z, y_truth=y_truth, smiles=smiles,
                    spec=spec, spec_idx=spec_idx, seed_idx=seed_idx,
                    base_seed=seed, k=k, rounds=rounds, out_dir=run_dir,
                    progress_cb=lambda r, R, m, s: ui.tick_round(m, s),
                )
                all_rows.append(df_rows)
                per_seed_histories.append(full_stack)
            history_by_label[spec["label"]] = np.mean(
                np.stack(per_seed_histories, axis=0), axis=0,
            )
        ui.advance_spec(preset_name, len(noise_specs), len(noise_specs), "—")

        results = pd.concat(all_rows, ignore_index=True)
        results.to_csv(run_dir / "results.csv", index=False)

        optimal_table = _optimal_round_table(results)
        optimal_table.to_csv(run_dir / "optimal_rounds.csv", index=False)

        if make_plots_flag:
            ui.status(preset_name, "[yellow]writing plots[/yellow]", mols=len(smiles))
            make_plots(results, history_by_label, y_truth, labels, plots_dir)

        # compute best across all specs in this preset
        opt = _optimal_round_table(results)
        best_idx = opt["mae_at_optimum"].idxmin()
        best_r = int(opt.loc[best_idx, "optimal_round"])
        best_mae = float(opt.loc[best_idx, "mae_at_optimum"])
        ui.finish_preset(
            preset_name,
            best_r=best_r, best_mae=best_mae,
            elapsed=_fmt_elapsed(time.time() - t0),
        )

        if owns_ui:
            ui.console.print(_sanity_table(results, labels))
            ui.console.print(f"[green]→[/green] {run_dir}")
    finally:
        if owns_ui and ui is not None:
            ui.__exit__(None, None, None)

    return {
        "run_dir": str(run_dir),
        "results_csv": str(run_dir / "results.csv"),
        "optimal_rounds_csv": str(run_dir / "optimal_rounds.csv"),
        "plots_dir": str(plots_dir) if make_plots_flag else None,
        "n_molecules": len(smiles),
        "z": z,
        "y_truth": y_truth,
        "smiles": smiles,
        "results": results,
        "labels": labels,
    }


def run_suite(
    preset_names: list[str],
    *,
    overrides: dict[str, dict] | None = None,
    train_config: dict | None = None,
    force_retrain: bool = False,
    ga_train_csv: Path | str = DEFAULT_GA_TRAIN_CSV,
    ga_test_csv: Path | str = DEFAULT_GA_TEST_CSV,
    ga_fit_csv: Path | str = DEFAULT_GA_FIT_CSV,
    embedder_dir: Path | str = DEFAULT_EMBEDDER_DIR,
    seed: int = 42,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    batch_size: int = 128,
) -> list[dict]:
    """Run a list of preset names sequentially.

    ``overrides[name]`` may patch any of {n_sample, rounds, n_seeds, k} for
    that preset. Embeddings are re-used across consecutive presets when
    n_sample matches.

    ``train_config`` controls embedder (re)training before the suite runs.
    Shape: ``{"n_train": int, "epochs": int, "batch_size": int}``. If the
    saved embedder's provenance doesn't match this config (or ``force_retrain``
    is set), a fresh embedder is trained on a stratified subsample of
    ``ga_train_csv`` (which must be disjoint from ``ga_test_csv`` for the
    held-out evaluation to be meaningful).
    """
    overrides = overrides or {}
    summaries: list[dict] = []
    last_n_sample: int | None = None
    cached_zyt: tuple | None = None

    # Filter to known presets first so the UI table reflects what'll actually run.
    valid = [n for n in preset_names if n in PRESETS]
    if not valid:
        _console.print("[red]No valid presets selected.[/red]")
        return summaries

    # ── Step 1/2: train embedder if needed ──
    if train_config is None:
        train_config = {"n_train": 100_000, "epochs": 30, "batch_size": 128}
    n_train = int(train_config["n_train"])
    epochs = int(train_config["epochs"])
    train_batch = int(train_config["batch_size"])

    needs = force_retrain or needs_retraining(
        embedder_dir,
        n_train=n_train, epochs=epochs, batch_size=train_batch,
        ga_train_csv=ga_train_csv,
    )
    _console.print()
    _console.rule(
        f"[bold cyan]Step 1/2 · Embedder training "
        f"({'fresh train' if needs else 'reuse'})[/bold cyan]"
    )
    if needs:
        try:
            train_embedder_for_benchmark(
                ga_train_csv=ga_train_csv,
                embedder_dir=embedder_dir,
                n_train=n_train,
                epochs=epochs,
                batch_size=train_batch,
                seed=seed,
            )
        except KeyboardInterrupt:
            _console.print(
                "\n[yellow]Embedder training cancelled by user (Ctrl-C). "
                "Aborting suite.[/yellow]"
            )
            return summaries
    else:
        cfg_saved = load_train_config(embedder_dir)
        _console.print(
            f"[green]✓[/green] embedder already trained at this config "
            f"(n_train={cfg_saved['n_train']:,}  epochs={cfg_saved['epochs']}  "
            f"batch={cfg_saved['batch_size']}, on {cfg_saved.get('trained_at')})"
        )

    # ── Step 2/2: run benchmarks ──
    _console.print()
    _console.rule(
        f"[bold cyan]Step 2/2 · Benchmark suite "
        f"({len(valid)} preset(s))[/bold cyan]"
    )

    cancelled = False
    with _SuiteUI(title=f"Benchmark suite  ({len(valid)} preset(s))") as ui:
        ui.init_presets([(n, PRESETS[n]["label"]) for n in valid])

        try:
            for name in valid:
                cfg = copy.deepcopy(PRESETS[name])
                cfg.update(overrides.get(name, {}))
                n_sample = int(cfg["n_sample"])
                zyt = cached_zyt if last_n_sample == n_sample else None
                if zyt is not None:
                    ui.log(f"[cyan]{name}[/cyan]: reusing cached embeddings "
                           f"(n={len(zyt[2])})")

                try:
                    summary = run_benchmark(
                        noise_specs=cfg["noise_specs"],
                        ga_test_csv=ga_test_csv,
                        ga_fit_csv=ga_fit_csv,
                        embedder_dir=embedder_dir,
                        rounds=int(cfg["rounds"]),
                        k=int(cfg["k"]),
                        n_sample=n_sample,
                        n_seeds=int(cfg["n_seeds"]),
                        seed=seed,
                        out_dir=out_dir,
                        batch_size=batch_size,
                        run_label=name,
                        cached_zyt=zyt,
                        ui=ui,
                    )
                except KeyboardInterrupt:
                    ui.mark_failed(name, "cancelled by user (Ctrl-C)")
                    cancelled = True
                    break
                except Exception as e:
                    ui.mark_failed(name, str(e))
                    continue

                cached_zyt = (summary["z"], summary["y_truth"], summary["smiles"])
                last_n_sample = n_sample
                ui.log(f"[green]{name}[/green] → {summary['run_dir']}")
                summaries.append(
                    {k: v for k, v in summary.items()
                     if k in {"run_dir", "results_csv",
                              "optimal_rounds_csv", "plots_dir",
                              "n_molecules", "results", "labels"}}
                    | {"preset": name, "label": PRESETS[name]["label"]}
                )
        except KeyboardInterrupt:
            cancelled = True

    if cancelled:
        _console.print(
            "\n[yellow]Suite cancelled by user (Ctrl-C). "
            f"{len(summaries)} preset(s) completed before cancel.[/yellow]"
        )

    # Past the live context: print one sanity table per preset for inspection.
    for s in summaries:
        _console.print()
        _console.rule(f"[bold magenta]{s['label']}[/bold magenta]")
        _console.print(_sanity_table(s["results"], s["labels"]))
        _console.print(f"[bright_black]→[/bright_black] {s['run_dir']}")

    return summaries


# ── CLI ───────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark iterative k-NN denoising vs GA ground truth.",
    )
    p.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                   help="run a single named preset (overrides --sigmas/--rounds/etc.)")
    p.add_argument("--suite", nargs="+", choices=list(PRESETS.keys()),
                   default=None, help="run multiple presets in sequence")
    p.add_argument("--ga-test-csv", default=str(DEFAULT_GA_TEST_CSV))
    p.add_argument("--ga-train-csv", default=str(DEFAULT_GA_TRAIN_CSV),
                   help="GA pool for embedder training (must be disjoint from test)")
    p.add_argument("--ga-fit-csv", default=str(DEFAULT_GA_FIT_CSV))
    p.add_argument("--embedder-dir", default=str(DEFAULT_EMBEDDER_DIR))
    p.add_argument("--train-n", type=int, default=100_000,
                   help="embedder training-set size (stratified by h298)")
    p.add_argument("--train-epochs", type=int, default=30)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--retrain", action="store_true",
                   help="force-retrain the embedder even if provenance matches")
    p.add_argument("--train-only", action="store_true",
                   help="only train the embedder, then exit")
    p.add_argument("--sigmas", nargs="+", type=float,
                   default=[1.0, 2.0, 5.0, 10.0, 20.0])
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-sample", type=int, default=100000)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    train_cfg = {
        "n_train": int(args.train_n),
        "epochs": int(args.train_epochs),
        "batch_size": int(args.train_batch_size),
    }

    if args.train_only:
        train_embedder_for_benchmark(
            ga_train_csv=args.ga_train_csv,
            embedder_dir=args.embedder_dir,
            n_train=train_cfg["n_train"],
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            seed=args.seed,
        )
        return

    if args.suite:
        run_suite(
            args.suite,
            train_config=train_cfg,
            force_retrain=args.retrain,
            ga_train_csv=args.ga_train_csv,
            ga_test_csv=args.ga_test_csv,
            ga_fit_csv=args.ga_fit_csv,
            embedder_dir=args.embedder_dir,
            seed=args.seed,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
        )
        return

    # Single-preset / manual paths optionally retrain too.
    needs = args.retrain or needs_retraining(
        args.embedder_dir,
        n_train=train_cfg["n_train"],
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        ga_train_csv=args.ga_train_csv,
    )
    if needs:
        train_embedder_for_benchmark(
            ga_train_csv=args.ga_train_csv,
            embedder_dir=args.embedder_dir,
            n_train=train_cfg["n_train"],
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            seed=args.seed,
        )

    if args.preset:
        cfg = PRESETS[args.preset]
        run_benchmark(
            noise_specs=cfg["noise_specs"],
            ga_test_csv=args.ga_test_csv,
            ga_fit_csv=args.ga_fit_csv,
            embedder_dir=args.embedder_dir,
            rounds=int(cfg["rounds"]),
            k=int(cfg["k"]),
            n_sample=int(cfg["n_sample"]),
            n_seeds=int(cfg["n_seeds"]),
            seed=args.seed,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            run_label=args.preset,
            make_plots_flag=not args.no_plots,
        )
        return

    run_benchmark(
        sigmas=args.sigmas,
        ga_test_csv=args.ga_test_csv,
        ga_fit_csv=args.ga_fit_csv,
        embedder_dir=args.embedder_dir,
        rounds=args.rounds,
        k=args.k,
        n_sample=args.n_sample,
        n_seeds=args.n_seeds,
        seed=args.seed,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        make_plots_flag=not args.no_plots,
    )


if __name__ == "__main__":
    main()
