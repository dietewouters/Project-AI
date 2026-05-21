#!/usr/bin/env python3
"""
Regenerate benchmark plots from existing run outputs.

Reads each run dir's config.json + y_truth.csv + rounds/*/round_*.csv and
re-applies the (deterministic, seeded) noise spec to reconstruct round 0.
No GA fitting, no embedder, no denoising — just plotting.

Usage:
    python -m src.replot_benchmarks                # all runs with results.csv
    python -m src.replot_benchmarks <run_dir> ...  # specific run dirs
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import benchmark_denoise as bd
import noise as noise_mod


def _safe_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_")


def _load_round_stack(
    rounds_dir: Path,
    smiles: list[str],
    rounds_total: int,
) -> np.ndarray:
    """Load round_1.csv … round_R.csv → array shape (R, N) aligned to smiles."""
    n = len(smiles)
    out = np.empty((rounds_total, n), dtype=float)
    for r in range(1, rounds_total + 1):
        df = pd.read_csv(rounds_dir / f"round_{r}.csv")
        # Saved order matches y_truth.csv, but verify and re-align if needed.
        if "smiles" in df.columns and df["smiles"].tolist() != smiles:
            df = df.set_index("smiles").loc[smiles].reset_index()
        out[r - 1] = df["y"].to_numpy(dtype=float)
    return out


def replot_run(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    truth_path = run_dir / "y_truth.csv"
    results_path = run_dir / "results.csv"
    if not (cfg_path.exists() and truth_path.exists() and results_path.exists()):
        print(f"  skip {run_dir.name}: missing config/y_truth/results")
        return

    with open(cfg_path) as f:
        cfg = json.load(f)

    truth_df = pd.read_csv(truth_path)
    smiles = truth_df["smiles"].astype(str).tolist()
    y_truth = truth_df["h298"].to_numpy(dtype=float)
    results = pd.read_csv(results_path)

    rounds_total = int(cfg["rounds"])
    n_seeds = int(cfg["n_seeds"])
    base_seed = int(cfg["seed"])
    if "noise_specs" in cfg:
        noise_specs = cfg["noise_specs"]
    elif "sigmas" in cfg:
        noise_specs = [
            {"label": f"normal σ={s:g}", "type": "normal", "scale": float(s),
             "layers": [{"type": "normal", "scale": float(s)}]}
            for s in cfg["sigmas"]
        ]
    else:
        print(f"  skip {run_dir.name}: config has neither noise_specs nor sigmas")
        return
    labels = [s["label"] for s in noise_specs]

    history_by_label: dict[str, np.ndarray] = {}
    for spec_idx, spec in enumerate(noise_specs):
        label = spec["label"]
        per_seed_stacks: list[np.ndarray] = []
        for seed_idx in range(n_seeds):
            noise_seed = base_seed + 1000 * spec_idx + seed_idx
            noise_vec = noise_mod.apply_layers(
                y_truth, smiles, spec["layers"], base_seed=noise_seed,
            )
            y_noisy = y_truth + noise_vec
            rounds_dir = run_dir / "rounds" / f"{_safe_label(label)}_seed{seed_idx}"
            if not rounds_dir.exists():
                print(f"  warn {run_dir.name}: missing {rounds_dir.name}")
                continue
            rest = _load_round_stack(rounds_dir, smiles, rounds_total)
            per_seed_stacks.append(np.vstack([y_noisy[None, :], rest]))
        if not per_seed_stacks:
            print(f"  skip {run_dir.name}: no seed data for {label}")
            return
        history_by_label[label] = np.mean(np.stack(per_seed_stacks, axis=0), axis=0)

    plots_dir = run_dir / "plots"
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    z_path = run_dir / "z.npy"
    z = np.load(z_path) if z_path.exists() else None

    bd.make_plots(
        results=results,
        history_by_label=history_by_label,
        y_truth=y_truth,
        labels=labels,
        out_dir=plots_dir,
        z=z,
        seed=base_seed,
    )
    print(f"  ✓ {run_dir.name}  ({len(labels)} specs, n={len(smiles)})")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        targets = [Path(a) for a in argv]
    else:
        bench_root = bd.DEFAULT_OUT_DIR
        targets = sorted(
            d for d in bench_root.iterdir()
            if d.is_dir() and (d / "results.csv").exists()
        )

    print(f"Replotting {len(targets)} run(s)…")
    for d in targets:
        try:
            replot_run(d)
        except Exception as e:
            print(f"  ✗ {d.name}: {e!r}")


if __name__ == "__main__":
    main()
