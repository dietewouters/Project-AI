"""
Inverse-distance-weighted k-NN denoising in embedding space.

Given embeddings ``z`` for a set of molecules and a noisy target value
``y`` for each, one *round* of denoising replaces every entry with the
inverse-distance-weighted mean of its ``k`` nearest neighbours (excluding
itself). Repeating this for ``rounds`` rounds is, conceptually, message
passing between molecules — every molecule's value converges toward the
local average of chemically-similar peers.

This is deliberately a tiny module: just three functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def idw_knn_denoise_round(
    z: np.ndarray,
    y: np.ndarray,
    k: int,
    eps: float = 1e-9,
) -> np.ndarray:
    """One full pass of self-denoising over the dataset.

    For every row ``i``:
      1. Find the ``k`` nearest rows of ``z`` (excluding ``i`` itself).
      2. Replace ``y[i]`` with the inverse-distance-weighted mean of
         ``y`` at those neighbours: ``w = 1 / (d + eps)``.

    Returns a new ``y`` array of the same shape as the input.
    """
    from scipy.spatial import cKDTree

    n = len(z)
    if n == 0:
        return np.array([], dtype=float)
    if y.shape[0] != n:
        raise ValueError(f"len(z)={n} != len(y)={y.shape[0]}")
    k_eff = max(1, min(k, n - 1))

    tree = cKDTree(z)
    # Query k+1 because the closest point is the row itself (distance 0).
    d, idx = tree.query(z, k=k_eff + 1)
    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    # Drop the self-match (column 0).
    d = d[:, 1:]
    idx = idx[:, 1:]

    w = 1.0 / (d + eps)
    w_sum = w.sum(axis=1) + eps
    y_neigh = y[idx]
    return (w * y_neigh).sum(axis=1) / w_sum


def iterative_denoise(
    z: np.ndarray,
    y: np.ndarray,
    k: int,
    rounds: int,
    log_dir: Path | str | None = None,
    smiles: list[str] | None = None,
    progress_cb=None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Apply ``idw_knn_denoise_round`` ``rounds`` times.

    Each round operates on the *previous* round's output, so the values
    drift further from the original noisy input on every pass.

    Returns ``(final_y, history)`` where ``history[i]`` is the y vector at
    the end of round ``i+1`` (so ``history[-1] is final_y``).

    If ``log_dir`` is given, every round is written to
    ``log_dir/round_{i}.csv`` (with the SMILES column if provided).
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")

    log_path = Path(log_dir) if log_dir is not None else None
    if log_path is not None:
        log_path.mkdir(parents=True, exist_ok=True)

    current = np.asarray(y, dtype=float).copy()
    history: list[np.ndarray] = []

    from rich import print as rprint
    for r in range(1, rounds + 1):
        current = idw_knn_denoise_round(z, current, k=k)
        history.append(current.copy())

        if log_path is not None:
            cols = {"y": current}
            if smiles is not None:
                cols = {"smiles": smiles, **cols}
            pd.DataFrame(cols).to_csv(log_path / f"round_{r}.csv", index=False)

        if progress_cb is not None:
            progress_cb(r, rounds, float(current.mean()), float(current.std()))
        else:
            rprint(
                f"  [muted]round {r}/{rounds}  "
                f"mean={current.mean():.4f}  std={current.std():.4f}[/muted]"
            )

    return current, history


def benchmark_against_truth(
    y_pred: np.ndarray,
    y_truth: np.ndarray,
) -> dict:
    """Compute MSE / MAE / RMSE / R² between two aligned arrays."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_truth = np.asarray(y_truth, dtype=float)
    if y_pred.shape != y_truth.shape:
        raise ValueError(
            f"shape mismatch: y_pred={y_pred.shape}, y_truth={y_truth.shape}"
        )
    err = y_pred - y_truth
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    truth_var = float(np.var(y_truth))
    r2 = 1.0 - mse / truth_var if truth_var > 0 else float("nan")
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def write_metrics(metrics: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
