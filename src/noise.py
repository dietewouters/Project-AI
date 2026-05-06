#!/usr/bin/env python3
"""
Noise injection for SMILES → target CSVs.

A `layer` is one additive noise contribution described by a dict:
    {"type": str, "scale": float, "fraction": float (optional), "seed": int (optional)}

Multiple layers are summed. `derive_dataset` reads a clean CSV, optionally
subsamples rows, applies the noise stack to a target column (overwriting it),
and writes the result to a new CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats


NOISE_TYPES: list[str] = [
    "normal",
    "uniform",
    "cosh",
    "bimodal",
    "half_and_half",
    "nitrogen",
    "outlier",
]

NOISE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "normal":         "additive Gaussian, scale = std",
    "uniform":        "uniform on ±√3·scale (std = scale)",
    "cosh":           "cosh-shaped, std≈1; `scale` rescales it",
    "bimodal":        "Gaussian + ±0.5·scale shift",
    "half_and_half":  "scale on positive y, scale/10 on negative",
    "nitrogen":       "scale on N-containing SMILES, scale/10 otherwise",
    "outlier":        "replace fraction of rows with N(0, scale)",
}


class _CoshDist(stats.rv_continuous):
    """Bounded cosh distribution, std=1 for the canonical loc."""

    def __init__(self, loc: float = 1.543404638418213):
        super().__init__(a=-loc, b=loc)
        self._scale_p = 2 * np.sinh(loc)

    def _pdf(self, x):
        return np.cosh(x) / self._scale_p


def generate_layer(
    y: np.ndarray,
    smiles: list[str],
    layer: dict,
) -> np.ndarray:
    """Return additive noise vector matching y.shape for one layer."""
    n = len(y)
    seed = layer.get("seed")
    rng = np.random.default_rng(seed)
    s = float(layer.get("scale", 0.0))
    t = layer["type"]

    if s <= 0 and t != "outlier":
        return np.zeros(n)

    if t == "normal":
        return rng.normal(scale=s, size=n)
    if t == "uniform":
        bound = s * np.sqrt(3)
        return rng.uniform(-bound, bound, size=n)
    if t == "cosh":
        dist = _CoshDist()
        with _seeded_scipy(seed):
            return dist.rvs(size=n) * s
    if t == "bimodal":
        return rng.normal(scale=s * 0.866025, size=n) + rng.choice([-s / 2, s / 2], size=n)
    if t == "half_and_half":
        hi = rng.normal(scale=s, size=n)
        lo = rng.normal(scale=s / 10.0, size=n)
        return np.where(np.asarray(y) > 0, hi, lo)
    if t == "nitrogen":
        has_n = np.array(["N" in str(x).upper() for x in smiles])
        hi = rng.normal(scale=s, size=n)
        lo = rng.normal(scale=s / 10.0, size=n)
        return np.where(has_n, hi, lo)
    if t == "outlier":
        frac = float(layer.get("fraction", 0.05))
        mask = rng.random(n) < frac
        vals = rng.normal(scale=s if s > 0 else 1.0, size=n)
        return np.where(mask, vals, 0.0)
    return np.zeros(n)


class _seeded_scipy:
    """Temporarily seed scipy's global RNG (it has no per-call seed for rv_continuous)."""

    def __init__(self, seed: int | None):
        self.seed = seed
        self._state = None

    def __enter__(self):
        self._state = np.random.get_state()
        if self.seed is not None:
            np.random.seed(self.seed)

    def __exit__(self, *_):
        np.random.set_state(self._state)


def apply_layers(
    y: np.ndarray,
    smiles: list[str],
    layers: list[dict],
    base_seed: int = 42,
) -> np.ndarray:
    """Sum every layer's contribution. Layers without an explicit seed get one
    derived from `base_seed` so repeated runs are reproducible."""
    out = np.zeros_like(y, dtype=float)
    for i, layer in enumerate(layers):
        layer_with_seed = dict(layer)
        if layer_with_seed.get("seed") is None:
            layer_with_seed["seed"] = base_seed + i
        out += generate_layer(y, smiles, layer_with_seed)
    return out


def _slug(layer: dict) -> str:
    abbrev = {
        "normal": "n", "uniform": "u", "cosh": "c", "bimodal": "b",
        "half_and_half": "h", "nitrogen": "N", "outlier": "o",
    }
    a = abbrev.get(layer["type"], layer["type"][:2])
    s = f"{float(layer.get('scale', 0)):.3g}"
    if layer["type"] == "outlier":
        f = f"x{float(layer.get('fraction', 0.05)):.2g}"
        return f"{a}{s}{f}"
    return f"{a}{s}"


def derived_filename(
    src_csv: str | Path,
    fraction: float,
    layers: list[dict],
    seed: int,
) -> str:
    parts = [Path(src_csv).stem]
    if fraction is not None and fraction < 1.0:
        parts.append(f"f{fraction:.3g}")
    for layer in layers or []:
        parts.append(_slug(layer))
    parts.append(f"s{seed}")
    name = "__".join(parts) + ".csv"
    if len(name) > 180:
        name = name[:170] + "__trunc.csv"
    return name


def derive_dataset(
    src_csv: str | Path,
    dst_csv: str | Path,
    smiles_col: str,
    target_col: str,
    fraction: float = 1.0,
    layers: list[dict] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Read src, optionally subsample, optionally inject noise into target_col,
    write to dst. The destination preserves the source schema."""
    df = pd.read_csv(src_csv)
    for col in (smiles_col, target_col):
        if col not in df.columns:
            raise ValueError(
                f"column {col!r} not found in {src_csv} (have {list(df.columns)})"
            )

    if 0 < fraction < 1.0:
        df = df.sample(frac=fraction, random_state=seed).reset_index(drop=True)

    if layers:
        smiles = df[smiles_col].astype(str).tolist()
        y = df[target_col].astype(float).to_numpy()
        noise = apply_layers(y, smiles, layers, base_seed=seed)
        df[target_col] = y + noise

    Path(dst_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_csv, index=False)
    return df
