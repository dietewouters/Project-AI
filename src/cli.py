#!/usr/bin/env python3
"""
MolAugment — denoise via embedded k-NN.

A small interactive CLI. The core flow is:

    1. Train a Chemprop encoder on a clean group-additivity dataset.
    2. Use that encoder to embed the molecules of a *noisy* CSV.
    3. Run R rounds of inverse-distance-weighted k-NN denoising over
       the embeddings.
    4. Compare the denoised values against a ground-truth CSV.

Run:
    python src/cli.py
"""

from __future__ import annotations

import copy
import curses
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

# Suppress PyTorch Lightning internal deprecation warnings.
warnings.filterwarnings(
    "ignore", message=".*LeafSpec.*deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore", message=".*LeafSpec.*deprecated.*",
    category=UserWarning,
)

# Make sibling modules importable when run as a script.
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _pip_install(pkg: str):
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


for _pkg in ("rich", "pandas", "scipy", "numpy"):
    try:
        __import__(_pkg)
    except ImportError:
        print(f"Installing '{_pkg}' …")
        _pip_install(_pkg)

import numpy as np
import pandas as pd

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


# ── theme ─────────────────────────────────────────────────────────────────

THEME = Theme({
    "brand":   "bold cyan",
    "accent":  "bold magenta",
    "success": "bold green",
    "warn":    "bold yellow",
    "err":     "bold red",
    "muted":   "dim white",
    "hi":      "bold white",
})
console = Console(theme=THEME)
VERSION = "4.0.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(__file__).resolve().parent / "cli_workspace"
STATE_FILE = WORKSPACE / ".state.json"


# ── default state ─────────────────────────────────────────────────────────

DEFAULT_GA_DIR = REPO_ROOT / "groupadditivity_h298" / "dataset"
DEFAULT_STATE: dict[str, Any] = {
    "embedder": {
        "train_csv":   str(DEFAULT_GA_DIR / "groupadditivity_1.csv"),
        "val_csv":     str(DEFAULT_GA_DIR / "groupadditivity_secondarytest.csv"),
        "test_csv":    str(DEFAULT_GA_DIR / "groupadditivity_test.csv"),
        "smiles_col":  "smiles",
        "target_col":  "h298",
        "model_dir":   str(WORKSPACE / "embedder"),
        "trained":     False,
    },
    "noisy": {
        "csv":         None,
        "smiles_col":  "smiles",
        "target_col":  "h298_noisy",
    },
    "truth": {
        "csv":         None,
        "smiles_col":  "smiles",
        "target_col":  "h298",
    },
    "knn": {
        "k":           10,
        "rounds":      5,
    },
    "train": {
        "epochs":      30,
        "batch_size":  128,
        "seed":        42,
    },
    "hpo": {
        "use_tuned":   False,
        "n_trials":    20,
        "hpo_epochs":  10,
        "best_params": None,
    },
    "last_run": {
        "denoised_csv": None,
        "metrics":      None,
    },
}


def load_state() -> dict:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                st = json.load(f)
            for k, v in DEFAULT_STATE.items():
                if k not in st:
                    st[k] = copy.deepcopy(v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if kk not in st[k]:
                            st[k][kk] = copy.deepcopy(vv)
            return st
        except Exception:
            pass
    return copy.deepcopy(DEFAULT_STATE)


def save_state(state: dict) -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── rich UI helpers ───────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    art = Text.from_markup(
        f"\n  [brand]MolAugment[/brand]  [muted]v{VERSION}[/muted]\n"
        "  [muted]Train an embedder, then denoise a noisy CSV via k-NN[/muted]\n"
    )
    console.print(Panel(
        art, border_style="cyan", box=box.DOUBLE_EDGE,
        expand=False, padding=(0, 2),
    ))


def section(title: str):
    console.print()
    console.print(Rule(f"[accent] {title} [/accent]", style="magenta"))
    console.print()


def info(msg: str):    console.print(f"  [muted]--[/muted]  {msg}")
def success(msg: str): console.print(f"  [success]OK[/success]  {msg}")
def warn(msg: str):    console.print(f"  [warn]!![/warn]  {msg}")
def error(msg: str):   console.print(f"  [err]ERR[/err] {msg}")


def pause():
    console.print()
    Prompt.ask("  [muted]Press Enter to continue[/muted]", default="", show_default=False)


# ── arrow-key list selector (curses) ──────────────────────────────────────

def curses_select(items: list[str], title: str = "", subtitle: str = "") -> int:
    """Arrow-key + Enter selector. Returns the chosen index or -1 on Esc/q."""
    def _impl(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        idx = 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            row = 1
            if title:
                stdscr.addnstr(row, 2, title[: w - 4], w - 4, curses.A_BOLD)
                row += 1
            if subtitle:
                stdscr.addnstr(row, 2, subtitle[: w - 4], w - 4, curses.A_DIM)
                row += 1
            row += 1

            for i, item in enumerate(items):
                marker = "▸ " if i == idx else "  "
                line = f"{marker}{item}"
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                if row + i < h - 1:
                    stdscr.addnstr(row + i, 2, line[: w - 4], w - 4, attr)

            footer = "↑/↓ navigate   Enter select   q/Esc quit"
            stdscr.addnstr(h - 1, 2, footer[: w - 4], w - 4, curses.A_DIM)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(items)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(items)
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                return idx
            elif key in (27, ord("q")):  # Esc / q
                return -1

    try:
        return curses.wrapper(_impl)
    except Exception:
        # Fallback to numeric prompt if curses cannot run.
        for i, item in enumerate(items):
            console.print(f"  [{i}] {item}")
        try:
            return IntPrompt.ask("  Choice", default=0)
        except Exception:
            return -1


# ── overview / status ─────────────────────────────────────────────────────

def _short(p: str | None) -> str:
    if not p:
        return "[muted]<not set>[/muted]"
    pp = Path(p)
    try:
        return str(pp.relative_to(REPO_ROOT))
    except Exception:
        return str(pp)


def print_status(state: dict) -> None:
    section("Overview")
    emb = state["embedder"]
    nz = state["noisy"]
    tr = state["truth"]
    knn = state["knn"]
    trn = state["train"]
    hpo = state["hpo"]
    last = state["last_run"]

    tbl = Table(box=box.SIMPLE_HEAVY, expand=False, border_style="cyan")
    tbl.add_column("Section", style="accent", width=18)
    tbl.add_column("Setting", style="muted", width=18)
    tbl.add_column("Value")

    tbl.add_row("Embedder", "train CSV",   _short(emb["train_csv"]))
    tbl.add_row("",         "val CSV",     _short(emb["val_csv"]))
    tbl.add_row("",         "test CSV",    _short(emb["test_csv"]))
    tbl.add_row("",         "smiles col",  emb["smiles_col"])
    tbl.add_row("",         "target col",  emb["target_col"])
    tbl.add_row("",         "trained",     "[success]yes[/success]" if emb["trained"] else "[warn]no[/warn]")

    tbl.add_row("Noisy",    "CSV",         _short(nz["csv"]))
    tbl.add_row("",         "smiles col",  nz["smiles_col"])
    tbl.add_row("",         "y col",       nz["target_col"])

    tbl.add_row("Truth",    "CSV",         _short(tr["csv"]))
    tbl.add_row("",         "smiles col",  tr["smiles_col"])
    tbl.add_row("",         "y col",       tr["target_col"])

    tbl.add_row("k-NN",     "k",           str(knn["k"]))
    tbl.add_row("",         "rounds",      str(knn["rounds"]))

    tbl.add_row("Train",    "epochs",      str(trn["epochs"]))
    tbl.add_row("",         "batch_size",  str(trn["batch_size"]))
    tbl.add_row("",         "seed",        str(trn["seed"]))

    tbl.add_row("HPO",      "use tuned",   "[success]yes[/success]" if hpo["use_tuned"] else "[muted]no[/muted]")
    tbl.add_row("",         "best params", "set" if hpo["best_params"] else "[muted]<none>[/muted]")

    if last["metrics"]:
        m = last["metrics"]
        metrics_str = (
            f"MSE={m.get('mse', float('nan')):.4f}  "
            f"MAE={m.get('mae', float('nan')):.4f}  "
            f"R²={m.get('r2', float('nan')):.4f}"
        )
        tbl.add_row("Last run", "metrics", metrics_str)
        tbl.add_row("",         "denoised", _short(last["denoised_csv"]))
    console.print(tbl)


# ── menu: configure embedder data ─────────────────────────────────────────

def menu_configure_embedder(state: dict) -> None:
    section("Configure embedder training data")
    emb = state["embedder"]
    info(f"Current train: {_short(emb['train_csv'])}")
    info(f"Current val:   {_short(emb['val_csv'])}")
    info(f"Current test:  {_short(emb['test_csv'])}")
    console.print()

    emb["train_csv"]  = Prompt.ask("  Train CSV path", default=emb["train_csv"])
    emb["val_csv"]    = Prompt.ask("  Val CSV path",   default=emb["val_csv"])
    emb["test_csv"]   = Prompt.ask("  Test CSV path",  default=emb["test_csv"])
    emb["smiles_col"] = Prompt.ask("  SMILES column",  default=emb["smiles_col"])
    emb["target_col"] = Prompt.ask("  Target column",  default=emb["target_col"])

    try:
        head = pd.read_csv(emb["train_csv"], nrows=1)
        ok = True
        for col in (emb["smiles_col"], emb["target_col"]):
            if col not in head.columns:
                warn(f"column {col!r} not found in {Path(emb['train_csv']).name}")
                ok = False
        if ok:
            success("CSV columns look good")
    except Exception as e:
        warn(f"could not read train CSV: {e}")

    save_state(state)
    pause()


# ── menu: train embedder ──────────────────────────────────────────────────

def run_train_embedder(state: dict) -> None:
    section("Train embedder")
    emb = state["embedder"]
    trn = state["train"]
    hpo = state["hpo"]

    for path in (emb["train_csv"], emb["val_csv"]):
        if not path or not Path(path).exists():
            error(f"path does not exist: {path}")
            return

    tuned = hpo["best_params"] if hpo["use_tuned"] and hpo["best_params"] else None
    if tuned:
        info("Using HPO-tuned hyperparameters")

    info(f"epochs={trn['epochs']}  batch_size={trn['batch_size']}  seed={trn['seed']}")

    try:
        import embedder as embedder_mod
    except Exception as e:
        error(f"failed to import embedder: {e}")
        return

    model_dir = Path(emb["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = embedder_mod.train_embedder(
            train_csv=emb["train_csv"],
            val_csv=emb["val_csv"],
            smiles_col=emb["smiles_col"],
            target_col=emb["target_col"],
            save_dir=str(model_dir),
            epochs=trn["epochs"],
            batch_size=trn["batch_size"],
            seed=trn["seed"],
            tuned_hparams=tuned,
        )
    except Exception as e:
        error(f"training failed: {e}")
        return

    weights_path = model_dir / "embedder.pt"
    embedder_mod.save_model(model, weights_path)
    success(f"Encoder saved → {_short(str(weights_path))}")
    if tuned:
        with open(model_dir / "embedder_hparams.json", "w") as f:
            json.dump(tuned, f, indent=2)

    emb["trained"] = True
    save_state(state)


def _resolve_embedder_hparams(state: dict) -> dict | None:
    """Find the hparams the trained embedder was built with, if any."""
    cfg_path = Path(state["embedder"]["model_dir"]) / "embedder_hparams.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ── menu: configure noisy / truth ─────────────────────────────────────────

def _menu_configure_csv(state: dict, key: str, label: str) -> None:
    section(f"Configure {label} dataset")
    cfg = state[key]
    info(f"Current CSV:    {_short(cfg['csv'])}")
    info(f"Current smiles: {cfg['smiles_col']}")
    info(f"Current y:      {cfg['target_col']}")
    console.print()

    csv = Prompt.ask("  CSV path", default=cfg["csv"] or "")
    if csv:
        cfg["csv"] = csv
    cfg["smiles_col"] = Prompt.ask("  SMILES column", default=cfg["smiles_col"])
    cfg["target_col"] = Prompt.ask("  Y column",      default=cfg["target_col"])

    if cfg["csv"]:
        try:
            head = pd.read_csv(cfg["csv"], nrows=1)
            ok = True
            for col in (cfg["smiles_col"], cfg["target_col"]):
                if col not in head.columns:
                    warn(f"column {col!r} not found in {Path(cfg['csv']).name}")
                    ok = False
            if ok:
                success("CSV columns look good")
        except Exception as e:
            warn(f"could not read CSV: {e}")

    save_state(state)
    pause()


# ── menu: run k-NN denoising ──────────────────────────────────────────────

def run_denoise(state: dict) -> None:
    section("Run k-NN denoising")
    emb = state["embedder"]
    nz = state["noisy"]
    knn = state["knn"]

    if not emb["trained"]:
        error("Embedder is not trained. Run option (2) first.")
        return
    if not nz["csv"] or not Path(nz["csv"]).exists():
        error("Noisy CSV not configured. Run option (3) first.")
        return

    weights_path = Path(emb["model_dir"]) / "embedder.pt"
    if not weights_path.exists():
        error(f"embedder weights not found at {weights_path}")
        return

    try:
        import embedder as embedder_mod
        import denoise as denoise_mod
    except Exception as e:
        error(f"failed to import modules: {e}")
        return

    df = pd.read_csv(nz["csv"])
    if nz["smiles_col"] not in df.columns:
        error(f"column {nz['smiles_col']!r} not in {nz['csv']}")
        return
    if nz["target_col"] not in df.columns:
        error(f"column {nz['target_col']!r} not in {nz['csv']}")
        return

    info(f"Loaded {len(df)} rows from {Path(nz['csv']).name}")

    smiles = df[nz["smiles_col"]].astype(str).tolist()
    y_initial = df[nz["target_col"]].astype(float).values

    info("Loading embedder …")
    model = embedder_mod.load_model(
        weights_path, hparams=_resolve_embedder_hparams(state),
    )

    info("Embedding molecules …")
    z, valid = embedder_mod.extract_embeddings(
        model, smiles, batch_size=state["train"]["batch_size"],
    )
    n_dropped = len(smiles) - len(valid)
    if n_dropped:
        warn(f"{n_dropped} SMILES could not be parsed and were dropped")
    if len(valid) == 0:
        error("no valid molecules to denoise")
        return

    smiles = [smiles[i] for i in valid]
    y_in = y_initial[valid]

    info(
        f"Running denoising  k={knn['k']}  rounds={knn['rounds']}  "
        f"on {len(z)} molecules (d={z.shape[1]})"
    )

    last_run_dir = WORKSPACE / "last_run"
    if last_run_dir.exists():
        shutil.rmtree(last_run_dir)
    last_run_dir.mkdir(parents=True, exist_ok=True)

    rounds_dir = last_run_dir / "rounds"
    y_final, _history = denoise_mod.iterative_denoise(
        z=z, y=y_in, k=knn["k"], rounds=knn["rounds"],
        log_dir=rounds_dir, smiles=smiles,
    )

    out_path = last_run_dir / "denoised.csv"
    pd.DataFrame({
        "smiles":     smiles,
        "y_initial":  y_in,
        "y_denoised": y_final,
    }).to_csv(out_path, index=False)
    success(f"Wrote {_short(str(out_path))}")

    state["last_run"]["denoised_csv"] = str(out_path)
    state["last_run"]["metrics"] = None
    save_state(state)


# ── menu: benchmark ───────────────────────────────────────────────────────

def run_benchmark(state: dict) -> None:
    section("Benchmark vs. ground truth")
    last = state["last_run"]
    tr = state["truth"]

    if not last["denoised_csv"] or not Path(last["denoised_csv"]).exists():
        error("No denoised output yet. Run option (5) first.")
        return
    if not tr["csv"] or not Path(tr["csv"]).exists():
        error("Truth CSV not configured. Run option (4) first.")
        return

    try:
        import denoise as denoise_mod
    except Exception as e:
        error(f"failed to import denoise: {e}")
        return

    df_pred = pd.read_csv(last["denoised_csv"])
    df_true = pd.read_csv(tr["csv"])

    if tr["smiles_col"] not in df_true.columns or tr["target_col"] not in df_true.columns:
        error(
            f"truth CSV is missing one of the expected columns "
            f"({tr['smiles_col']!r}, {tr['target_col']!r})"
        )
        return

    truth = (
        df_true[[tr["smiles_col"], tr["target_col"]]]
        .rename(columns={tr["smiles_col"]: "smiles", tr["target_col"]: "y_truth"})
    )
    merged = df_pred.merge(truth, on="smiles", how="inner")
    n = len(merged)
    if n == 0:
        error("no SMILES overlap between denoised output and truth CSV")
        return
    if n < len(df_pred):
        warn(f"only {n}/{len(df_pred)} denoised rows had a truth match")

    metrics_initial = denoise_mod.benchmark_against_truth(
        merged["y_initial"].values, merged["y_truth"].values,
    )
    metrics_final = denoise_mod.benchmark_against_truth(
        merged["y_denoised"].values, merged["y_truth"].values,
    )

    tbl = Table(
        box=box.ROUNDED, border_style="green", expand=False,
        title="[bold green]Denoising benchmark[/bold green]",
    )
    tbl.add_column("Metric",     style="accent")
    tbl.add_column("Noisy input", style="warn")
    tbl.add_column("Denoised",   style="success")
    tbl.add_column("Δ",          style="muted")
    for key in ("mse", "mae", "rmse", "r2"):
        before = metrics_initial[key]
        after = metrics_final[key]
        delta = after - before
        tbl.add_row(key.upper(), f"{before:.4f}", f"{after:.4f}", f"{delta:+.4f}")
    console.print(tbl)

    metrics_path = WORKSPACE / "last_run" / "metrics.json"
    denoise_mod.write_metrics(
        {"initial": metrics_initial, "denoised": metrics_final, "n": n},
        metrics_path,
    )
    success(f"Wrote {_short(str(metrics_path))}")
    state["last_run"]["metrics"] = metrics_final
    save_state(state)


# ── menu: HPO ─────────────────────────────────────────────────────────────

def menu_hpo(state: dict) -> None:
    while True:
        clear()
        banner()
        section("Hyperparameter tuning (advanced)")
        hpo = state["hpo"]
        emb = state["embedder"]
        info(f"use tuned params: {'YES' if hpo['use_tuned'] else 'no'}")
        info(f"n_trials:         {hpo['n_trials']}")
        info(f"epochs/trial:     {hpo['hpo_epochs']}")
        info(f"best params:      {'set' if hpo['best_params'] else '<none>'}")
        console.print()

        opts = [
            "Run HPO trials",
            f"Toggle 'use tuned params' [{'ON' if hpo['use_tuned'] else 'OFF'}]",
            "Edit n_trials / epochs",
            "View best params",
            "Clear best params",
            "Back",
        ]
        idx = curses_select(opts, title="HPO menu")
        if idx < 0 or idx == 5:
            return

        if idx == 0:
            paths_ok = True
            for path in (emb["train_csv"], emb["val_csv"]):
                if not path or not Path(path).exists():
                    error(f"path does not exist: {path}")
                    paths_ok = False
                    break
            if not paths_ok:
                pause()
                continue
            try:
                import hpo as hpo_mod
            except Exception as e:
                error(f"failed to import hpo: {e}")
                pause()
                continue
            try:
                best = hpo_mod.run_hpo(
                    train_csv=emb["train_csv"],
                    val_csv=emb["val_csv"],
                    smiles_col=emb["smiles_col"],
                    target_col=emb["target_col"],
                    seed=state["train"]["seed"],
                    n_trials=hpo["n_trials"],
                    hpo_epochs=hpo["hpo_epochs"],
                    save_dir=str(WORKSPACE / "hpo"),
                )
                hpo["best_params"] = best
                hpo["use_tuned"] = True
                save_state(state)
                success("HPO finished and best params stored")
            except Exception as e:
                error(f"HPO failed: {e}")
            pause()

        elif idx == 1:
            hpo["use_tuned"] = not hpo["use_tuned"]
            save_state(state)
        elif idx == 2:
            hpo["n_trials"]   = IntPrompt.ask("  n_trials",     default=hpo["n_trials"])
            hpo["hpo_epochs"] = IntPrompt.ask("  epochs/trial", default=hpo["hpo_epochs"])
            save_state(state)
        elif idx == 3:
            section("Best params")
            if hpo["best_params"]:
                console.print_json(json.dumps(hpo["best_params"]))
            else:
                info("no best params stored")
            pause()
        elif idx == 4:
            if Confirm.ask("  Clear stored best params?", default=False):
                hpo["best_params"] = None
                hpo["use_tuned"] = False
                save_state(state)


# ── menu: settings ────────────────────────────────────────────────────────

def menu_settings(state: dict) -> None:
    section("Settings")
    trn = state["train"]
    knn = state["knn"]

    trn["epochs"]     = IntPrompt.ask("  Training epochs",  default=trn["epochs"])
    trn["batch_size"] = IntPrompt.ask("  Batch size",        default=trn["batch_size"])
    trn["seed"]       = IntPrompt.ask("  Seed",              default=trn["seed"])
    knn["k"]          = IntPrompt.ask("  k (neighbours)",    default=knn["k"])
    knn["rounds"]     = IntPrompt.ask("  Denoising rounds",  default=knn["rounds"])

    save_state(state)
    success("Settings updated")
    pause()


# ── reset ─────────────────────────────────────────────────────────────────

def reset_workspace(state: dict) -> dict:
    section("Reset workspace")
    if Confirm.ask("  [err]Delete everything in cli_workspace/?[/err]", default=False):
        try:
            shutil.rmtree(WORKSPACE)
            success("workspace deleted")
        except FileNotFoundError:
            success("workspace already empty")
        except Exception as e:
            error(f"delete failed: {e}")
        return copy.deepcopy(DEFAULT_STATE)
    return state


# ── main loop ─────────────────────────────────────────────────────────────

MAIN_OPTS = [
    "Overview",                                    # 0
    "Configure embedder training data",            # 1
    "Train embedder",                              # 2
    "Configure noisy dataset",                     # 3
    "Configure ground-truth dataset",              # 4
    "Run k-NN denoising",                          # 5
    "Benchmark last run vs. truth",                # 6
    "Hyperparameter tuning (advanced)",            # 7
    "Settings (epochs, batch, seed, k, rounds)",   # 8
    "Reset workspace",                             # 9
    "Quit",                                        # 10
]


def main():
    clear()
    state = load_state()

    while True:
        clear()
        banner()
        idx = curses_select(
            MAIN_OPTS,
            title="Main menu",
            subtitle=f"Workspace: {WORKSPACE}",
        )

        if idx < 0 or idx == 10:
            console.print("\n  [muted]Goodbye![/muted]\n")
            break

        if idx == 0:
            print_status(state)
            pause()
        elif idx == 1:
            menu_configure_embedder(state)
        elif idx == 2:
            run_train_embedder(state)
            pause()
        elif idx == 3:
            _menu_configure_csv(state, "noisy", "noisy")
        elif idx == 4:
            _menu_configure_csv(state, "truth", "ground-truth")
        elif idx == 5:
            run_denoise(state)
            pause()
        elif idx == 6:
            run_benchmark(state)
            pause()
        elif idx == 7:
            menu_hpo(state)
        elif idx == 8:
            menu_settings(state)
        elif idx == 9:
            state = reset_workspace(state)
            save_state(state)
            pause()


if __name__ == "__main__":
    main()
