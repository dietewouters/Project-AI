#!/usr/bin/env python3
"""
User Interface CLI v3 -- Pipeline-driven dataset synthesis & GNN training toolkit.

All navigation uses arrow keys + ENTER. No number input, no emoji.

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
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from add_noise import add_noise

# -- ensure sibling modules are importable ------------------------------------
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# -- dependency bootstrap -----------------------------------------------------
def _pip_install(pkg: str):
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for _pkg in ("rich", "pandas", "scipy"):
    try:
        __import__(_pkg)
    except ImportError:
        print(f"Installing '{_pkg}' ...")
        _pip_install(_pkg)

from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt, FloatPrompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# =============================================================================
#  Theme & console
# =============================================================================
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

VERSION = "3.0.0"

# =============================================================================
#  Workspace paths
# =============================================================================
WORKSPACE  = Path(__file__).resolve().parent / "cli_workspace"
STATE_FILE = WORKSPACE / ".state.json"
SPLIT_NAMES = ("train", "test", "val")

# =============================================================================
#  Default state
# =============================================================================
DEFAULT_STATE: dict[str, Any] = {
    "pipelines": {
        "train": {"base_file": None, "slices": [], "augment": False},
        "test":  {"base_file": None, "slices": [], "augment": False, "sync": False},
        "val":   {"base_file": None, "slices": [], "augment": False, "sync": False},
    },
    "prepared": {"train": None, "test": None, "val": None},
    "node_noise": {
        "node_fraction": 0.0,
        "dim_fraction":  0.0,
        "layers":        [],
    },
    "scaffold_split": {
        "enabled":    False,
        "train_size": 0.8,
        "val_size":   0.1,
        "test_size":  0.1,
        "use_generic": False,
    },
    "hpo": {
        "enabled":    False,
        "n_trials":   20,
        "hpo_epochs": 10,
        "best_params": None,
    },
    "config": {
        "smiles_col":    "smiles",
        "target_col":    "h298",
        "epochs":        30,
        "batch_size":    128,
        "seed":          42,
        "max_tautomers": 5,
        "do_mirror":     True,
        "do_tautomers":  True,
        "attributes":    ["h298"],
        "shuffle":       True,
        "disable_output_scaling": False,
        "disable_input_scaling":  False,
    },
    "models": {},
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


def save_state(state: dict):
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# =============================================================================
#  Rich UI helpers
# =============================================================================

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    art = Text.from_markup(
        f"\n  [brand]MolAugment[/brand]  [muted]v{VERSION}[/muted]\n"
        "  [muted]Data synthesis, noise generation & GNN training pipeline[/muted]\n"
    )
    console.print(Panel(art, border_style="cyan", box=box.DOUBLE_EDGE,
                        expand=False, padding=(0, 2)))


def section(title: str):
    console.print()
    console.print(Rule(f"[accent] {title} [/accent]", style="magenta"))
    console.print()


def success(msg: str):  console.print(f"  [success]OK[/success]  {msg}")
def warn(msg: str):     console.print(f"  [warn]!![/warn]  {msg}")
def error(msg: str):    console.print(f"  [err]ERR[/err] {msg}")
def info(msg: str):     console.print(f"  [muted]--[/muted]  {msg}")


def pause():
    console.print()
    Prompt.ask("  [muted]Press Enter to continue[/muted]", default="")


def humanize_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.0f} {unit}"
        n /= 1024  # type: ignore
    return f"{n:,.1f} TB"


# =============================================================================
#  Curses helpers
# =============================================================================

def curses_select(items: list[str], title: str = "", subtitle: str = "") -> int:
    result = [-1]

    def _run(stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN,   -1)   # selected row
        curses.init_pair(2, curses.COLOR_YELLOW, -1)   # title
        curses.init_pair(3, curses.COLOR_GREEN,  -1)   # hint

        cursor = 0

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            if title:
                try:
                    stdscr.addstr(0, 0, f"  {title}  "[:w - 1].ljust(w - 1)[:w - 1],
                                  curses.color_pair(2) | curses.A_BOLD)
                except curses.error:
                    pass

            row = 1
            if subtitle:
                try:
                    stdscr.addstr(row, 0, f"  {subtitle}"[:w - 1], curses.A_DIM)
                    row += 1
                except curses.error:
                    pass

            try:
                stdscr.hline(row, 0, curses.ACS_HLINE, w - 1)
                row += 1
            except curses.error:
                pass

            max_rows = max(1, h - row - 2)
            scroll_off = max(0, cursor - max_rows + 1)
            visible = items[scroll_off: scroll_off + max_rows]

            for i, label in enumerate(visible):
                real_i  = i + scroll_off
                is_cur  = real_i == cursor
                prefix  = " -> " if is_cur else "    "
                line    = f"{prefix}{label}"
                line    = line[:w - 1].ljust(w - 1)[:w - 1]
                attr    = (curses.color_pair(1) | curses.A_BOLD) if is_cur else curses.A_NORMAL
                try:
                    stdscr.addstr(row + i, 0, line, attr)
                except curses.error:
                    pass

            hint = "  UP/DOWN  navigate    ENTER  select    ESC/Q  cancel"
            try:
                stdscr.addstr(h - 1, 0, hint[:w - 1].ljust(w - 1)[:w - 1], curses.color_pair(3))
            except curses.error:
                pass

            stdscr.refresh()
            key = stdscr.getch()

            if key in (curses.KEY_UP, ord('k')):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                cursor = min(len(items) - 1, cursor + 1)
            elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                result[0] = cursor
                break
            elif key in (27, ord('q'), ord('Q')):
                result[0] = -1
                break

    curses.wrapper(_run)
    return result[0]


_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", "cli_workspace"}

def _list_dir_entries(path: Path) -> list[tuple[str, bool]]:
    entries = []
    try:
        for p in sorted(path.iterdir()):
            if p.name.startswith(".") or p.name in _SKIP_DIRS:
                continue
            entries.append((p.name, p.is_dir()))
    except PermissionError:
        pass
    entries.sort(key=lambda x: (not x[1], x[0].lower()))
    return entries


def browse_and_select_single(start_path: Path) -> Path | None:
    """Interactive curses file browser. Allows picking a SINGLE file."""
    selected_path: list[str] = [None]

    def _run(stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_CYAN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)

        nav_stack: list[Path] = []
        current = start_path.resolve()
        cursor = 0

        while True:
            entries = _list_dir_entries(current)
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            path_disp = str(current)
            if len(path_disp) > w - 7:
                path_disp = "..." + path_disp[-(w - 10):]
            header = f" DIR: {path_disp}"
            try:
                stdscr.addstr(0, 0, header[:w - 1].ljust(w - 1)[:w - 1], curses.A_REVERSE)
            except curses.error: pass

            hint = " UP/DOWN select   ENTER open/confirm   BKSP go up   Q cancel"
            try:
                stdscr.addstr(1, 0, hint[:w - 1], curses.color_pair(3))
                stdscr.hline(2, 0, curses.ACS_HLINE, w - 1)
            except curses.error: pass

            max_rows = max(1, h - 5)
            scroll_off = max(0, cursor - max_rows + 3)
            visible = entries[scroll_off: scroll_off + max_rows]

            for i, (name, is_dir) in enumerate(visible):
                real_i = i + scroll_off
                is_cur = real_i == cursor
                prefix = ">" if is_dir else " "
                line = f" {prefix} {name}"
                line = line[:w - 1].ljust(w - 1)[:w - 1]

                if is_cur:
                    attr = curses.A_REVERSE
                elif is_dir:
                    attr = curses.color_pair(2)
                else:
                    attr = curses.A_NORMAL

                try:
                    stdscr.addstr(3 + i, 0, line, attr)
                except curses.error: pass

            stdscr.refresh()
            key = stdscr.getch()

            if key in (curses.KEY_UP, ord('k')):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                cursor = min(max(0, len(entries) - 1), cursor + 1)
            elif key in (ord('q'), ord('Q'), 27):
                break
            elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                if entries and 0 <= cursor < len(entries):
                    name, is_dir = entries[cursor]
                    if is_dir:
                        nav_stack.append(current)
                        current = current / name
                        cursor = 0
                    else:
                        selected_path[0] = str(current / name)
                        break
            elif key in (curses.KEY_BACKSPACE, 127, curses.KEY_LEFT):
                if nav_stack:
                    current = nav_stack.pop()
                    cursor = 0

    curses.wrapper(_run)
    return Path(selected_path[0]) if selected_path[0] else None


# =============================================================================
#  Status / Overview
# =============================================================================

def print_status(state: dict):
    section("Workspace Overview")
    cfg = state["config"]

    # Config Summary
    cfg_tbl = Table(box=box.SIMPLE, show_header=False, expand=False, padding=(0, 1))
    cfg_tbl.add_column("Key", style="accent", width=24)
    cfg_tbl.add_column("Value", style="white")
    cfg_tbl.add_row("SMILES column", cfg["smiles_col"])
    cfg_tbl.add_row("Target column", cfg["target_col"])
    cfg_tbl.add_row("Attributes", ", ".join(cfg["attributes"]))
    cfg_tbl.add_row("Shuffle Data", "[success]ENABLED[/success]" if cfg.get("shuffle", True) else "[err]DISABLED[/err]")
    cfg_tbl.add_row("Output Scaling",
                    "[err]DISABLED[/err]" if cfg.get("disable_output_scaling", False)
                    else "[success]ENABLED[/success]")
    cfg_tbl.add_row("Input Scaling",
                    "[err]DISABLED[/err]" if cfg.get("disable_input_scaling", False)
                    else "[success]ENABLED[/success]")
    console.print(Panel(cfg_tbl, title="[hi]Config[/hi]", border_style="cyan", expand=False))

    # Scaffold Split Summary
    sc_cfg = state.get("scaffold_split", {})
    if sc_cfg.get("enabled"):
        sc_tbl = Table(box=box.SIMPLE, show_header=False, expand=False, padding=(0, 1))
        sc_tbl.add_column("Key", style="accent", width=24)
        sc_tbl.add_column("Value", style="white")
        sc_tbl.add_row("Status", "[success]ENABLED[/success]")
        sc_tbl.add_row("Train / Val / Test",
                       f"{sc_cfg.get('train_size',0.8)*100:.0f}% / "
                       f"{sc_cfg.get('val_size',0.1)*100:.0f}% / "
                       f"{sc_cfg.get('test_size',0.1)*100:.0f}%")
        sc_tbl.add_row("Scaffold type",
                       "Generic (rings only)" if sc_cfg.get("use_generic") else "Murcko (with heteroatoms)")
        console.print(Panel(sc_tbl, title="[hi]Scaffold Split[/hi]", border_style="green", expand=False))
    else:
        console.print(Panel("  [muted]Scaffold split: OFF — Using manually assigned CSVs[/muted]",
                            title="[hi]Scaffold Split[/hi]", border_style="green", expand=False))

    # Node Noise Summary
    nn_cfg = state.get("node_noise", {})
    nn_layers = nn_cfg.get("layers", [])
    nn_nf = nn_cfg.get("node_fraction", 0.0)
    nn_df = nn_cfg.get("dim_fraction", 0.0)
    if nn_layers and nn_nf > 0 and nn_df > 0:
        nn_tbl = Table(box=box.SIMPLE, show_header=False, expand=False, padding=(0, 1))
        nn_tbl.add_column("Key", style="accent", width=24)
        nn_tbl.add_column("Value", style="white")
        nn_tbl.add_row("Node fraction", f"{nn_nf*100:.1f}%")
        nn_tbl.add_row("Dimension fraction", f"{nn_df*100:.1f}%")
        for i, layer in enumerate(nn_layers):
            nn_tbl.add_row(f"Noise layer {i+1}", f"{layer.get('type','normal')} (scale={layer.get('scale',0):.4f})")
        nn_tbl.add_row("Training mode", "[brand]Python API (node noise active)[/brand]")
        console.print(Panel(nn_tbl, title="[hi]Node Feature Noise[/hi]", border_style="yellow", expand=False))
    else:
        console.print(Panel("  [muted]Node noise: OFF — Training uses Chemprop CLI[/muted]",
                            title="[hi]Node Feature Noise[/hi]", border_style="yellow", expand=False))

    # HPO Summary
    hpo_cfg = state.get("hpo", {})
    if hpo_cfg.get("enabled"):
        hpo_tbl = Table(box=box.SIMPLE, show_header=False, expand=False, padding=(0, 1))
        hpo_tbl.add_column("Key", style="accent", width=24)
        hpo_tbl.add_column("Value", style="white")
        hpo_tbl.add_row("Status", "[success]ENABLED[/success]")
        hpo_tbl.add_row("Trials", str(hpo_cfg.get("n_trials", 20)))
        hpo_tbl.add_row("Epochs/trial", str(hpo_cfg.get("hpo_epochs", 10)))
        bp = hpo_cfg.get("best_params")
        if bp:
            bp_str = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in bp.items())
            hpo_tbl.add_row("Best params", f"[success]Cached[/success]")
            hpo_tbl.add_row("", f"[muted]{bp_str[:80]}[/muted]")
        else:
            hpo_tbl.add_row("Best params", "[warn]Not yet tuned[/warn]")
        console.print(Panel(hpo_tbl, title="[hi]Hyperparameter Optimisation[/hi]", border_style="magenta", expand=False))
    else:
        console.print(Panel("  [muted]HPO: OFF — Training uses default hyperparameters[/muted]",
                            title="[hi]Hyperparameter Optimisation[/hi]", border_style="magenta", expand=False))

    # Pipeline status
    for split in SPLIT_NAMES:
        p_cfg = state["pipelines"][split]
        prepared = state["prepared"][split]

        tbl = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
        tbl.add_column("Param", style="white", width=14)
        tbl.add_column("Value", style="muted")

        tbl.add_row("Base CSV", p_cfg["base_file"] or "None")
        if p_cfg.get("sync"):
            tbl.add_row("Sync", "[brand]ON (Follows Train)[/brand]")
        
        if p_cfg["base_file"]:
            source_slices = state["pipelines"]["train"]["slices"] if p_cfg.get("sync") else p_cfg["slices"]
            source_augment = state["pipelines"]["train"]["augment"] if p_cfg.get("sync") else p_cfg["augment"]
            
            slices_str = ", ".join(f"{s['fraction']*100}% at {s['noise']} noise" for s in source_slices)
            tbl.add_row("Slices", slices_str if slices_str else "100% at 0 noise")
            tbl.add_row("Augment", "ON" if source_augment else "OFF")

            if prepared and (WORKSPACE / prepared).exists():
                sz = (WORKSPACE / prepared).stat().st_size
                tbl.add_row("Prepared", f"[success]{prepared}[/success] ({humanize_bytes(sz)})")
            else:
                tbl.add_row("Prepared", "[warn]Not prepared yet[/warn]")

        border = {"train": "magenta", "test": "blue", "val": "green"}[split]
        console.print(Panel(tbl, title=f"[hi]{split.capitalize()} Pipeline[/hi]", border_style=border))


# =============================================================================
#  Pipeline Configuration Sub-menus
# =============================================================================

def menu_pipeline_split(state: dict, split: str):
    p_cfg = state["pipelines"][split]
    
    while True:
        clear()
        banner()
        section(f"Configure {split.capitalize()} Pipeline")
        
        base = p_cfg["base_file"] or "None"
        s_count = len(p_cfg["slices"])
        aug_st = "ON" if p_cfg["augment"] else "OFF"
        
        opts = [
            f"Assign base CSV file  [{base}]",
            f"Manage noise slices    [{s_count} slices configured]",
            f"Toggle augmentation    [{aug_st}]",
        ]
        if "sync" in p_cfg:
            sync_st = "ON" if p_cfg["sync"] else "OFF"
            opts.append(f"Sync with Train noise  [{sync_st}]")
            
        opts.append("-- Back --")
        
        idx = curses_select(opts, title=f"{split.capitalize()} Settings")
        if idx < 0 or idx == len(opts) - 1:
            break
            
        elif idx == 0:
            proj_root = Path(__file__).resolve().parent.parent
            console.print(f"\n  [muted]Opening browser in {proj_root}[/muted]")
            pause()
            selected = browse_and_select_single(proj_root)
            if selected:
                dest = WORKSPACE / selected.name
                shutil.copy2(selected, dest)
                p_cfg["base_file"] = dest.name
                save_state(state)
                success(f"Assigned base CSV: [brand]{dest.name}[/brand]")
                pause()
                
        elif idx == 1:
            if p_cfg.get("sync"):
                warn("Manage slices is disabled when 'Sync with Train' is ON.")
                pause()
            else:
                menu_manage_slices(state, split)
            
        elif idx == 2:
            if p_cfg.get("sync"):
                warn("Toggle augmentation is disabled when 'Sync with Train' is ON.")
                pause()
            else:
                p_cfg["augment"] = not p_cfg["augment"]
                save_state(state)
        
        elif idx == 3 and "sync" in p_cfg:
            p_cfg["sync"] = not p_cfg["sync"]
            save_state(state)


def menu_manage_slices(state: dict, split: str):
    p_cfg = state["pipelines"][split]
    
    while True:
        clear()
        banner()
        section(f"Manage Noise Slices ({split})")
        
        slices = p_cfg["slices"]
        opts = []
        tot_frac = 0.0
        for i, s in enumerate(slices):
            n_type = s.get("type", "normal")
            opts.append(f"Remove Slice: {s['fraction']*100:.1f}% data | {n_type} noise, scale: {s['noise']}")
            tot_frac += s["fraction"]
            
        opts.append(f"Add new slice (Remaining: {(1.0 - tot_frac)*100:.1f}%)")
        opts.append("-- Done --")
        
        idx = curses_select(opts, title="Noise Configuration")
        if idx < 0 or idx == len(opts) - 1:
            break
            
        if idx < len(slices):
            # Remove
            slices.pop(idx)
            save_state(state)
        else:
            # Add
            rem = 1.0 - tot_frac
            if rem <= 0:
                warn("100% of data is already sliced! Remove a slice first.")
                pause()
                continue
                
            fraction = FloatPrompt.ask(f"  [accent]Fraction of data to use[/accent] (e.g. 0.1 for 10%, max {rem:.2f})")
            if fraction <= 0 or fraction > rem:
                warn(f"Invalid fraction. Must be between 0.01 and {rem:.2f}")
                pause()
                continue
                
            noise = FloatPrompt.ask("  [accent]Noise scale (std dev for normal distribution or base level)[/accent]")
            
            noise_type = "normal"
            if noise > 0:
                types = ["normal", "cosh", "uniform", "bimodal", "half_and_half", "nitrogen"]
                t_idx = curses_select(types, title="Select Noise Distribution Type")
                if t_idx >= 0:
                    noise_type = types[t_idx]
            
            slices.append({"fraction": fraction, "noise": noise, "type": noise_type})
            save_state(state)


# =============================================================================
#  Scaffold Splitting
# =============================================================================

def scaffold_split_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    use_generic: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Split a dataframe by Murcko scaffold so that molecules with the same
    core ring structure are kept in the same split.

    Returns (train_df, val_df, test_df, stats_dict).
    """
    from collections import defaultdict
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    # 1. Compute scaffold for each molecule
    scaffolds: dict[str, list[int]] = defaultdict(list)
    no_scaffold_indices = []

    for i, smi in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            no_scaffold_indices.append(i)
            continue
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            if use_generic:
                core = MurckoScaffold.MakeScaffoldGeneric(core)
            scaffold_smi = Chem.MolToSmiles(core)
            if not scaffold_smi or scaffold_smi == "":
                scaffold_smi = "__no_rings__"
        except Exception:
            scaffold_smi = "__no_rings__"
        scaffolds[scaffold_smi].append(i)

    # 2. Sort scaffolds by size (largest first) for balanced assignment
    scaffold_groups = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)

    # 3. Greedy assignment: each scaffold goes entirely to the split
    #    that is furthest below its target count.
    n = len(df)
    target_counts = {
        "train": int(n * train_size),
        "val":   int(n * val_size),
        "test":  int(n * test_size),
    }
    current_counts = {"train": 0, "val": 0, "test": 0}
    assignments: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    rng = np.random.RandomState(seed)
    # Shuffle scaffolds of same size for randomness
    scaffold_groups_shuffled = []
    i = 0
    while i < len(scaffold_groups):
        j = i
        while j < len(scaffold_groups) and len(scaffold_groups[j][1]) == len(scaffold_groups[i][1]):
            j += 1
        group = list(scaffold_groups[i:j])
        rng.shuffle(group)
        scaffold_groups_shuffled.extend(group)
        i = j

    for scaffold_smi, indices in scaffold_groups_shuffled:
        # Find the split with the largest remaining deficit
        deficits = {s: target_counts[s] - current_counts[s] for s in ("train", "val", "test")}
        best_split = max(deficits, key=deficits.get)
        assignments[best_split].extend(indices)
        current_counts[best_split] += len(indices)

    # Add no-scaffold molecules to training
    if no_scaffold_indices:
        assignments["train"].extend(no_scaffold_indices)

    train_df = df.iloc[sorted(assignments["train"])].copy().reset_index(drop=True)
    val_df   = df.iloc[sorted(assignments["val"])].copy().reset_index(drop=True)
    test_df  = df.iloc[sorted(assignments["test"])].copy().reset_index(drop=True)

    stats = {
        "total": n,
        "n_scaffolds": len(scaffolds),
        "train_count": len(train_df),
        "val_count":   len(val_df),
        "test_count":  len(test_df),
        "no_scaffold":  len(no_scaffold_indices),
    }
    return train_df, val_df, test_df, stats


# =============================================================================
#  Execution Pipeline (Prep)
# =============================================================================

def _ensure_rdkit() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except ImportError:
        error("RDKit is not installed.")
        return False


def run_preparation_pipeline(state: dict):
    section("Data Preparation Pipeline Execution")

    sc_cfg = state.get("scaffold_split", {})
    use_scaffold = sc_cfg.get("enabled", False)

    # ── SCAFFOLD SPLIT MODE ────────────────────────────────────────────
    if use_scaffold:
        train_base = state["pipelines"]["train"].get("base_file")
        if not train_base:
            warn("Scaffold split is enabled but no base CSV is assigned to the Train pipeline.")
            warn("Assign a CSV to Train first — scaffold split uses it as the single source.")
            return

        base_path = WORKSPACE / train_base
        if not base_path.exists():
            error(f"Base file {train_base} does not exist.")
            return

        if not _ensure_rdkit():
            return

        c_cfg = state["config"]
        df_full = pd.read_csv(base_path)
        if c_cfg["smiles_col"] not in df_full.columns or c_cfg["target_col"] not in df_full.columns:
            error(f"Columns {c_cfg['smiles_col']} or {c_cfg['target_col']} missing in {base_path.name}")
            return

        info(f"Scaffold-splitting {len(df_full)} molecules from {train_base}...")

        train_df, val_df, test_df, stats = scaffold_split_dataframe(
            df_full,
            smiles_col=c_cfg["smiles_col"],
            train_size=sc_cfg.get("train_size", 0.8),
            val_size=sc_cfg.get("val_size", 0.1),
            test_size=sc_cfg.get("test_size", 0.1),
            seed=c_cfg["seed"],
            use_generic=sc_cfg.get("use_generic", False),
        )

        # Show scaffold stats
        st_tbl = Table(box=box.ROUNDED, border_style="green", expand=False, show_header=False)
        st_tbl.add_column("Stat", style="accent", width=22)
        st_tbl.add_column("Value", style="white")
        st_tbl.add_row("Total molecules",    str(stats["total"]))
        st_tbl.add_row("Unique scaffolds",   str(stats["n_scaffolds"]))
        st_tbl.add_row("Train",              f"{stats['train_count']} ({stats['train_count']/stats['total']*100:.1f}%)")
        st_tbl.add_row("Validation",         f"{stats['val_count']} ({stats['val_count']/stats['total']*100:.1f}%)")
        st_tbl.add_row("Test",               f"{stats['test_count']} ({stats['test_count']/stats['total']*100:.1f}%)")
        if stats["no_scaffold"]:
            st_tbl.add_row("No scaffold (→train)", str(stats["no_scaffold"]))
        console.print(Padding(st_tbl, (0, 2)))

        # Write scaffold-split CSVs into workspace so the per-split pipeline can pick them up
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            out_name = f"_scaffold_{split_name}.csv"
            split_df.to_csv(WORKSPACE / out_name, index=False)
            state["pipelines"][split_name]["base_file"] = out_name

        save_state(state)
        success("Scaffold split written. Now running per-split noise/augmentation pipeline...")
        console.print()

    # ── STANDARD PER-SPLIT PIPELINE ────────────────────────────────────
    # Force the order to Val -> Test -> Train to prioritize clean validation sets
    ordered_tasks = []
    for sp in ["val", "test", "train"]:
        if state["pipelines"][sp]["base_file"]:
            ordered_tasks.append(sp)

    if not ordered_tasks:
        warn("No splits have a base file assigned. Configure the pipeline first.")
        return
        
    global_used_smiles = set()
    
    for split in ordered_tasks:
        p_cfg = state["pipelines"][split]
        c_cfg = state["config"]
        
        # Determine effective slices/augment logic
        if p_cfg.get("sync"):
            source_cfg = state["pipelines"]["train"]
            eff_slices = source_cfg["slices"]
            eff_augment = source_cfg["augment"]
        else:
            eff_slices = p_cfg["slices"]
            eff_augment = p_cfg["augment"]
            
        base_path = WORKSPACE / p_cfg["base_file"]
        
        if not base_path.exists():
            error(f"Base file {p_cfg['base_file']} for {split} does not exist.")
            continue
            
        info(f"Processing split: {split}")
        
        # 1. Load data
        df = pd.read_csv(base_path)
        if c_cfg["smiles_col"] not in df.columns or c_cfg["target_col"] not in df.columns:
            error(f"Columns {c_cfg['smiles_col']} or {c_cfg['target_col']} missing in {base_path.name}")
            continue
        
        original_len = len(df)
        
        # 2. Filter out globally used SMILES to prevent data leakage across splits
        smiles_col = c_cfg["smiles_col"]
        if global_used_smiles:
            df = df[~df[smiles_col].isin(global_used_smiles)].copy()
            
        remaining_len = len(df)
        
        if remaining_len < original_len:
            info(f"  Filtered out {original_len - remaining_len} duplicated overlapping SMILES. Remaining: {remaining_len}")
            
        # 3. Handle slicing safely by grouping by SMILES
        # Grouping by SMILES ensures that data splits are taken cleanly without
        # splitting a single SMILES across boundaries, which would result in data loss
        # during the 'global_used_smiles' anti-leakage filter step.
        unique_smiles = df[smiles_col].unique()
        if c_cfg.get("shuffle", True):
            np.random.seed(c_cfg["seed"] + ord(split[0])) # Different shuffle per split
            np.random.shuffle(unique_smiles)

        slices = eff_slices
        if not slices:
            slices = [{"fraction": 1.0, "noise": 0.0}]
            
        slice_dfs = []
        current_idx = 0
        total_unique = len(unique_smiles)
        
        for idx, s in enumerate(slices):
            frac, noise = s["fraction"], s["noise"]
            n_unique = int(total_unique * frac)  # Calculate proportion based on original SMILES count
            
            if current_idx + n_unique > len(unique_smiles):
                warn_msg = f"Not enough unused SMILES remain in {base_path.name} to fulfill slice {idx+1} of {split}. Using remaining {len(unique_smiles) - current_idx} SMILES groups."
                warn(warn_msg)
                n_unique = len(unique_smiles) - current_idx
                
            if n_unique <= 0:
                continue
                
            selected_smiles = set(unique_smiles[current_idx:current_idx + n_unique])
            current_idx += n_unique
            
            sub_df = df[df[smiles_col].isin(selected_smiles)].copy()
            if len(sub_df) == 0:
                continue
            
            # Register claimed SMILES globally
            global_used_smiles.update(selected_smiles)
            
            # --- AUGMENTATION ---
            if eff_augment and _ensure_rdkit():
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')
                from augment_dataset import mirror_molecule, enumerate_tautomers
                
                aug_rows = []
                mirrors_added = 0
                tautomers_added = 0
                
                with Progress(
                    SpinnerColumn("dots", style="cyan"),
                    TextColumn(f"[brand]Augmenting slice {idx+1}...[/brand]"),
                    BarColumn(complete_style="cyan", finished_style="green"),
                    TimeElapsedColumn(), console=console
                ) as prog:
                    task_id = prog.add_task("Augmenting", total=len(sub_df))
                    
                    for _, row in sub_df.iterrows():
                        smi = row[smiles_col]
                        base_dict = row.to_dict()
                        
                        orig_dict = base_dict.copy()
                        orig_dict["augmentation"] = "original"
                        aug_rows.append(orig_dict)
                        
                        smiles_to_tautomerize = [smi]
                        
                        if c_cfg["do_mirror"]:
                            try:
                                m_smi = mirror_molecule(smi)
                                if m_smi:
                                    m_dict = base_dict.copy()
                                    m_dict[smiles_col] = m_smi
                                    m_dict["augmentation"] = "mirror"
                                    aug_rows.append(m_dict)
                                    mirrors_added += 1
                                    smiles_to_tautomerize.append(m_smi)
                            except Exception: pass
                            
                        if c_cfg["do_tautomers"]:
                            for base_smi in smiles_to_tautomerize:
                                try:
                                    for t_smi in enumerate_tautomers(base_smi, max_tautomers=c_cfg["max_tautomers"]):
                                        t_dict = base_dict.copy()
                                        t_dict[smiles_col] = t_smi
                                        # Distinguish if it originated from the mirror or the original
                                        t_dict["augmentation"] = "mirror_tautomer" if base_smi != smi else "tautomer"
                                        aug_rows.append(t_dict)
                                        tautomers_added += 1
                                except Exception: pass
                                
                        prog.advance(task_id)
                sub_df = pd.DataFrame(aug_rows)
                info(f"    ↳ Slice {idx+1}: Added {mirrors_added} enantiomers and {tautomers_added} tautomers.")
            else:
                sub_df["augmentation"] = "original"

            # --- NOISE INJECTION ---
            n_type = s.get("type", "normal")
            target_vals = sub_df[c_cfg["target_col"]].values
            smiles_vals = sub_df[smiles_col].values
            
            noise_arr = add_noise(target_vals, smiles_vals, n_type, noise)
            
            if noise > 0:
                sub_df[f"{c_cfg['target_col']}_noisy"] = target_vals + noise_arr
            else:
                sub_df[f"{c_cfg['target_col']}_noisy"] = target_vals.astype(float)
                    
            sub_df["noise_level"] = noise
            sub_df["noise_type"] = s.get("type", "normal") if noise > 0 else "none"
            slice_dfs.append(sub_df)
            
        if not slice_dfs:
            warn(f"Skipping {split} because absolutely no rows were available (likely entirely consumed by previous splits).")
            continue

        combined = pd.concat(slice_dfs, ignore_index=True)
        # Final mix of distinct ranges (if enabled)
        if c_cfg.get("shuffle", True):
            combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        out_name = f"{split}_prepared.csv"
        out_path = WORKSPACE / out_name
        combined.to_csv(out_path, index=False)
        state["prepared"][split] = out_name
        
        save_state(state)
        success(f"{split.capitalize()} Pipeline finished -> {out_name}")


# =============================================================================
#  Execution Pipeline (Train)
# =============================================================================

def _has_node_noise(state: dict) -> bool:
    """Return True when node noise is configured and should trigger Python API training."""
    nn = state.get("node_noise", {})
    return (nn.get("node_fraction", 0) > 0
            and nn.get("dim_fraction", 0) > 0
            and len(nn.get("layers", [])) > 0)


def _needs_python_api(state: dict) -> bool:
    """Return True when any feature requires the Python API training path."""
    return _has_node_noise(state) or state.get("hpo", {}).get("enabled", False)


def run_train_pipeline(state: dict, auto: bool = False):
    train_prep = state["prepared"].get("train")
    if not train_prep or not (WORKSPACE / train_prep).exists():
        warn("No prepared train file. Run Data Preparation Pipeline first.")
        return False

    train_csv  = str(WORKSPACE / train_prep)
    cfg        = state["config"]
    attributes = cfg.get("attributes", [cfg["target_col"]])

    val_prep = state["prepared"].get("val")
    val_csv = (str(WORKSPACE / val_prep) if val_prep and (WORKSPACE / val_prep).exists() else None)

    test_prep = state["prepared"].get("test")
    test_csv = (str(WORKSPACE / test_prep) if test_prep and (WORKSPACE / test_prep).exists() else None)

    use_python_api = _needs_python_api(state)

    hpo_cfg = state.get("hpo", {})
    hpo_enabled = hpo_cfg.get("enabled", False) and val_csv is not None

    section("Train Pipeline Configuration")
    tbl = Table(box=box.ROUNDED, border_style="cyan", expand=False, show_header=False)
    tbl.add_column("Param",   style="accent", width=22)
    tbl.add_column("Value",   style="white")
    tbl.add_row("Train CSV",  train_csv)
    tbl.add_row("Val CSV",    val_csv or "(none)")
    tbl.add_row("Test CSV",   test_csv or "(none)")
    tbl.add_row("Attributes", ", ".join(attributes))
    tbl.add_row("Epochs",     str(cfg["epochs"]))
    tbl.add_row("Batch size", str(cfg["batch_size"]))
    tbl.add_row("Seed",       str(cfg["seed"]))
    tbl.add_row("Training mode",
                "[brand]Python API[/brand]" if use_python_api
                else "[muted]Chemprop CLI[/muted]")
    if hpo_enabled:
        tbl.add_row("HPO",
                    f"[brand]ENABLED[/brand] ({hpo_cfg.get('n_trials',20)} trials, "
                    f"{hpo_cfg.get('hpo_epochs',10)} epochs/trial)")
        if hpo_cfg.get("best_params"):
            tbl.add_row("Cached best params", "[success]Available (skip HPO? will ask)[/success]")
    else:
        tbl.add_row("HPO", "[muted]OFF[/muted]")
    if _has_node_noise(state):
        nn = state["node_noise"]
        tbl.add_row("Node fraction",  f"{nn['node_fraction']*100:.1f}%")
        tbl.add_row("Dim fraction",   f"{nn['dim_fraction']*100:.1f}%")
        for i, layer in enumerate(nn["layers"]):
            tbl.add_row(f"Noise layer {i+1}", f"{layer['type']} (scale={layer['scale']:.4f})")
    console.print(Padding(tbl, (0, 2)))

    if not auto and not Confirm.ask("  [accent]Start Chemprop training?[/accent]", default=True):
        return False

    # ── HPO phase ──────────────────────────────────────────────────────
    tuned_hparams = None
    if hpo_enabled and use_python_api:
        reuse_cached = False
        if hpo_cfg.get("best_params") and not auto:
            reuse_cached = Confirm.ask(
                "  [accent]Cached HPO results found. Reuse them?[/accent]",
                default=True,
            )

        if reuse_cached:
            tuned_hparams = hpo_cfg["best_params"]
            info("Using cached best hyperparameters from previous HPO run.")
        else:
            section("Hyperparameter Optimisation")
            try:
                from hpo import run_hpo
                best_params = run_hpo(
                    train_csv=train_csv,
                    val_csv=val_csv,
                    target_col=attributes[0],  # HPO on primary attribute
                    smiles_col=cfg["smiles_col"],
                    descriptor_columns=[f"{attributes[0]}_noisy"],
                    node_noise_config=state.get("node_noise") if _has_node_noise(state) else None,
                    seed=cfg["seed"],
                    n_trials=hpo_cfg.get("n_trials", 20),
                    hpo_epochs=hpo_cfg.get("hpo_epochs", 10),
                    save_dir=str(WORKSPACE / "hpo"),
                    disable_output_scaling=cfg.get("disable_output_scaling", False),
                    disable_input_scaling=cfg.get("disable_input_scaling", False),
                )
                tuned_hparams = best_params
                state["hpo"]["best_params"] = best_params
                save_state(state)
                success("HPO complete! Best params saved.")
            except ImportError:
                error("Optuna not installed. Run: pip install optuna")
                if not Confirm.ask("  [accent]Continue with default hyperparameters?[/accent]", default=True):
                    return False
            except Exception as e:
                error(f"HPO failed: {e}")
                import traceback; traceback.print_exc()
                if not Confirm.ask("  [accent]Continue with default hyperparameters?[/accent]", default=True):
                    return False

    # ── Training phase ─────────────────────────────────────────────────
    if use_python_api:
        from gnn import train_chemprop_python

        for attr in attributes:
            section(f"Training: {attr}")
            model_dir = str(WORKSPACE / "models" / f"model_{attr}")
            try:
                train_chemprop_python(
                    train_csv=train_csv,
                    save_dir=model_dir,
                    target_col=attr,
                    epochs=cfg["epochs"],
                    batch_size=cfg["batch_size"],
                    seed=cfg["seed"],
                    val_csv=val_csv,
                    test_csv=test_csv,
                    smiles_col=cfg["smiles_col"],
                    descriptor_columns=[f"{attr}_noisy"],
                    node_noise_config=state.get("node_noise") if _has_node_noise(state) else None,
                    tuned_hparams=tuned_hparams,
                    disable_output_scaling=cfg.get("disable_output_scaling", False),
                    disable_input_scaling=cfg.get("disable_input_scaling", False),
                )
                rel = str(Path(model_dir).relative_to(WORKSPACE))
                state["models"][attr] = rel
                save_state(state)
                success(f"Model for [brand]{attr}[/brand] saved -> {model_dir}")
            except Exception as e:
                error(f"Training failed for {attr}: {e}")
                import traceback; traceback.print_exc()
                return False
    else:
        import subprocess
        try:
            subprocess.run(["chemprop", "--help"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            error("Chemprop CLI not found. Activate your conda env: conda activate chemprop")
            return False

        from gnn import train_chemprop

        for attr in attributes:
            section(f"Training: {attr}")
            model_dir = str(WORKSPACE / "models" / f"model_{attr}")
            try:
                train_chemprop(
                    train_csv=train_csv,
                    save_dir=model_dir,
                    target_col=attr,
                    epochs=cfg["epochs"],
                    batch_size=cfg["batch_size"],
                    seed=cfg["seed"],
                    val_csv=val_csv,
                    test_csv=test_csv,
                    smiles_col=cfg["smiles_col"],
                    descriptor_columns=[f"{attr}_noisy"],
                )
                rel = str(Path(model_dir).relative_to(WORKSPACE))
                state["models"][attr] = rel
                save_state(state)
                success(f"Model for [brand]{attr}[/brand] saved -> {model_dir}")
            except Exception as e:
                error(f"Training failed for {attr}: {e}")
                return False

    return True


# =============================================================================
#  Config Editor
# =============================================================================

_CONFIG_FIELDS = [
    ("smiles_col",      "SMILES column name",            "str"),
    ("target_col",      "Primary target column name",    "str"),
    ("attributes",      "Train attributes (comma-sep)",  "list"),
    ("epochs",          "Training epochs",               "int"),
    ("batch_size",      "Batch size",                    "int"),
    ("seed",            "Random seed",                   "int"),
    ("max_tautomers",   "Max tautomers per molecule",    "int"),
    ("do_mirror",       "Enable mirror augmentation",    "bool"),
    ("do_tautomers",    "Enable tautomer augmentation",  "bool"),
    ("shuffle",         "Shuffle data in preparation",   "bool"),
    ("disable_output_scaling", "Disable target output scaling", "bool"),
    ("disable_input_scaling",  "Disable descriptor input scaling", "bool"),
]

def menu_edit_config(state: dict):
    cfg = state["config"]

    while True:
        section("Edit Global Configuration")
        opts = [f"{key:<15} [{cfg.get(key)}]" for key, _, _ in _CONFIG_FIELDS]
        opts.append("-- Back --")

        idx = curses_select(opts, title="Select Config to Edit")
        if idx < 0 or idx == len(_CONFIG_FIELDS):
            break

        key, desc, kind = _CONFIG_FIELDS[idx]
        current_raw = cfg.get(key)

        if kind == "bool":
            cfg[key] = Confirm.ask(f"  [accent]{desc}[/accent]", default=bool(current_raw))
        elif kind == "int":
            cfg[key] = IntPrompt.ask(f"  [accent]{desc}[/accent]", default=int(current_raw))
        elif kind == "list":
            cur_str = ", ".join(current_raw) if current_raw else ""
            new_str = Prompt.ask(f"  [accent]{desc}[/accent]", default=cur_str)
            cfg[key] = [s.strip() for s in new_str.split(",") if s.strip()]
        else:
            cfg[key] = Prompt.ask(f"  [accent]{desc}[/accent]", default=str(current_raw))

        save_state(state)
        success(f"[brand]{key}[/brand] updated.")


# =============================================================================
#  HPO Configuration Menu
# =============================================================================

def menu_hpo(state: dict):
    """Interactive menu for configuring Optuna hyperparameter optimisation."""
    hpo = state.setdefault("hpo", {
        "enabled": False, "n_trials": 20, "hpo_epochs": 10, "best_params": None,
    })

    while True:
        clear()
        banner()
        section("Hyperparameter Optimisation (HPO)")

        enabled = hpo.get("enabled", False)
        n_trials = hpo.get("n_trials", 20)
        hpo_epochs = hpo.get("hpo_epochs", 10)
        best_params = hpo.get("best_params")
        status = "[success]ENABLED[/success]" if enabled else "[err]DISABLED[/err]"

        console.print("  [muted]Optuna-based HPO uses the validation set to automatically[/muted]")
        console.print("  [muted]find optimal model architecture and training parameters.[/muted]")
        console.print("  [muted]HPO runs before training; best params are cached for reuse.[/muted]")
        console.print()

        if best_params:
            bp_str = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in best_params.items())
            best_label = f"[success]Cached[/success]: {bp_str[:60]}{'...' if len(bp_str) > 60 else ''}"
        else:
            best_label = "[muted]None (will run HPO on next train)[/muted]"

        opts = [
            f"Toggle HPO                 [{status}]",
            f"Set number of trials       [{n_trials}]",
            f"Set epochs per trial       [{hpo_epochs}]",
            f"Cached best params         [{best_label}]",
            "Clear cached best params",
            "-- Back --",
        ]

        idx = curses_select(opts, title="HPO Configuration")
        if idx < 0 or idx == len(opts) - 1:
            break

        elif idx == 0:
            hpo["enabled"] = not hpo["enabled"]
            save_state(state)
            if hpo["enabled"]:
                success("HPO ENABLED.")
                info("Before training, Optuna will search for optimal hyperparameters.")
                info("Requires a validation set. Requires: pip install optuna")
            else:
                success("HPO DISABLED. Training uses default/manual hyperparameters.")
            pause()

        elif idx == 1:
            console.print()
            console.print("  [muted]More trials = better exploration but longer search time.[/muted]")
            console.print("  [muted]Typical: 20-50 trials for a good search. 100+ for thorough.[/muted]")
            v = IntPrompt.ask("  [accent]Number of trials[/accent]", default=n_trials)
            hpo["n_trials"] = max(5, v)
            save_state(state)

        elif idx == 2:
            console.print()
            console.print("  [muted]Epochs per trial controls how long each trial trains.[/muted]")
            console.print("  [muted]Shorter = faster search. 10-15 is a good balance.[/muted]")
            console.print("  [muted]Bad trials get pruned early by Optuna automatically.[/muted]")
            v = IntPrompt.ask("  [accent]Epochs per trial[/accent]", default=hpo_epochs)
            hpo["hpo_epochs"] = max(3, v)
            save_state(state)

        elif idx == 3:
            if best_params:
                console.print()
                from rich.table import Table as RTable
                bp_tbl = RTable(box=box.ROUNDED, border_style="green", expand=False,
                                title="[bold green]Cached Best Hyperparameters[/bold green]")
                bp_tbl.add_column("Parameter", style="accent", width=18)
                bp_tbl.add_column("Value", style="white")
                for k, v in sorted(best_params.items()):
                    bp_tbl.add_row(k, f"{v:.6g}" if isinstance(v, float) else str(v))
                console.print(Padding(bp_tbl, (0, 2)))
            else:
                info("No cached params yet. Run training to trigger HPO.")
            pause()

        elif idx == 4:
            hpo["best_params"] = None
            save_state(state)
            success("Cached best params cleared. HPO will re-run on next training.")
            pause()


# =============================================================================
#  Main Menu
# =============================================================================

# =============================================================================
#  Node Noise Configuration Menu
# =============================================================================

_NODE_NOISE_TYPES = ["normal", "uniform", "bimodal", "laplace"]

def menu_node_noise(state: dict):
    """Interactive menu for configuring node-level feature noise injection per split."""
    nn = state.setdefault("node_noise", {})
    current_split = "train"

    while True:
        clear()
        banner()
        section("Node Feature Noise")

        # Initialize split config if missing
        split_cfg = nn.setdefault(current_split, {"node_fraction": 0.0, "dim_fraction": 0.0, "layers": []})
        
        nf = split_cfg.get("node_fraction", 0.0)
        df_frac = split_cfg.get("dim_fraction", 0.0)
        layers = split_cfg.get("layers", [])
        n_layers = len(layers)

        active = nf > 0 and df_frac > 0 and n_layers > 0
        status_str = f"[success]ACTIVE[/success] currently editing {current_split}" if active else f"[muted]INACTIVE[/muted] currently editing {current_split}"

        opts = [
            f"Select Phase to configure (Current: {current_split.upper()})",
            f"Set node fraction          [{nf*100:.1f}%]",
            f"Set dimension fraction     [{df_frac*100:.1f}%]",
            f"Manage noise layers        [{n_layers} layers]",
            f"Status: {status_str}",
            "Clear all node noise settings",
            "-- Back --",
        ]

        idx = curses_select(opts, title="Node Feature Noise Config",
                            subtitle="Noise injected into atom-feature vectors dynamically during model forward pass.")
        if idx < 0 or idx == len(opts) - 1:
            break

        elif idx == 0:
            console.print()
            split_idx = curses_select(["train", "val", "test"], title="Select Phase to configure")
            if split_idx >= 0:
                current_split = ["train", "val", "test"][split_idx]

        elif idx == 1:
            console.print()
            console.print("  [muted]Fraction of atoms (nodes) to target per molecule.[/muted]")
            v = FloatPrompt.ask("  [accent]Node fraction (0.0 - 1.0)[/accent]", default=nf)
            split_cfg["node_fraction"] = max(0.0, min(1.0, v))
            save_state(state)

        elif idx == 2:
            console.print()
            console.print("  [muted]Fraction of feature-vector dimensions to perturb.[/muted]")
            v = FloatPrompt.ask("  [accent]Dimension fraction (0.0 - 1.0)[/accent]", default=df_frac)
            split_cfg["dim_fraction"] = max(0.0, min(1.0, v))
            save_state(state)

        elif idx == 3:
            menu_node_noise_layers(state, current_split)

        elif idx == 4:
            if active:
                info(f"Node noise is configured for {current_split}.")
            else:
                info("Set both fractions > 0 and add at least one noise layer to activate.")
            pause()

        elif idx == 5:
            state["node_noise"] = {}
            save_state(state)
            success("All Node noise settings cleared.")
            pause()


def menu_node_noise_layers(state: dict, current_split: str):
    """Manage the stackable noise layers for node features."""
    split_cfg = state["node_noise"].setdefault(current_split, {})
    layers = split_cfg.setdefault("layers", [])

    while True:
        clear()
        banner()
        section(f"Node Noise Layers ({current_split})")

        opts = []
        for i, layer in enumerate(layers):
            opts.append(f"Remove Layer {i+1}: {layer['type']} (scale={layer['scale']:.4f})")

        opts.append("Add new noise layer")
        opts.append("-- Done --")

        idx = curses_select(opts, title="Stacked Noise Layers",
                            subtitle="Layers are applied additively in order to the selected entries")
        if idx < 0 or idx == len(opts) - 1:
            break

        if idx < len(layers):
            # Remove selected layer
            removed = layers.pop(idx)
            save_state(state)
            success(f"Removed layer: {removed['type']} (scale={removed['scale']:.4f})")
            pause()
        else:
            # Add new layer
            console.print()
            t_idx = curses_select(_NODE_NOISE_TYPES, title="Select Noise Distribution")
            if t_idx < 0:
                continue
            noise_type = _NODE_NOISE_TYPES[t_idx]

            console.print()
            scale = FloatPrompt.ask(
                f"  [accent]Noise scale for {noise_type}[/accent]",
                default=0.1,
            )
            if scale <= 0:
                warn("Scale must be > 0.")
                pause()
                continue

            layers.append({"type": noise_type, "scale": scale})
            save_state(state)
            success(f"Added layer: {noise_type} (scale={scale:.4f})")
            pause()


# =============================================================================
#  Scaffold Split Configuration Menu
# =============================================================================

def menu_scaffold_split(state: dict):
    """Interactive menu for configuring scaffold-based train/val/test splitting."""
    sc = state.setdefault("scaffold_split", {
        "enabled": False, "train_size": 0.8, "val_size": 0.1,
        "test_size": 0.1, "use_generic": False,
    })

    while True:
        clear()
        banner()
        section("Scaffold Split Configuration")

        enabled   = sc.get("enabled", False)
        tr_sz     = sc.get("train_size", 0.8)
        va_sz     = sc.get("val_size", 0.1)
        te_sz     = sc.get("test_size", 0.1)
        generic   = sc.get("use_generic", False)
        status    = "[success]ENABLED[/success]" if enabled else "[err]DISABLED[/err]"
        gen_label = "Generic (rings only)" if generic else "Murcko (with heteroatoms)"

        console.print("  [muted]Scaffold splitting ensures the test set contains[/muted]")
        console.print("  [muted]chemical scaffolds NOT seen during training, testing[/muted]")
        console.print("  [muted]the model's ability to generalise to novel structures.[/muted]")
        console.print()

        opts = [
            f"Toggle scaffold split      [{status}]",
            f"Set split ratios           [Train {tr_sz*100:.0f}% / Val {va_sz*100:.0f}% / Test {te_sz*100:.0f}%]",
            f"Toggle scaffold type       [{gen_label}]",
            "-- Back --",
        ]

        idx = curses_select(opts, title="Scaffold Split")
        if idx < 0 or idx == len(opts) - 1:
            break

        elif idx == 0:
            sc["enabled"] = not sc["enabled"]
            save_state(state)
            if sc["enabled"]:
                success("Scaffold split ENABLED.")
                info("During preparation, the Train base CSV will be scaffold-split")
                info("into train/val/test before noise and augmentation.")
            else:
                success("Scaffold split DISABLED. Using manual per-split CSVs.")
            pause()

        elif idx == 1:
            console.print()
            console.print("  [muted]Enter ratios that sum to 1.0[/muted]")
            new_tr = FloatPrompt.ask("  [accent]Train fraction[/accent]", default=tr_sz)
            new_va = FloatPrompt.ask("  [accent]Validation fraction[/accent]", default=va_sz)
            new_te = FloatPrompt.ask("  [accent]Test fraction[/accent]", default=te_sz)
            total = new_tr + new_va + new_te
            if abs(total - 1.0) > 0.01:
                warn(f"Ratios sum to {total:.2f}, not 1.0. Normalising...")
                new_tr /= total
                new_va /= total
                new_te /= total
            sc["train_size"] = round(new_tr, 3)
            sc["val_size"]   = round(new_va, 3)
            sc["test_size"]  = round(new_te, 3)
            save_state(state)
            success(f"Split ratios: {sc['train_size']*100:.0f}% / {sc['val_size']*100:.0f}% / {sc['test_size']*100:.0f}%")
            pause()

        elif idx == 2:
            sc["use_generic"] = not sc["use_generic"]
            save_state(state)
            if sc["use_generic"]:
                info("Generic scaffolds: heteroatoms replaced with carbon, side chains removed.")
                info("Results in fewer, larger scaffold clusters (stricter generalisation test).")
            else:
                info("Murcko scaffolds: preserves heteroatoms in the core ring system.")
                info("More fine-grained scaffold groups (moderate generalisation test).")
            pause()


def menu_preparation(state: dict):
    while True:
        sc_status = "ON" if state.get("scaffold_split", {}).get("enabled") else "OFF"
        opts = [
            "Configure Train Split",
            "Configure Test Split",
            "Configure Validation Split",
            f"Configure Scaffold Split    [{sc_status}]",
            "Configure Node Feature Noise",
            "Run Data Preparation Pipeline",
            "-- Back --"
        ]
        
        idx = curses_select(opts, title="Preparation Pipeline")
        if idx < 0 or idx == 6:
            break
            
        elif idx == 0: menu_pipeline_split(state, "train")
        elif idx == 1: menu_pipeline_split(state, "test")
        elif idx == 2: menu_pipeline_split(state, "val")
        elif idx == 3: menu_scaffold_split(state)
        elif idx == 4: menu_node_noise(state)
        elif idx == 5:
            run_preparation_pipeline(state)
            pause()


def full_reset(state: dict):
    if Confirm.ask("  [err]Are you sure you want to delete the whole workspace?[/err]", default=False):
        try:
            shutil.rmtree(WORKSPACE)
            return copy.deepcopy(DEFAULT_STATE)
        except Exception:
            pass
    return state


def main():
    clear()
    state = load_state()

    while True:
        clear()
        banner()
        
        hpo_status = "ON" if state.get("hpo", {}).get("enabled") else "OFF"
        opts = [
            "Overview",
            "Preparation Pipeline (Noise -> Merge -> Augment)",
            "Train Pipeline (Chemprop)",
            f"Hyperparameter Optimisation  [{hpo_status}]",
            "Run Full Auto-Pipeline (Prepare + Train)",
            "Settings / Config",
            "Reset Workspace",
            "Quit"
        ]
        
        idx = curses_select(opts, title="Main Menu", subtitle=f"Workspace: {WORKSPACE}")
        
        if idx < 0 or idx == 7:
            console.print("\n  [muted]Goodbye![/muted]\n")
            break
            
        elif idx == 0:
            print_status(state)
            pause()
        elif idx == 1:
            menu_preparation(state)
        elif idx == 2:
            run_train_pipeline(state)
            pause()
        elif idx == 3:
            menu_hpo(state)
        elif idx == 4:
            run_preparation_pipeline(state)
            run_train_pipeline(state, auto=True)
            pause()
        elif idx == 5:
            menu_edit_config(state)
        elif idx == 6:
            state = full_reset(state)

if __name__ == "__main__":
    main()
