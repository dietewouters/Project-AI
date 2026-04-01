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
    console.print(Panel(cfg_tbl, title="[hi]Config[/hi]", border_style="cyan", expand=False))

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
            
        # 3. Shuffle completely before slicing (if enabled)
        if c_cfg.get("shuffle", True):
            df = df.sample(frac=1, random_state=c_cfg["seed"]).reset_index(drop=True)
        
        slices = eff_slices
        if not slices:
            slices = [{"fraction": 1.0, "noise": 0.0}]
            
        slice_dfs = []
        current_idx = 0
        
        for idx, s in enumerate(slices):
            frac, noise = s["fraction"], s["noise"]
            n_rows = int(original_len * frac)  # Calculate proportion based on original size
            
            if current_idx + n_rows > remaining_len:
                warn_msg = f"Not enough unused rows remain in {base_path.name} to fulfill slice {idx+1} of {split}. Using remaining {remaining_len - current_idx} rows."
                warn(warn_msg)
                n_rows = remaining_len - current_idx
                
            if n_rows <= 0:
                continue
                
            sub_df = df.iloc[current_idx:current_idx + n_rows].copy()
            current_idx += n_rows
            
            # Register claimed SMILES globally
            global_used_smiles.update(sub_df[smiles_col].tolist())
            
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
    console.print(Padding(tbl, (0, 2)))

    if not auto and not Confirm.ask("  [accent]Start Chemprop training?[/accent]", default=True):
        return False

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
    ("smiles_col",    "SMILES column name",            "str"),
    ("target_col",    "Primary target column name",    "str"),
    ("attributes",    "Train attributes (comma-sep)",  "list"),
    ("epochs",        "Training epochs",               "int"),
    ("batch_size",    "Batch size",                    "int"),
    ("seed",          "Random seed",                   "int"),
    ("max_tautomers", "Max tautomers per molecule",    "int"),
    ("do_mirror",     "Enable mirror augmentation",    "bool"),
    ("do_tautomers",  "Enable tautomer augmentation",  "bool"),
    ("shuffle",       "Shuffle data in preparation",   "bool"),
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
#  Main Menu
# =============================================================================

def menu_preparation(state: dict):
    while True:
        opts = [
            "Configure Train Split",
            "Configure Test Split",
            "Configure Validation Split",
            "Run Data Preparation Pipeline",
            "-- Back --"
        ]
        
        idx = curses_select(opts, title="Preparation Pipeline")
        if idx < 0 or idx == 4:
            break
            
        elif idx == 0: menu_pipeline_split(state, "train")
        elif idx == 1: menu_pipeline_split(state, "test")
        elif idx == 2: menu_pipeline_split(state, "val")
        elif idx == 3:
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
        
        opts = [
            "Overview",
            "Preparation Pipeline (Noise -> Merge -> Augment)",
            "Train Pipeline (Chemprop)",
            "Run Full Auto-Pipeline (Prepare + Train)",
            "Settings / Config",
            "Reset Workspace",
            "Quit"
        ]
        
        idx = curses_select(opts, title="Main Menu", subtitle=f"Workspace: {WORKSPACE}")
        
        if idx < 0 or idx == 6:
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
            run_preparation_pipeline(state)
            run_train_pipeline(state, auto=True)
            pause()
        elif idx == 4:
            menu_edit_config(state)
        elif idx == 5:
            state = full_reset(state)

if __name__ == "__main__":
    main()
