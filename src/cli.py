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


for _pkg in ("rich", "pandas", "scipy", "numpy", "openpyxl", "matplotlib"):
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
    "truth": {
        # Reference table used to compute deterministic h298 for any SMILES.
        # Default is the bundled 7.9M-row group-additivity dataset.
        "ga_csv":      str(DEFAULT_GA_DIR / "groupadditivity.csv"),
        "smiles_col":  "smiles",
        "target_col":  "h298",
        "train_frac":  0.7,
        "val_frac":    0.15,
        # test_frac is implicitly (1 - train_frac - val_frac)
        "split_seed":  42,
        # Auto-derived: a CSV of (smiles, h298) covering only those noisy
        # SMILES that were found in the GA reference. Cached in the workspace.
        "derived_csv": None,
        "coverage":    None,    # {"matched": int, "total": int}
    },
    "noisy": {
        "mode":            "derived",     # "external" or "derived"
        "source_csv":      None,
        "smiles_col":      "smiles",
        "target_col":      "h298",
        "subset_fraction": 1.0,
        "subset_seed":     42,
        "noise_layers":    [],            # only used when mode == "derived"
        "noise_seed":      42,
        "rendered_csv":    None,          # cached path produced by the last render
    },
    "knn": {
        "k":           10,
        "rounds":      5,
    },
    "train": {
        "epochs":      30,
        "batch_size":  128,
        "seed":        42,
        "model_dir":   str(WORKSPACE / "embedder"),
        "trained":     False,
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
            # Migrate legacy schemas
            nz = st.get("noisy", {})
            if "csv" in nz and not nz.get("source_csv"):
                nz["source_csv"] = nz.get("csv")
                nz["mode"] = "external"
            tr = st.get("truth", {})
            if "csv" in tr and not tr.get("ga_csv"):
                tr["ga_csv"] = str(DEFAULT_GA_DIR / "groupadditivity.csv")
                tr.pop("csv", None)
            tr.pop("csv", None)
            emb = st.pop("embedder", None)
            if emb and "model_dir" in emb and "model_dir" not in st.get("train", {}):
                st["train"]["model_dir"] = emb["model_dir"]
            if emb and emb.get("trained") and not st["train"].get("trained"):
                st["train"]["trained"] = True
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

# Color pair IDs.
_C_BRAND   = 1
_C_ACCENT  = 2
_C_MUTED   = 3
_C_SUCCESS = 4
_C_SELECT  = 5
_C_BORDER  = 6


def _init_colors() -> bool:
    """Initialise curses colour pairs. Returns True if colour is available."""
    if not curses.has_colors():
        return False
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except Exception:
        bg = curses.COLOR_BLACK
    curses.init_pair(_C_BRAND,   curses.COLOR_CYAN,    bg)
    curses.init_pair(_C_ACCENT,  curses.COLOR_MAGENTA, bg)
    curses.init_pair(_C_MUTED,   curses.COLOR_WHITE,   bg)
    curses.init_pair(_C_SUCCESS, curses.COLOR_GREEN,   bg)
    curses.init_pair(_C_SELECT,  curses.COLOR_BLACK,   curses.COLOR_CYAN)
    curses.init_pair(_C_BORDER,  curses.COLOR_CYAN,    bg)
    return True


def _addstr(stdscr, y: int, x: int, text: str, attr: int = 0, max_w: int | None = None):
    """Safe addstr that clips and swallows boundary errors."""
    try:
        h, w = stdscr.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        avail = w - x - 1
        if max_w is not None:
            avail = min(avail, max_w)
        if avail <= 0:
            return
        stdscr.addnstr(y, x, text, avail, attr)
    except curses.error:
        pass


def curses_select(items: list[str], title: str = "", subtitle: str = "") -> int:
    """Arrow-key + Enter selector. Returns the chosen index or -1 on Esc/q."""
    def _impl(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        has_color = _init_colors()

        def attr(pair: int, *flags: int) -> int:
            base = curses.color_pair(pair) if has_color else 0
            for f in flags:
                base |= f
            return base

        a_brand   = attr(_C_BRAND,  curses.A_BOLD)
        a_accent  = attr(_C_ACCENT, curses.A_BOLD)
        a_muted   = attr(_C_MUTED,  curses.A_DIM)
        a_border  = attr(_C_BORDER)
        a_arrow   = attr(_C_ACCENT, curses.A_BOLD)
        a_picked  = attr(_C_BRAND,  curses.A_BOLD)

        idx = 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            # ── top banner ────────────────────────────────────────────
            inner_w = max(40, min(w - 4, 64))
            box_top    = "╭" + "─" * (inner_w - 2) + "╮"
            box_bot    = "╰" + "─" * (inner_w - 2) + "╯"

            brand_text = f"◆ MolAugment"
            ver_text   = f"v{VERSION}"
            tagline    = "Train an embedder, then denoise via k-NN"

            row = 1
            _addstr(stdscr, row, 2, box_top, a_border)
            row += 1
            # brand line: "│  ◆ MolAugment              v4.0.0  │"
            pad = inner_w - 2 - len(brand_text) - len(ver_text) - 4
            pad = max(1, pad)
            _addstr(stdscr, row, 2, "│", a_border)
            _addstr(stdscr, row, 4, brand_text, a_brand)
            _addstr(stdscr, row, 4 + len(brand_text) + pad, ver_text, a_muted)
            _addstr(stdscr, row, 2 + inner_w - 1, "│", a_border)
            row += 1
            # tagline
            _addstr(stdscr, row, 2, "│", a_border)
            _addstr(stdscr, row, 4, tagline[: inner_w - 6], a_muted)
            _addstr(stdscr, row, 2 + inner_w - 1, "│", a_border)
            row += 1
            _addstr(stdscr, row, 2, box_bot, a_border)
            row += 2

            # ── title / subtitle (subtitle may be multi-line) ─────────
            if title:
                _addstr(stdscr, row, 2, "▌ ", a_accent)
                _addstr(stdscr, row, 4, title, a_accent)
                row += 1
            if subtitle:
                for sub_line in subtitle.split("\n"):
                    if not sub_line.strip():
                        row += 1
                        continue
                    _addstr(stdscr, row, 4, sub_line, a_muted)
                    row += 1
            row += 1

            # ── menu items ────────────────────────────────────────────
            for i, item in enumerate(items):
                if row >= h - 2:
                    break
                if i == idx:
                    _addstr(stdscr, row, 2, "  ▸ ", a_arrow)
                    _addstr(stdscr, row, 6, item, a_picked)
                else:
                    _addstr(stdscr, row, 2, "    ", 0)
                    _addstr(stdscr, row, 6, item, 0)
                row += 1

            # ── footer ────────────────────────────────────────────────
            footer = "  ↑/↓ navigate     ⏎ select     q/Esc quit"
            _addstr(stdscr, h - 1, 0, footer, a_muted)

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


# ── arrow-key file browser (curses) ───────────────────────────────────────

def curses_file_picker(
    start_dir: str | Path | None = None,
    extensions: list[str] | None = None,
    title: str = "Select a file",
) -> str | None:
    """Curses file browser. Returns absolute path string, or None on cancel."""
    start = Path(start_dir).expanduser() if start_dir else Path.cwd()
    try:
        start = start.resolve()
    except Exception:
        start = Path.cwd()
    while not start.is_dir():
        if start.parent == start:
            start = Path.cwd()
            break
        start = start.parent

    exts = {e.lower() for e in extensions} if extensions else None

    def _entries(d: Path) -> list[tuple[str, Path | None]]:
        """Return [(label, path|None)]; None path means parent dir."""
        out: list[tuple[str, Path | None]] = [("../", None)]
        try:
            kids = list(d.iterdir())
        except PermissionError:
            return out
        kids.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
        dirs, files = [], []
        for p in kids:
            if p.name.startswith("."):
                continue
            try:
                if p.is_dir():
                    dirs.append((p.name + "/", p))
                elif exts is None or p.suffix.lower() in exts:
                    files.append((p.name, p))
            except OSError:
                continue
        out.extend(dirs)
        out.extend(files)
        return out

    state_box = {"cur": start}

    def _impl(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        has_color = _init_colors()

        def attr(pair: int, *flags: int) -> int:
            base = curses.color_pair(pair) if has_color else 0
            for f in flags:
                base |= f
            return base

        a_accent = attr(_C_ACCENT, curses.A_BOLD)
        a_muted  = attr(_C_MUTED,  curses.A_DIM)
        a_arrow  = attr(_C_ACCENT, curses.A_BOLD)
        a_dir    = attr(_C_BRAND,  curses.A_BOLD)
        a_picked = attr(_C_BRAND,  curses.A_BOLD)
        a_file   = 0

        idx = 0
        offset = 0

        while True:
            cur = state_box["cur"]
            entries = _entries(cur)
            if idx >= len(entries):
                idx = max(0, len(entries) - 1)

            stdscr.erase()
            h, w = stdscr.getmaxyx()

            # Header
            row = 1
            _addstr(stdscr, row, 2, "▌ ", a_accent)
            _addstr(stdscr, row, 4, title, a_accent)
            row += 1

            # Current path
            path_str = str(cur)
            if len(path_str) > w - 6:
                path_str = "…" + path_str[-(w - 7):]
            _addstr(stdscr, row, 4, path_str, a_muted)
            row += 1

            if exts:
                _addstr(stdscr, row, 4, "filter: " + " ".join(sorted(exts)), a_muted)
                row += 1
            row += 1

            # Visible window — leave footer space
            visible_h = max(1, h - row - 2)
            if idx < offset:
                offset = idx
            elif idx >= offset + visible_h:
                offset = idx - visible_h + 1

            for i, (label, path) in enumerate(entries[offset:offset + visible_h]):
                real_i = i + offset
                is_dir = path is None or path.is_dir()
                if real_i == idx:
                    _addstr(stdscr, row + i, 2, "  ▸ ", a_arrow)
                    label_attr = a_picked
                else:
                    _addstr(stdscr, row + i, 2, "    ", 0)
                    label_attr = a_dir if is_dir else a_file
                _addstr(stdscr, row + i, 6, label, label_attr)

            footer = "  ↑/↓ navigate     ⏎ open/select     ← parent     q/Esc cancel"
            _addstr(stdscr, h - 1, 0, footer, a_muted)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(entries)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(entries)
            elif key in (curses.KEY_LEFT, ord("h"), curses.KEY_BACKSPACE, 127):
                parent = state_box["cur"].parent
                if parent != state_box["cur"]:
                    state_box["cur"] = parent
                idx, offset = 0, 0
            elif key in (curses.KEY_RIGHT, curses.KEY_ENTER, ord("\n"), ord("\r"), ord("l")):
                label, path = entries[idx]
                if path is None:
                    parent = state_box["cur"].parent
                    if parent != state_box["cur"]:
                        state_box["cur"] = parent
                    idx, offset = 0, 0
                elif path.is_dir():
                    state_box["cur"] = path
                    idx, offset = 0, 0
                else:
                    return str(path)
            elif key in (27, ord("q")):
                return None

    try:
        return curses.wrapper(_impl)
    except Exception:
        return None


# ── selectors for column names / fractions ───────────────────────────────

def _pick_column(csv_path: str | None, label: str, default: str | None = None) -> str | None:
    """Curses-pick a column from a CSV. Falls back to typed input if the file
    can't be read or the user picks 'type custom'."""
    cols: list[str] = []
    if csv_path:
        try:
            head = pd.read_csv(csv_path, nrows=1)
            cols = list(head.columns)
        except Exception:
            cols = []
    if not cols:
        clear()
        banner()
        return Prompt.ask(f"  {label}", default=default or "")

    ordered = list(cols)
    if default and default in ordered:
        ordered.remove(default)
        ordered.insert(0, default)

    options = ordered + ["[type custom]"]
    subtitle = f"in {Path(csv_path).name}" if csv_path else ""
    idx = curses_select(options, title=label, subtitle=subtitle)
    if idx < 0:
        return default if default in cols else (cols[0] if cols else None)
    if idx == len(options) - 1:
        clear()
        banner()
        return Prompt.ask(f"  {label}", default=default or "")
    return ordered[idx]


def _pick_fraction(default: float = 0.1, label: str = "Fraction") -> float:
    """Curses-pick a fraction from common presets; supports custom input."""
    presets = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    labels = [f"{p:>6.0%}   ({p:g})" for p in presets] + ["[type custom]"]
    idx = curses_select(
        labels, title=label, subtitle="portion to keep / apply",
    )
    if idx < 0:
        return default
    if idx == len(labels) - 1:
        clear()
        banner()
        try:
            return float(Prompt.ask(f"  {label} (0-1]", default=str(default)))
        except ValueError:
            return default
    return presets[idx]


# ── noise / subset configurator ───────────────────────────────────────────

def _summarize_layer(layer: dict) -> str:
    t = layer["type"]
    s = layer.get("scale", 0)
    if t == "outlier":
        f = layer.get("fraction", 0.05)
        return f"{t:<14} scale={s:g}  fraction={f:.0%}"
    return f"{t:<14} scale={s:g}"


def _layer_param_prompt(type_: str, existing: dict | None = None) -> dict:
    """Prompt for the params of a single noise layer. Uses rich Prompt."""
    base = existing or {}
    section(f"Configure {type_} noise")
    info(f"description: {__noise_descr(type_)}")
    console.print()
    scale = float(Prompt.ask("  scale", default=str(base.get("scale", 0.1))))
    layer = {"type": type_, "scale": scale}
    if type_ == "outlier":
        layer["fraction"] = float(
            Prompt.ask("  fraction (0-1]", default=str(base.get("fraction", 0.05)))
        )
    return layer


def __noise_descr(type_: str) -> str:
    import noise as noise_mod
    return noise_mod.NOISE_TYPE_DESCRIPTIONS.get(type_, "")


def _pick_noise_type() -> str | None:
    import noise as noise_mod
    labels = [
        f"{t:<16} — {noise_mod.NOISE_TYPE_DESCRIPTIONS.get(t, '')}"
        for t in noise_mod.NOISE_TYPES
    ]
    idx = curses_select(
        labels,
        title="Choose noise type",
        subtitle=(
            "each type adds a different shape of noise to the target column.\n"
            "scale roughly controls the standard deviation of the contribution."
        ),
    )
    if idx < 0:
        return None
    return noise_mod.NOISE_TYPES[idx]


def _configure_layers(initial: list[dict] | None = None) -> list[dict]:
    """Interactive sub-menu to add/edit/remove noise layers. Returns the final list,
    or the original list unchanged on cancel."""
    layers = [dict(l) for l in (initial or [])]
    original = [dict(l) for l in (initial or [])]
    while True:
        rows = [f"{i+1}. {_summarize_layer(l)}" for i, l in enumerate(layers)]
        if not rows:
            rows = ["(no layers yet)"]
        rows = rows + ["[+] Add layer", "[✓] Done", "[✗] Cancel"]
        idx = curses_select(
            rows,
            title="Noise layers",
            subtitle=(
                f"{len(layers)} layer(s) — additive contributions stacked together.\n"
                "each layer has a type (normal/uniform/outlier/...), a scale, and\n"
                "its own deterministic seed. pick one to edit or remove."
            ),
        )
        n_layers = len(layers)
        # Map menu indices: 0..n_layers-1 are layers (or 0 is the placeholder),
        # n_layers (or 1 if empty) is Add, then Done, then Cancel.
        offset = max(1, n_layers)
        if idx < 0 or idx == offset + 2:
            return original
        if idx == offset + 1:
            return layers
        if idx == offset:
            type_ = _pick_noise_type()
            if type_ is None:
                continue
            clear()
            banner()
            try:
                layers.append(_layer_param_prompt(type_))
            except Exception as e:
                error(f"invalid input: {e}")
                pause()
            continue
        if n_layers == 0:
            continue  # placeholder row, not selectable
        layer_idx = idx
        sub = curses_select(
            ["Edit", "Remove", "Cancel"],
            title=f"Layer {layer_idx + 1}",
            subtitle=_summarize_layer(layers[layer_idx]),
        )
        if sub == 0:
            clear()
            banner()
            try:
                layers[layer_idx] = _layer_param_prompt(
                    layers[layer_idx]["type"], existing=layers[layer_idx]
                )
            except Exception as e:
                error(f"invalid input: {e}")
                pause()
        elif sub == 1:
            del layers[layer_idx]


def _ensure_csv(path: str | Path) -> str:
    """If the path is an Excel file, convert (and cache) it to CSV. Returns the
    CSV path. Pass-through for `.csv` and unknown suffixes.

    If most columns come back as ``Unnamed: N`` (the spreadsheet has a blank
    leading row before real headers), the converter retries with ``header=1``."""
    p = Path(path)
    if p.suffix.lower() not in (".xlsx", ".xls"):
        return str(p)
    out_dir = WORKSPACE / "converted"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (p.stem + ".csv")
    if out.exists() and out.stat().st_mtime >= p.stat().st_mtime:
        return str(out)

    def _frac_unnamed(cols) -> float:
        if len(cols) == 0:
            return 1.0
        return sum(1 for c in cols if str(c).startswith("Unnamed:")) / len(cols)

    try:
        df = pd.read_excel(p, header=0)
        if _frac_unnamed(df.columns) > 0.5:
            # blank/empty leading row — retry with the next row as header
            for hdr_row in (1, 2):
                df_try = pd.read_excel(p, header=hdr_row)
                if _frac_unnamed(df_try.columns) <= 0.5:
                    df = df_try
                    info(f"detected blank leading row(s) — using row {hdr_row + 1} as header")
                    break
    except Exception as e:
        error(f"could not read {p.name}: {e}")
        return str(p)

    df.to_csv(out, index=False)
    info(f"converted {p.name} → {_short(str(out))}  ({len(df)} rows)")
    return str(out)


def _pick_csv(label: str, current: str | None) -> str | None:
    """File picker for CSV / XLSX inputs. Excel files are auto-converted to a
    cached CSV in the workspace so the rest of the pipeline only sees CSVs.
    Returns the (possibly converted) path or `current` on cancel."""
    if current and Path(current).exists():
        start = Path(current).parent
    elif current:
        p = Path(current)
        while p != p.parent and not p.exists():
            p = p.parent
        start = p
    else:
        start = REPO_ROOT
    picked = curses_file_picker(
        start_dir=start,
        extensions=[".csv", ".xlsx", ".xls"],
        title=label,
    )
    if not picked:
        return current
    return _ensure_csv(picked)


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
    truth = state["truth"]
    nz = state["noisy"]
    knn = state["knn"]
    trn = state["train"]
    hpo = state["hpo"]
    last = state["last_run"]

    tbl = Table(box=box.SIMPLE_HEAVY, expand=False, border_style="cyan")
    tbl.add_column("Section", style="accent", width=14)
    tbl.add_column("Setting", style="muted", width=18)
    tbl.add_column("Value")

    test_pct = max(0.0, 1.0 - truth["train_frac"] - truth["val_frac"])
    cov = truth.get("coverage")
    cov_label = (
        f"{cov['matched']:,} / {cov['total']:,}  ({cov['matched']/max(1,cov['total']):.0%})"
        if cov else "[muted]<not derived>[/muted]"
    )
    tbl.add_row("Truth",    "GA reference", _short(truth["ga_csv"]))
    tbl.add_row("",         "derived CSV",  _short(truth.get("derived_csv")))
    tbl.add_row("",         "coverage",     cov_label)
    tbl.add_row("",         "split",
                f"train {truth['train_frac']:.0%} / val {truth['val_frac']:.0%} / test {test_pct:.0%}")
    tbl.add_row("",         "split seed",   str(truth["split_seed"]))

    tbl.add_row("Noisy",    "mode",        nz["mode"])
    tbl.add_row("",         "source CSV",  _short(nz["source_csv"]))
    tbl.add_row("",         "smiles col",  nz["smiles_col"])
    tbl.add_row("",         "target col",  nz["target_col"])
    tbl.add_row("",         "subset",      f"{nz['subset_fraction']:.0%}  (seed {nz['subset_seed']})")
    if nz["mode"] == "derived":
        tbl.add_row("",     "noise layers", str(len(nz["noise_layers"])))
    tbl.add_row("",         "rendered",    _short(nz.get("rendered_csv")))

    tbl.add_row("Embedder", "trained",     "[success]yes[/success]" if trn.get("trained") else "[warn]no[/warn]")
    tbl.add_row("",         "model dir",   _short(trn["model_dir"]))

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


# ── helpers: truth split + noisy render ───────────────────────────────────

def _derive_truth(state: dict, verbose: bool = True) -> Path | None:
    """Compute GA-derived h298 for the noisy SMILES and write a truth CSV.

    Coefficients are fitted once from the bundled reference (Morgan radius-1
    atom environments → h298) and cached. After fitting, predictions work on
    any SMILES whose fragments are all in the fitted vocabulary — molecules
    with elements outside that set (e.g. S, P, halogens) remain unpredictable
    and are dropped with a coverage report.

    Returns the derived path, or None on failure. Updates
    ``state['truth']['derived_csv']`` and ``coverage``."""
    import ga_compute as ga_mod
    truth = state["truth"]
    nz = state["noisy"]

    rendered = nz.get("rendered_csv")
    if not rendered or not Path(rendered).exists():
        rendered = _render_noisy(state, verbose=verbose)
        if not rendered:
            return None

    df = pd.read_csv(rendered)
    if nz["smiles_col"] not in df.columns:
        if verbose:
            error(f"column {nz['smiles_col']!r} not in rendered noisy CSV")
        return None
    smiles = df[nz["smiles_col"]].astype(str).tolist()

    if verbose:
        section("Compute ground truth (group-additivity)")
        info(f"reference (fitting source): {_short(truth['ga_csv'])}")
        info(f"computing h298 for {len(smiles):,} SMILES…")

    def _cb(stage: str, detail):
        if not verbose:
            return
        if stage == "loading_cache":
            info("loaded cached GA model (lookup + fitted coefficients)")
        elif stage == "loading":
            info(f"first run — building GA model from {Path(detail).name}…")
        elif stage == "featurizing":
            info(f"extracting fragments from {detail:,} training molecules…")
        elif stage == "solving":
            info(f"sparse least-squares ({detail[0]:,}×{detail[1]:,})…")
        elif stage == "done":
            info(f"lookup table: {detail['lookup_size']:,} molecules  |  "
                 f"fitted coefs: {detail['n_frags']:,} fragments  |  "
                 f"fit RMSE {detail['rmse']:.3f} kcal/mol")

    try:
        model = ga_mod.get_or_fit_model(
            truth["ga_csv"],
            cache_dir=WORKSPACE / "ga_cache",
            progress_cb=_cb,
        )
    except Exception as e:
        if verbose:
            error(f"could not fit / load GA coefficients: {e}")
        return None

    out_dir = WORKSPACE / "truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{Path(rendered).stem}__truth.csv"

    try:
        path, matched, total = ga_mod.derive_truth_csv(
            smiles, model, out_csv,
            smiles_col=truth["smiles_col"], target_col=truth["target_col"],
        )
    except Exception as e:
        if verbose:
            error(f"derive failed: {e}")
        return None

    truth["derived_csv"] = str(path) if matched > 0 else None
    truth["coverage"] = {"matched": int(matched), "total": int(total)}
    save_state(state)

    pct = matched / max(1, total)
    if matched == 0:
        if verbose:
            error("no molecules could be GA-computed — every input contains a "
                  "fragment outside the fitted vocabulary (likely an element "
                  "the reference doesn't cover)")
        return None
    if verbose:
        if matched < total:
            warn(f"computed {matched:,}/{total:,} ({pct:.1%}) — "
                 f"{total - matched:,} dropped (unknown fragments / atoms)")
        else:
            success(f"computed {matched:,}/{total:,} ({pct:.1%})")
    return path


def _ensure_truth_split(state: dict) -> tuple[Path, Path]:
    """Read the (auto-derived) truth CSV and split into train/val. Returns paths."""
    truth = state["truth"]
    cached = truth.get("derived_csv")
    if not cached or not Path(cached).exists():
        derived = _derive_truth(state, verbose=True)
        if not derived:
            raise FileNotFoundError("could not auto-derive truth from noisy SMILES")
        cached = str(derived)

    csv = Path(cached)
    df = pd.read_csv(csv)
    if truth["smiles_col"] not in df.columns or truth["target_col"] not in df.columns:
        raise ValueError(
            f"derived truth CSV missing column(s): "
            f"{truth['smiles_col']!r}, {truth['target_col']!r}"
        )
    rng = np.random.default_rng(int(truth["split_seed"]))
    perm = rng.permutation(len(df))
    n = len(df)
    n_train = max(1, int(n * float(truth["train_frac"])))
    n_val   = max(1, int(n * float(truth["val_frac"])))
    if n_train + n_val >= n:
        n_train = max(1, int(n * 0.7))
        n_val   = max(1, int(n * 0.15))
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]

    split_dir = WORKSPACE / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.csv"
    val_path   = split_dir / "val.csv"
    df.iloc[train_idx].reset_index(drop=True).to_csv(train_path, index=False)
    df.iloc[val_idx].reset_index(drop=True).to_csv(val_path, index=False)
    return train_path, val_path


def _render_noisy(state: dict, verbose: bool = True) -> str | None:
    """Materialise state.noisy into a CSV that run_denoise can read.

    Returns the rendered path or None on failure. Also caches the result in
    ``state['noisy']['rendered_csv']``."""
    import noise as noise_mod
    nz = state["noisy"]

    src = nz.get("source_csv")
    if not src or not Path(src).exists():
        if verbose:
            error("source CSV is not set or missing")
        return None

    layers = list(nz["noise_layers"]) if nz["mode"] == "derived" else []
    fraction = max(min(float(nz.get("subset_fraction", 1.0)), 1.0), 1e-6)

    # Shortcut: external CSV, no subset, no layers → use the source directly.
    if nz["mode"] == "external" and fraction >= 1.0 and not layers:
        nz["rendered_csv"] = src
        save_state(state)
        if verbose:
            info(f"using external CSV as-is → {_short(src)}")
        return src

    seed = int(nz.get("noise_seed", 42)) if layers else int(nz.get("subset_seed", 42))
    dst_name = noise_mod.derived_filename(src, fraction, layers, seed)
    dst = WORKSPACE / "derived" / dst_name

    if verbose:
        section("Render noisy dataset")
        console.print(
            "  [muted]Materialises the noisy CSV that kNN will read. Subset is applied[/muted]\n"
            "  [muted]first (deterministic by seed), then noise layers are summed onto[/muted]\n"
            "  [muted]the target column. Original SMILES are preserved.[/muted]\n"
        )
        info(f"source : {_short(src)}")
        info(f"output : {_short(str(dst))}")
        info(f"fraction={fraction:g}   layers={len(layers)}   seed={seed}")

    try:
        df = noise_mod.derive_dataset(
            src_csv=src, dst_csv=str(dst),
            smiles_col=nz["smiles_col"], target_col=nz["target_col"],
            fraction=fraction, layers=layers, seed=seed,
        )
        if verbose:
            success(f"{len(df)} rows written")
    except Exception as e:
        if verbose:
            error(f"render failed: {e}")
        return None

    nz["rendered_csv"] = str(dst)
    save_state(state)
    return str(dst)


# ── menu: configure ground-truth dataset ──────────────────────────────────

def menu_configure_truth(state: dict) -> None:
    """Sub-menu for the (auto-derived) ground-truth dataset."""
    truth = state["truth"]
    while True:
        test_pct = max(0.0, 1.0 - truth["train_frac"] - truth["val_frac"])
        cov = truth.get("coverage")
        if cov:
            pct = cov["matched"] / max(1, cov["total"])
            cov_label = f"{cov['matched']:,} / {cov['total']:,} ({pct:.0%})"
        else:
            cov_label = "<not computed yet>"
        derived = truth.get("derived_csv")
        derived_label = (
            _short(derived) if derived and Path(derived).exists()
            else "<not derived yet>"
        )

        rows = [
            f"GA fitting source  {_short(truth['ga_csv'])}",
            f"Split              train {truth['train_frac']:.0%} / val {truth['val_frac']:.0%} / test {test_pct:.0%}",
            f"Split seed         {truth['split_seed']}",
            f"Coverage           {cov_label}",
            f"Truth file         {derived_label}",
            "[derive] Re-compute truth from noisy SMILES now",
            "[✓] Done",
        ]
        idx = curses_select(
            rows,
            title="Embedder similarity oracle (GA-derived)",
            subtitle=(
                "group additivity defines what 'similar' means in the embedding\n"
                "space. GA coefficients are fitted ONCE from the bundled reference\n"
                "(Morgan radius-1 fragments → h298, cached in ga_cache/), then\n"
                "applied to every SMILES in your noisy dataset. matched mols are\n"
                "used as the EMBEDDER's training target — they are NOT a ground\n"
                "truth for your noisy y values. unmatched SMILES are dropped\n"
                "(can't train without a target value for them)."
            ),
        )
        if idx < 0 or idx == len(rows) - 1:
            return
        if idx == 0:
            picked = _pick_csv("Select GA fitting source CSV", truth["ga_csv"])
            if picked:
                truth["ga_csv"] = picked
                truth["derived_csv"] = None
                truth["coverage"] = None
                # invalidate fitted-coefficient cache so the new source is used
                cache = WORKSPACE / "ga_cache" / "ga_coefficients.pkl"
                if cache.exists():
                    cache.unlink()
        elif idx == 1:
            tf = _pick_fraction(truth["train_frac"], label="Train fraction")
            vf = _pick_fraction(truth["val_frac"], label="Val fraction")
            if tf + vf >= 1.0:
                clear(); banner()
                error("train + val must be < 1 (rest goes to test)")
                pause()
                continue
            truth["train_frac"] = tf
            truth["val_frac"]   = vf
        elif idx == 2:
            clear(); banner()
            section("Split seed")
            truth["split_seed"] = IntPrompt.ask("  Seed", default=truth["split_seed"])
        elif idx == 5:
            clear(); banner()
            _derive_truth(state, verbose=True)
            pause()
        save_state(state)


# ── menu: configure noisy dataset ─────────────────────────────────────────

def menu_configure_noisy(state: dict) -> None:
    """Sub-menu for the noisy dataset — external CSV or derived from clean source."""
    nz = state["noisy"]
    while True:
        mode_label = (
            "external — use a noisy CSV directly"
            if nz["mode"] == "external"
            else "derived  — apply noise to a clean source"
        )
        if nz["mode"] == "derived":
            layers_summary = (
                f"{len(nz['noise_layers'])} layer(s)"
                if nz["noise_layers"] else "(none)"
            )
        else:
            layers_summary = "(only used in derived mode)"
        rendered = nz.get("rendered_csv")
        rendered_ok = rendered and Path(rendered).exists()
        rendered_label = _short(rendered) if rendered_ok else "<not rendered>"

        rows = [
            f"Mode          {mode_label}",
            f"Source CSV    {_short(nz['source_csv'])}",
            f"SMILES col    {nz['smiles_col']}",
            f"Target col    {nz['target_col']}",
            f"Subset        {nz['subset_fraction']:.0%}  (seed {nz['subset_seed']})",
            f"Noise layers  {layers_summary}",
            f"Rendered      {rendered_label}",
            "[render] Render noisy CSV now",
            "[✓] Done",
        ]
        idx = curses_select(
            rows,
            title="Noisy dataset",
            subtitle=(
                "the dataset we WANT to clean. two ways to make one:\n"
                "  external — point at an existing noisy CSV (e.g. all-small-\n"
                "             molecules, real measurements). subset is optional.\n"
                "  derived  — pick a clean source (e.g. truth) and stack custom\n"
                "             noise layers on top of its target column.\n"
                "the rendered CSV is what kNN denoising will read."
            ),
        )
        if idx < 0 or idx == len(rows) - 1:
            return

        invalidate = True
        if idx == 0:
            mode_idx = curses_select(
                [
                    "external — use a noisy CSV as-is (with optional subset)",
                    "derived  — start from a clean CSV and add custom noise",
                ],
                title="Noisy dataset mode",
            )
            if mode_idx == 0:
                nz["mode"] = "external"
            elif mode_idx == 1:
                nz["mode"] = "derived"
            else:
                invalidate = False
        elif idx == 1:
            picked = _pick_csv("Select source CSV", nz["source_csv"])
            if picked:
                nz["source_csv"] = picked
        elif idx == 2:
            new = _pick_column(nz["source_csv"], "SMILES column", nz["smiles_col"])
            if new:
                nz["smiles_col"] = new
        elif idx == 3:
            new = _pick_column(nz["source_csv"], "Target column", nz["target_col"])
            if new:
                nz["target_col"] = new
        elif idx == 4:
            nz["subset_fraction"] = max(
                min(_pick_fraction(nz["subset_fraction"], label="Subset fraction"), 1.0),
                1e-6,
            )
            clear(); banner()
            section("Subset seed")
            nz["subset_seed"] = IntPrompt.ask("  Seed", default=nz["subset_seed"])
        elif idx == 5:
            if nz["mode"] != "derived":
                clear(); banner()
                warn("noise layers only apply in derived mode — switch mode first")
                pause()
                invalidate = False
            else:
                nz["noise_layers"] = _configure_layers(nz["noise_layers"])
                # Fresh seed when layers actually changed-ish — keep existing.
        elif idx == 6:
            invalidate = False  # rendered row is read-only
        elif idx == 7:
            invalidate = False
            clear(); banner()
            _render_noisy(state, verbose=True)
            pause()

        if invalidate:
            nz["rendered_csv"] = None
            # Truth is keyed off the noisy SMILES; any change invalidates it.
            state["truth"]["derived_csv"] = None
            state["truth"]["coverage"] = None
        save_state(state)


# ── menu: train embedder ──────────────────────────────────────────────────

def run_train_embedder(state: dict) -> None:
    section("Train embedder")
    console.print(
        "  [muted]Trains a Chemprop encoder on the ground-truth CSV. The encoder learns[/muted]\n"
        "  [muted]to map each SMILES to a fixed-dim vector whose neighbours have similar[/muted]\n"
        "  [muted]target values. After training, only the encoder is kept (the prediction[/muted]\n"
        "  [muted]head is dropped). This embedding space is what kNN denoising will use.[/muted]\n"
    )
    truth = state["truth"]
    trn = state["train"]
    hpo = state["hpo"]

    nz = state["noisy"]
    if not nz.get("source_csv") or not Path(nz["source_csv"]).exists():
        error("noisy dataset not configured — see 'Configure noisy dataset'")
        return

    try:
        train_path, val_path = _ensure_truth_split(state)
    except Exception as e:
        error(f"truth split failed: {e}")
        return
    info(f"target column: {truth['target_col']!r}  (this defines what 'similar' means)")
    info(f"split: {train_path.name} / {val_path.name}")

    tuned = hpo["best_params"] if hpo["use_tuned"] and hpo["best_params"] else None
    if tuned:
        info("Using HPO-tuned hyperparameters")
    info(f"epochs={trn['epochs']}  batch_size={trn['batch_size']}  seed={trn['seed']}")

    try:
        import embedder as embedder_mod
    except Exception as e:
        error(f"failed to import embedder: {e}")
        return

    model_dir = Path(trn["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = embedder_mod.train_embedder(
            train_csv=str(train_path),
            val_csv=str(val_path),
            smiles_col=truth["smiles_col"],
            target_col=truth["target_col"],
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

    trn["trained"] = True
    save_state(state)


def _resolve_embedder_hparams(state: dict) -> dict | None:
    """Find the hparams the trained embedder was built with, if any."""
    cfg_path = Path(state["train"]["model_dir"]) / "embedder_hparams.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ── menu: run k-NN denoising ──────────────────────────────────────────────

def run_denoise(state: dict) -> None:
    section("Run k-NN denoising")
    console.print(
        "  [muted]For each round 1..N: every molecule's y is replaced by the inverse-[/muted]\n"
        "  [muted]distance-weighted mean of its k nearest neighbours' y values in the[/muted]\n"
        "  [muted]embedding space. Larger k smooths more; more rounds tighten[/muted]\n"
        "  [muted]convergence. Per-round CSVs are saved under last_run/rounds/.[/muted]\n"
    )
    trn = state["train"]
    nz = state["noisy"]
    knn = state["knn"]

    if not trn.get("trained"):
        error("Embedder is not trained. Train it first.")
        return

    weights_path = Path(trn["model_dir"]) / "embedder.pt"
    if not weights_path.exists():
        error(f"embedder weights not found at {weights_path}")
        return

    csv = nz.get("rendered_csv")
    if not csv or not Path(csv).exists():
        info("rendering noisy CSV from current config…")
        csv = _render_noisy(state, verbose=True)
        if not csv:
            return

    smiles_col = nz["smiles_col"]
    target_col = nz["target_col"]

    try:
        import embedder as embedder_mod
        import denoise as denoise_mod
    except Exception as e:
        error(f"failed to import modules: {e}")
        return

    df = pd.read_csv(csv)
    if smiles_col not in df.columns:
        error(f"column {smiles_col!r} not in {csv}")
        return
    if target_col not in df.columns:
        error(f"column {target_col!r} not in {csv}")
        return

    info(f"Loaded {len(df)} rows from {Path(csv).name}")

    smiles = df[smiles_col].astype(str).tolist()
    y_initial = df[target_col].astype(float).values

    info("Loading embedder …")
    model = embedder_mod.load_model(
        weights_path, hparams=_resolve_embedder_hparams(state),
    )

    info("Embedding molecules …")
    z, valid = embedder_mod.extract_embeddings(
        model, smiles, batch_size=trn["batch_size"],
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
    # Persist the embedding matrix so the visualisation step can cluster on it
    # without having to re-load and re-embed.
    np.save(last_run_dir / "z.npy", z.astype(np.float32))
    success(f"Wrote {_short(str(out_path))}")

    state["last_run"]["denoised_csv"] = str(out_path)
    state["last_run"]["metrics"] = None
    save_state(state)


# ── menu: benchmark ───────────────────────────────────────────────────────

def _plot_scatter_noisy_denoised(y_in, y_out, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 7))
    plt.scatter(y_in, y_out, s=4, alpha=0.4, color="#1f77b4")
    lo = float(min(y_in.min(), y_out.min()))
    hi = float(max(y_in.max(), y_out.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x  (no change)")
    plt.xlabel("noisy input  y")
    plt.ylabel("denoised  y")
    plt.title("noisy → denoised  (each point = one molecule)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()


def _plot_correction_hist(delta, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.hist(delta, bins=60, edgecolor="black", alpha=0.75, color="#2ca02c")
    plt.axvline(0, color="red", linestyle="--", linewidth=1, label="no change")
    plt.xlabel("correction  (denoised − noisy)")
    plt.ylabel("count")
    plt.title("distribution of corrections applied by k-NN smoothing")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()


def _plot_round_trails(
    df_final: pd.DataFrame,
    rounds_dir: Path,
    z_path: Path,
    out_path: Path,
    n_indicators: int = 12,
    n_overlay: int = 400,
) -> tuple[int, int] | None:
    """Trace the y value of `n_indicators` representative molecules across kNN
    rounds. Indicators are picked as cluster centroids in the embedding space
    so they characterise different neighbourhoods. A faint sample of the rest
    is overlaid for context.

    Returns ``(n_indicators_actual, n_rounds)`` or ``None`` if insufficient data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not z_path.exists() or not rounds_dir.exists():
        return None
    round_files = sorted(
        rounds_dir.glob("round_*.csv"),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    if not round_files:
        return None

    smiles = df_final["smiles"].astype(str).tolist()
    n = len(smiles)
    smi_to_idx = {s: i for i, s in enumerate(smiles)}

    rounds = len(round_files)
    Y = np.full((n, rounds + 1), np.nan, dtype=float)
    Y[:, 0] = df_final["y_initial"].astype(float).to_numpy()
    for r, p in enumerate(round_files, start=1):
        rdf = pd.read_csv(p)
        for s, y in zip(rdf["smiles"], rdf["y"]):
            i = smi_to_idx.get(str(s))
            if i is not None:
                Y[i, r] = float(y)

    z = np.load(z_path)
    if z.shape[0] != n:
        return None

    n_indicators = max(1, min(n_indicators, n))
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return None
    km = KMeans(n_clusters=n_indicators, random_state=42, n_init=10)
    labels = km.fit_predict(z)
    indicators: list[int] = []
    for c in range(n_indicators):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(z[members] - km.cluster_centers_[c], axis=1)
        indicators.append(int(members[d.argmin()]))

    fig, ax = plt.subplots(figsize=(11, 6))
    rng = np.random.default_rng(42)
    pool = rng.choice(n, size=min(n_overlay, n), replace=False)
    indset = set(indicators)
    for i in pool:
        if i in indset:
            continue
        ax.plot(range(rounds + 1), Y[i], color="lightgray",
                linewidth=0.5, alpha=0.35, zorder=1)

    cmap = plt.get_cmap("tab20")
    for k, i in enumerate(indicators):
        c = cmap(k % 20)
        smi = smiles[i]
        label = smi if len(smi) <= 28 else (smi[:27] + "…")
        ax.plot(range(rounds + 1), Y[i], color=c, linewidth=2.0, zorder=3, label=label)
        ax.scatter([0],      [Y[i, 0]],      color=c, s=42, zorder=4,
                   edgecolor="white", linewidth=1.0)
        ax.scatter([rounds], [Y[i, rounds]], color=c, s=42, zorder=4,
                   edgecolor="white", linewidth=1.0, marker="s")

    ax.set_xlabel("k-NN round  (round 0 = noisy input)")
    ax.set_ylabel("y value")
    ax.set_title(
        f"value evolution across {rounds} rounds — "
        f"{len(indicators)} cluster-representative molecules "
        f"(○ start, □ end), {min(n_overlay, n)} others overlaid"
    )
    ax.set_xticks(range(rounds + 1))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, title="indicators (SMILES)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return len(indicators), rounds


def _plot_against_clean(y_clean, y_noisy, y_denoised, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.scatter(y_clean, y_noisy, s=4, alpha=0.4, color="#d62728", label="noisy")
    plt.scatter(y_clean, y_denoised, s=4, alpha=0.4, color="#1f77b4", label="denoised")
    lo = float(min(y_clean.min(), y_noisy.min(), y_denoised.min()))
    hi = float(max(y_clean.max(), y_noisy.max(), y_denoised.max()))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="ideal (y = clean)")
    plt.xlabel("clean source  y")
    plt.ylabel("predicted  y")
    plt.title("recovery vs. clean source  (derived mode only)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()


def run_compare(state: dict) -> None:
    """Visualise the effect of denoising. We do NOT have ground truth for
    external noisy data — group additivity drives the embedder's similarity
    space, but it isn't a reference to score the denoised values against.
    The honest comparison is just noisy → denoised."""
    section("Visualise denoising")
    console.print(
        "  [muted]Group additivity is used to define the EMBEDDING SPACE — molecules[/muted]\n"
        "  [muted]with similar GA values land near each other. It is NOT a ground[/muted]\n"
        "  [muted]truth for your noisy y values; PM7 (or whatever your noisy data is)[/muted]\n"
        "  [muted]and GA are different physical estimates and disagree systematically.[/muted]\n"
        "  [muted]The honest visualisation is the scatter of noisy → denoised plus[/muted]\n"
        "  [muted]summary stats. If your noisy CSV was DERIVED (we kept the clean[/muted]\n"
        "  [muted]source), we can also score recovery against that clean reference.[/muted]\n"
    )
    last = state["last_run"]
    nz = state["noisy"]

    if not last.get("denoised_csv") or not Path(last["denoised_csv"]).exists():
        error("No denoised output yet. Run k-NN denoising first.")
        return

    df = pd.read_csv(last["denoised_csv"])
    if not {"smiles", "y_initial", "y_denoised"}.issubset(df.columns):
        error(f"denoised CSV missing expected columns: {list(df.columns)}")
        return
    y_in  = df["y_initial"].astype(float).to_numpy()
    y_out = df["y_denoised"].astype(float).to_numpy()
    delta = y_out - y_in

    var_in, var_out = float(np.var(y_in)), float(np.var(y_out))
    var_ratio = var_out / max(1e-12, var_in)
    rho = float(np.corrcoef(y_in, y_out)[0, 1])

    tbl = Table(box=box.ROUNDED, border_style="cyan", expand=False,
                title="[bold cyan]noisy → denoised  (no ground truth)[/bold cyan]")
    tbl.add_column("metric", style="accent")
    tbl.add_column("value", style="hi")
    tbl.add_row("molecules denoised",            f"{len(y_in):,}")
    tbl.add_row("std (input)",                   f"{float(np.std(y_in)):.4f}")
    tbl.add_row("std (denoised)",                f"{float(np.std(y_out)):.4f}")
    tbl.add_row("variance ratio (out / in)",     f"{var_ratio:.3f}× "
                f"({'spread shrunk' if var_ratio < 1 else 'spread grew'})")
    tbl.add_row("mean correction Δy",            f"{float(np.mean(delta)):+.4f}")
    tbl.add_row("mean |Δy|",                     f"{float(np.mean(np.abs(delta))):.4f}")
    tbl.add_row("max |Δy|",                      f"{float(np.max(np.abs(delta))):.4f}")
    tbl.add_row("ρ(noisy, denoised)",            f"{rho:.4f}")
    console.print(tbl)

    last_run_dir = WORKSPACE / "last_run"
    scatter_path = last_run_dir / "scatter_noisy_vs_denoised.png"
    hist_path    = last_run_dir / "corrections_hist.png"
    trails_path  = last_run_dir / "trails.png"
    _plot_scatter_noisy_denoised(y_in, y_out, scatter_path)
    _plot_correction_hist(delta, hist_path)
    success(f"scatter (noisy → denoised) → {_short(str(scatter_path))}")
    success(f"correction histogram       → {_short(str(hist_path))}")

    trail_info = _plot_round_trails(
        df_final=df,
        rounds_dir=last_run_dir / "rounds",
        z_path=last_run_dir / "z.npy",
        out_path=trails_path,
        n_indicators=12,
        n_overlay=400,
    )
    if trail_info is None:
        warn("trails plot skipped (need rounds/ + z.npy from a fresh denoise run)")
    else:
        n_ind, n_rounds = trail_info
        success(f"value-trail plot ({n_ind} indicators × {n_rounds} rounds) → "
                f"{_short(str(trails_path))}")

    # If derived mode and we still have the clean source, score recovery.
    if (
        nz.get("mode") == "derived"
        and nz.get("source_csv")
        and Path(nz["source_csv"]).exists()
    ):
        try:
            src_df = pd.read_csv(nz["source_csv"])
        except Exception as e:
            warn(f"could not read clean source for recovery comparison: {e}")
            src_df = None
        if src_df is not None:
            smi_col, tgt_col = nz["smiles_col"], nz["target_col"]
            if smi_col in src_df.columns and tgt_col in src_df.columns:
                ref = (
                    src_df[[smi_col, tgt_col]]
                    .rename(columns={smi_col: "smiles", tgt_col: "y_clean"})
                )
                merged = df.merge(ref, on="smiles", how="inner").dropna(
                    subset=["y_initial", "y_denoised", "y_clean"]
                )
                n = len(merged)
                if n > 0:
                    section(f"Recovery vs. clean source  ({n:,} matched)")
                    console.print(
                        "  [muted]This IS a meaningful benchmark: derived mode means we artificially[/muted]\n"
                        "  [muted]added noise to a clean source, so we know exactly what each[/muted]\n"
                        "  [muted]molecule's clean y was. Lower error / higher R² = denoising recovered[/muted]\n"
                        "  [muted]the underlying signal.[/muted]\n"
                    )
                    try:
                        import denoise as denoise_mod
                    except Exception as e:
                        warn(f"could not import denoise for metrics: {e}")
                    else:
                        m_n = denoise_mod.benchmark_against_truth(
                            merged["y_initial"].values, merged["y_clean"].values,
                        )
                        m_d = denoise_mod.benchmark_against_truth(
                            merged["y_denoised"].values, merged["y_clean"].values,
                        )
                        rec_tbl = Table(
                            box=box.ROUNDED, border_style="green", expand=False,
                            title="[bold green]recovery vs. clean source[/bold green]",
                        )
                        rec_tbl.add_column("metric",      style="accent")
                        rec_tbl.add_column("noisy",       style="warn")
                        rec_tbl.add_column("denoised",    style="success")
                        rec_tbl.add_column("Δ",           style="muted")
                        for key in ("mse", "mae", "rmse", "r2"):
                            before = m_n[key]; after = m_d[key]
                            rec_tbl.add_row(
                                key.upper(),
                                f"{before:.4f}", f"{after:.4f}",
                                f"{after - before:+.4f}",
                            )
                        console.print(rec_tbl)

                        truth_scatter = last_run_dir / "scatter_vs_clean.png"
                        _plot_against_clean(
                            merged["y_clean"].to_numpy(),
                            merged["y_initial"].to_numpy(),
                            merged["y_denoised"].to_numpy(),
                            truth_scatter,
                        )
                        success(f"recovery scatter → {_short(str(truth_scatter))}")

                        denoise_mod.write_metrics(
                            {"initial": m_n, "denoised": m_d, "n": int(n)},
                            last_run_dir / "metrics.json",
                        )
                        state["last_run"]["metrics"] = m_d
                else:
                    warn("derived mode but no SMILES overlap with the clean source")

    state["last_run"]["scatter_png"] = str(scatter_path)
    save_state(state)


# ── menu: HPO ─────────────────────────────────────────────────────────────

def menu_hpo(state: dict) -> None:
    while True:
        hpo = state["hpo"]
        truth = state["truth"]

        opts = [
            "Run HPO trials",
            f"Toggle 'use tuned params' [{'ON' if hpo['use_tuned'] else 'OFF'}]",
            "Edit n_trials / epochs",
            "View best params",
            "Clear best params",
            "Back",
        ]
        subtitle = (
            "Optuna search over embedder hyperparameters using the truth split.\n"
            "trains many small embedders and picks the one with the best val MSE.\n"
            f"trials={hpo['n_trials']}  epochs/trial={hpo['hpo_epochs']}  "
            f"best={'set' if hpo['best_params'] else '—'}"
        )
        idx = curses_select(opts, title="HPO menu", subtitle=subtitle)
        if idx < 0 or idx == 5:
            return

        if idx == 0:
            if not truth["csv"] or not Path(truth["csv"]).exists():
                error("ground-truth CSV not configured")
                pause()
                continue
            try:
                train_path, val_path = _ensure_truth_split(state)
            except Exception as e:
                error(f"split failed: {e}")
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
                    train_csv=str(train_path),
                    val_csv=str(val_path),
                    smiles_col=truth["smiles_col"],
                    target_col=truth["target_col"],
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


# ── pipeline explanation ──────────────────────────────────────────────────

def print_pipeline_help() -> None:
    """Walk-through of how every piece fits together, in rich format."""
    section("How the pipeline works")

    diagram = Text.from_markup(
        "                  [warn]Noisy CSV[/warn]   [muted](your only input)[/muted]\n"
        "                      │\n"
        "                      ▼\n"
        "         [accent]┌────────────────────────┐[/accent]\n"
        "         [accent]│ 0. AUTO-COMPUTE truth  │[/accent]   GA coefficients fitted\n"
        "         [accent]│    via group additivity│[/accent]   from reference, applied\n"
        "         [accent]└────────────┬───────────┘[/accent]   to each SMILES\n"
        "                      │\n"
        "                      ▼\n"
        "         [accent]┌────────────────────────┐[/accent]\n"
        "         [accent]│ 1. TRAIN encoder       │[/accent]   chemprop, on the\n"
        "         [accent]│    on truth split      │[/accent]   matched (SMILES, h298)\n"
        "         [accent]└────────────┬───────────┘[/accent]\n"
        "                      │\n"
        "                      ▼\n"
        "         [accent]┌────────────────────────┐[/accent]\n"
        "         [accent]│ 2. EMBED noisy mols    │[/accent]   deterministic vectors\n"
        "         [accent]│    via trained encoder │[/accent]\n"
        "         [accent]└────────────┬───────────┘[/accent]\n"
        "                      │\n"
        "                      ▼\n"
        "         [accent]┌────────────────────────┐[/accent]\n"
        "         [accent]│ 3. k-NN, N passes      │[/accent]   inv-distance weighted\n"
        "         [accent]│    over embeddings     │[/accent]   smoothing of y\n"
        "         [accent]└────────────┬───────────┘[/accent]\n"
        "                      │\n"
        "                      ▼\n"
        "         [success]┌────────────────────────┐[/success]\n"
        "         [success]│ 4. VISUALISE noisy →   │[/success]   scatter + stats;\n"
        "         [success]│    denoised            │[/success]   recovery vs. clean\n"
        "         [success]└────────────────────────┘[/success]   if derived mode\n"
    )
    console.print(Panel(diagram, border_style="cyan", box=box.ROUNDED, padding=(0, 2)))

    explanation = Text.from_markup(
        "\n[hi]One input, everything else automatic[/hi]\n"
        "  You only configure the [warn]noisy dataset[/warn] — the system derives the\n"
        "  ground truth from a bundled group-additivity reference table\n"
        "  (~7.9M molecules, deterministic h298). For every SMILES in your\n"
        "  noisy CSV the GA value is looked up; matched rows become the\n"
        "  truth. Unmatched SMILES are dropped with a coverage report.\n"
        "\n[hi]Datasets[/hi]\n"
        "  • [warn]Noisy[/warn] — what we want to clean. Two ways to make one:\n"
        "      external — point at an existing noisy CSV (real measurements,\n"
        "                 the all-small-molecules dataset, etc.). Optional\n"
        "                 deterministic subset (fraction + seed).\n"
        "      derived  — start from a clean source and apply stacked custom\n"
        "                 noise layers (normal/uniform/outliers/nitrogen/…).\n"
        "                 Each layer has its own deterministic seed.\n"
        "  • [brand]Ground truth[/brand] — never user-supplied. GA-derived from your noisy\n"
        "    SMILES on demand, and cached in cli_workspace/truth/.\n"
        "\n[hi]Step 0 — Auto-compute truth (group additivity)[/hi]\n"
        "  Each molecule is decomposed into Morgan radius-1 atom environments\n"
        "  (each heavy atom + its directly-bonded neighbours). h298 is the sum\n"
        "  ``Σ count_i · coef_i`` over those environments. Coefficients are\n"
        "  fitted ONCE via sparse least-squares from the bundled reference\n"
        "  (~20s first run, cached afterwards). After fitting, predictions work\n"
        "  on any SMILES whose fragments are all in the fitted vocabulary —\n"
        "  no exact-SMILES match needed. Molecules with elements outside the\n"
        "  reference set (gdb11 → C/H/N/O/F) are unpredictable and dropped.\n"
        "\n[hi]Step 1 — Train embedder[/hi]\n"
        "  A Chemprop message-passing encoder is trained on the matched truth\n"
        "  split (auto train/val). After training, the FFN head is dropped\n"
        "  and only the encoder is kept. Because h298 is deterministic, mols\n"
        "  with similar h298 land near each other in the embedding space.\n"
        "\n[hi]Step 2 — Embed noisy molecules[/hi]\n"
        "  The encoder is applied to every SMILES in the rendered noisy CSV,\n"
        "  giving a fixed-dim vector per molecule. y-noise doesn't affect the\n"
        "  embedding — only the SMILES does.\n"
        "\n[hi]Step 3 — k-NN denoising, N passes[/hi]\n"
        "  For each round 1..N, every molecule's y is replaced by the inverse-\n"
        "  distance-weighted mean of its k nearest neighbours in the embedding\n"
        "  space. Larger k smooths more; more rounds tighten convergence.\n"
        "  Per-round CSVs are saved under last_run/rounds/.\n"
        "\n[hi]Step 4 — Visualise[/hi]\n"
        "  GA is the EMBEDDER's similarity oracle, not a ground truth for your\n"
        "  noisy y values. PM7 (or whatever your noisy data is) and GA are\n"
        "  different physical estimates that disagree systematically — scoring\n"
        "  denoised PM7 against GA punishes that disagreement, not the denoising.\n"
        "  So Step 4 shows: (a) scatter of noisy → denoised per molecule,\n"
        "  (b) variance reduction and correction-magnitude stats, (c) histogram\n"
        "  of |Δy|. If your noisy CSV was DERIVED (we kept the clean source),\n"
        "  we additionally score recovery against that clean source — there\n"
        "  the underlying signal is known, so MSE/MAE/R² are meaningful.\n"
        "  Plots land in cli_workspace/last_run/*.png.\n"
        "\n[hi]Tips[/hi]\n"
        "  • The variance ratio (out / in) is the cleanest single-number stat:\n"
        "    < 1 means the denoiser shrunk the spread of y values.\n"
        "  • Low GA coverage (in step 0) just means fewer molecules are usable\n"
        "    for embedder training — the rest of the pipeline still runs on\n"
        "    whatever matched, and kNN denoising operates on ALL noisy rows\n"
        "    even those without GA values.\n"
        "  • External + subset 100% + 0 layers is a no-op render; the renderer\n"
        "    points straight at the source CSV.\n"
    )
    console.print(explanation)


# ── main loop ─────────────────────────────────────────────────────────────

MAIN_OPTS = [
    "Overview",                                    # 0
    "How it works (pipeline explained)",           # 1
    "Configure embedder similarity (GA-derived)",  # 2
    "Configure noisy dataset",                     # 3
    "Train embedder",                              # 4
    "Run k-NN denoising",                          # 5
    "Visualise denoising (scatter + stats)",       # 6
    "Settings (epochs, batch, seed, k, rounds)",   # 7
    "Hyperparameter tuning (advanced)",            # 8
    "Reset workspace",                             # 9
    "Quit",                                        # 10
]


def main():
    clear()
    state = load_state()

    while True:
        idx = curses_select(
            MAIN_OPTS,
            title="Main menu",
            subtitle=(
                "MolAugment — denoise a noisy SMILES → y CSV via a deterministic\n"
                "embedding learned from ground truth. New here? Pick option 1."
            ),
        )

        if idx < 0 or idx == 10:
            console.print("\n  [muted]Goodbye![/muted]\n")
            break

        if idx == 0:
            print_status(state)
            pause()
        elif idx == 1:
            print_pipeline_help()
            pause()
        elif idx == 2:
            menu_configure_truth(state)
        elif idx == 3:
            menu_configure_noisy(state)
        elif idx == 4:
            run_train_embedder(state)
            pause()
        elif idx == 5:
            run_denoise(state)
            pause()
        elif idx == 6:
            run_compare(state)
            pause()
        elif idx == 7:
            menu_settings(state)
        elif idx == 8:
            menu_hpo(state)
        elif idx == 9:
            state = reset_workspace(state)
            save_state(state)
            pause()


if __name__ == "__main__":
    main()
