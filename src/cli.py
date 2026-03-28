#!/usr/bin/env python3
"""
MolAugment CLI — Interactive dataset augmentation toolkit.

A sleek terminal UI for scanning, inspecting, and augmenting molecular
property datasets (SMILES + target CSV files).

Run:
    python src/cli.py                     # auto-scan from project root
    python src/cli.py --root ./datasets   # scan a specific directory
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure sibling modules (augment_dataset.py) are importable
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Dependency bootstrap — install rich automatically if missing
# ---------------------------------------------------------------------------
try:
    import rich
except ImportError:
    print("Installing 'rich' for terminal UI …")
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    import rich  # noqa: F811 — re-import after install

import pandas as pd
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Theme & console
# ---------------------------------------------------------------------------
CUSTOM_THEME = Theme({
    "brand":    "bold cyan",
    "accent":   "bold magenta",
    "success":  "bold green",
    "warn":     "bold yellow",
    "err":      "bold red",
    "muted":    "dim white",
    "heading":  "bold white on rgb(30,30,60)",
})

console = Console(theme=CUSTOM_THEME)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BRAND = "[brand]⚗  MolAugment[/brand]"
VERSION = "1.0.0"


# ═══════════════════════════════════════════════════════════════════════════
#  UI helpers
# ═══════════════════════════════════════════════════════════════════════════

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    """Print the app banner."""
    art = Text.from_markup(
        "\n"
        "  [brand]⚗  MolAugment[/brand]  [muted]v" + VERSION + "[/muted]\n"
        "  [muted]Interactive molecular dataset augmentation toolkit[/muted]\n"
    )
    console.print(
        Panel(
            art,
            border_style="cyan",
            box=box.DOUBLE_EDGE,
            expand=False,
            padding=(0, 2),
        )
    )


def section(title: str):
    console.print()
    console.print(Rule(f"[accent] {title} [/accent]", style="magenta"))
    console.print()


def success(msg: str):
    console.print(f"  [success]✓[/success] {msg}")


def warn(msg: str):
    console.print(f"  [warn]⚠[/warn]  {msg}")


def error(msg: str):
    console.print(f"  [err]✗[/err] {msg}")


def humanize_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.0f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:,.1f} TB"


def pause():
    console.print()
    Prompt.ask("  [muted]Press Enter to continue[/muted]", default="")


# ═══════════════════════════════════════════════════════════════════════════
#  CSV scanning & inspection
# ═══════════════════════════════════════════════════════════════════════════

def scan_csvs(root: Path, max_depth: int = 6) -> list[Path]:
    """Recursively discover CSV files, skipping hidden dirs & venvs."""
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox"}
    found: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # prune
        depth = str(dirpath).count(os.sep) - str(root).count(os.sep)
        if depth >= max_depth:
            dirnames.clear()
            continue
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith('.')]
        for f in sorted(filenames):
            if f.lower().endswith(".csv"):
                found.append(Path(dirpath) / f)
    return found


def peek_csv(path: Path, n: int = 5) -> tuple[int, list[str], pd.DataFrame]:
    """Return (row_count, columns, head_df) for a CSV."""
    df = pd.read_csv(path, nrows=0)
    cols = list(df.columns)
    # count rows cheaply
    with open(path, "rb") as f:
        row_count = sum(1 for _ in f) - 1          # minus header
    head_df = pd.read_csv(path, nrows=n)
    return row_count, cols, head_df


def display_csv_table(csvs: list[Path], root: Path) -> Table:
    """Build a rich Table listing discovered CSVs."""
    table = Table(
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold white on rgb(30,30,60)",
        row_styles=["", "on rgb(20,20,35)"],
        show_lines=False,
        pad_edge=True,
        expand=True,
    )
    table.add_column("#", style="brand", justify="right", width=4)
    table.add_column("File", style="white", ratio=4)
    table.add_column("Size", style="muted", justify="right", width=10)
    table.add_column("Rows", style="muted", justify="right", width=10)
    table.add_column("Columns", style="cyan", ratio=3)

    for i, csv_path in enumerate(csvs, 1):
        rel = csv_path.relative_to(root)
        size = csv_path.stat().st_size
        try:
            row_count, cols, _ = peek_csv(csv_path)
            cols_str = ", ".join(cols[:5])
            if len(cols) > 5:
                cols_str += f" [muted](+{len(cols)-5})[/muted]"
        except Exception:
            row_count = -1
            cols_str = "[err]error reading[/err]"
        table.add_row(
            str(i),
            str(rel),
            humanize_bytes(size),
            f"{row_count:,}" if row_count >= 0 else "?",
            cols_str,
        )
    return table


def inspect_csv(csv_path: Path):
    """Show detailed info + head of a CSV."""
    section(f"Inspecting: {csv_path.name}")
    try:
        row_count, cols, head_df = peek_csv(csv_path, n=8)
    except Exception as e:
        error(f"Could not read file: {e}")
        return

    info_items = [
        f"[brand]Path:[/brand]    {csv_path}",
        f"[brand]Size:[/brand]    {humanize_bytes(csv_path.stat().st_size)}",
        f"[brand]Rows:[/brand]    {row_count:,}",
        f"[brand]Columns:[/brand] {', '.join(cols)}",
    ]
    console.print(Panel(
        "\n".join(info_items),
        title="[accent]File info[/accent]",
        border_style="magenta",
        expand=False,
        padding=(1, 3),
    ))

    # preview table
    preview = Table(
        box=box.SIMPLE_HEAD,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=False,
    )
    for col in head_df.columns:
        preview.add_column(col, overflow="fold")
    for _, row in head_df.iterrows():
        preview.add_row(*[str(v) for v in row])

    console.print(Padding(preview, (1, 2)))


# ═══════════════════════════════════════════════════════════════════════════
#  Column auto-detection
# ═══════════════════════════════════════════════════════════════════════════

_SMILES_HINTS = {"smiles", "smi", "smile", "canonical_smiles", "mol", "molecule", "input"}
_TARGET_HINTS = {"h298", "target", "value", "property", "enthalpy", "energy", "y"}


def guess_columns(cols: list[str]) -> tuple[str | None, str | None]:
    """Try to auto-detect the SMILES and target columns."""
    smiles_col = None
    target_col = None
    lower_cols = {c.lower(): c for c in cols}
    for hint in _SMILES_HINTS:
        if hint in lower_cols:
            smiles_col = lower_cols[hint]
            break
    for hint in _TARGET_HINTS:
        if hint in lower_cols:
            target_col = lower_cols[hint]
            break
    return smiles_col, target_col


# ═══════════════════════════════════════════════════════════════════════════
#  Augmentation engine  (wraps augment_dataset.py functions)
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except ImportError:
        error("RDKit is not installed in this Python environment.")
        warn("Install via:  [bold]conda install -c conda-forge rdkit[/bold]")
        warn("Or:           [bold]pip install rdkit[/bold]")
        return False


def run_augmentation(
    csv_path: Path,
    smiles_col: str,
    target_col: str,
    do_mirror: bool,
    do_tautomers: bool,
    max_tautomers: int,
    output_path: Path,
):
    """Run the augmentation pipeline with a live rich progress bar."""

    if not _ensure_rdkit():
        return

    # Late import so the CLI menu still works without rdkit
    from augment_dataset import mirror_molecule, enumerate_tautomers

    df = pd.read_csv(csv_path)
    total = len(df)

    rows: list[dict] = []
    mirror_count = 0
    tautomer_count = 0
    invalid_count = 0

    with Progress(
        SpinnerColumn("dots", style="cyan"),
        TextColumn("[brand]{task.description}[/brand]"),
        BarColumn(bar_width=40, style="magenta", complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Augmenting molecules …", total=total)

        for _, row in df.iterrows():
            smi = row[smiles_col]
            target = row[target_col]

            rows.append({smiles_col: smi, target_col: target, "augmentation": "original"})

            # mirror
            if do_mirror:
                try:
                    mirror_smi = mirror_molecule(smi)
                    if mirror_smi is not None:
                        rows.append({
                            smiles_col: mirror_smi,
                            target_col: target,
                            "augmentation": "mirror",
                        })
                        mirror_count += 1
                except Exception:
                    invalid_count += 1

            # tautomers
            if do_tautomers:
                try:
                    for taut_smi in enumerate_tautomers(smi, max_tautomers=max_tautomers):
                        rows.append({
                            smiles_col: taut_smi,
                            target_col: target,
                            "augmentation": "tautomer",
                        })
                        tautomer_count += 1
                except Exception:
                    invalid_count += 1

            progress.advance(task)

    aug_df = pd.DataFrame(rows)
    aug_df.to_csv(output_path, index=False)

    # ── result summary ──
    section("Results")

    ratio = len(aug_df) / total if total else 0

    stats = Table(box=box.ROUNDED, border_style="green", expand=False, show_header=False)
    stats.add_column("Metric", style="white", width=28)
    stats.add_column("Value", style="brand", justify="right", width=14)

    stats.add_row("Original rows", f"{total:,}")
    stats.add_row("+ Mirror (enantiomers)", f"{mirror_count:,}")
    stats.add_row("+ Tautomers", f"{tautomer_count:,}")
    if invalid_count:
        stats.add_row("[warn]Skipped (invalid SMILES)[/warn]", f"[warn]{invalid_count:,}[/warn]")
    stats.add_row("", "")
    stats.add_row("[success]Total rows[/success]", f"[success]{len(aug_df):,}[/success]")
    stats.add_row("Expansion factor", f"{ratio:.2f}×")
    stats.add_row("", "")
    stats.add_row("Saved to", str(output_path))

    console.print(Padding(stats, (0, 2)))
    console.print()
    success(f"Augmented dataset written to [bold]{output_path}[/bold]")


# ═══════════════════════════════════════════════════════════════════════════
#  Interactive menu system
# ═══════════════════════════════════════════════════════════════════════════

def menu_select_csv(csvs: list[Path], root: Path) -> Path | None:
    """Let the user pick a CSV — with search/filter support."""
    active = csvs  # the currently displayed subset

    while True:
        section("Discovered CSV files")

        if len(active) > 30:
            console.print(
                f"  [muted]{len(active)} files found — too many to list all.[/muted]\n"
                f"  [muted]Type a search term to filter, or 'all' to show everything.[/muted]"
            )
            console.print()
            query = Prompt.ask(
                "  [accent]Search / filter[/accent] [muted](text, 'all', or 'q' to quit)[/muted]",
                default="groupadditivity_0",
            )
            if query.lower() in ("q", "quit", "exit"):
                return None
            if query.lower() == "all":
                filtered = active
            else:
                filtered = [p for p in active if query.lower() in str(p.relative_to(root)).lower()]
            if not filtered:
                warn(f"No files matching '{query}'. Try again.")
                continue
            active = filtered
            continue  # re-display with the filtered list

        console.print(display_csv_table(active, root))
        console.print()

        choice = Prompt.ask(
            "  [accent]Select a file[/accent] [muted](number, 'f' to re-filter, or 'q' to quit)[/muted]",
            default="1",
        )
        if choice.lower() in ("q", "quit", "exit"):
            return None
        if choice.lower() in ("f", "filter", "search"):
            active = csvs  # reset to full list and re-filter
            continue
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(active):
                return active[idx]
        except ValueError:
            pass
        error("Invalid selection — try a number from the list.")


def menu_main_action() -> str:
    """Main action menu after CSV selection."""
    section("What would you like to do?")

    actions = Table(box=box.SIMPLE, show_header=False, expand=False, padding=(0, 2))
    actions.add_column("Key", style="brand", width=4)
    actions.add_column("Action", style="white")
    actions.add_column("Description", style="muted")

    actions.add_row("1", "🔬  Inspect dataset",       "Preview rows, columns, and stats")
    actions.add_row("2", "⚗️   Augment dataset",       "Run mirror + tautomer augmentation")
    actions.add_row("3", "📂  Pick a different file",  "Go back to file selection")
    actions.add_row("q", "🚪  Quit",                   "Exit the tool")

    console.print(Padding(actions, (0, 2)))
    console.print()

    choice = Prompt.ask(
        "  [accent]Choose[/accent] [muted](1-3 or q)[/muted]",
        choices=["1", "2", "3", "q"],
        default="2",
    )
    return choice


def menu_configure_augmentation(
    csv_path: Path, cols: list[str]
) -> dict | None:
    """Interactive parameter configuration for augmentation."""
    section("Configure augmentation")

    # ── auto-detect columns ──
    smiles_guess, target_guess = guess_columns(cols)

    if smiles_guess:
        success(f"Auto-detected SMILES column: [brand]{smiles_guess}[/brand]")
    if target_guess:
        success(f"Auto-detected target column: [brand]{target_guess}[/brand]")

    console.print()

    # SMILES column
    if smiles_guess:
        smiles_col = Prompt.ask(
            "  [accent]SMILES column[/accent]",
            default=smiles_guess,
        )
    else:
        console.print(f"  Available columns: [muted]{', '.join(cols)}[/muted]")
        smiles_col = Prompt.ask("  [accent]SMILES column[/accent]")
    if smiles_col not in cols:
        error(f"Column '{smiles_col}' not found. Aborting.")
        return None

    # Target column
    if target_guess:
        target_col = Prompt.ask(
            "  [accent]Target column[/accent]",
            default=target_guess,
        )
    else:
        remaining = [c for c in cols if c != smiles_col]
        console.print(f"  Remaining columns: [muted]{', '.join(remaining)}[/muted]")
        target_col = Prompt.ask("  [accent]Target column[/accent]")
    if target_col not in cols:
        error(f"Column '{target_col}' not found. Aborting.")
        return None

    console.print()

    # ── augmentation toggles ──
    section("Augmentation methods")

    toggle_table = Table(box=box.SIMPLE, show_header=False, expand=False, padding=(0, 1))
    toggle_table.add_column("", width=3)
    toggle_table.add_column("Method", style="white", width=24)
    toggle_table.add_column("Description", style="muted")
    toggle_table.add_row("1", "🪞  Mirror molecules", "Flip chiral centres → enantiomers (same H298)")
    toggle_table.add_row("2", "🧪  Tautomers",        "Keto↔enol, imine↔enamine, etc. (same H298)")
    console.print(Padding(toggle_table, (0, 2)))
    console.print()

    do_mirror = Confirm.ask(
        "  [accent]Enable mirror molecules?[/accent]",
        default=True,
    )
    do_tautomers = Confirm.ask(
        "  [accent]Enable tautomer enumeration?[/accent]",
        default=True,
    )

    max_tautomers = 5
    if do_tautomers:
        max_tautomers = IntPrompt.ask(
            "  [accent]Max tautomers per molecule[/accent]",
            default=5,
        )

    if not do_mirror and not do_tautomers:
        warn("No augmentation methods selected. Nothing to do.")
        return None

    # ── output path ──
    console.print()
    default_out = csv_path.parent / f"{csv_path.stem}_augmented{csv_path.suffix}"
    output_path = Prompt.ask(
        "  [accent]Output path[/accent]",
        default=str(default_out),
    )

    # ── confirmation ──
    console.print()
    confirm_table = Table(box=box.ROUNDED, border_style="cyan", expand=False, show_header=False)
    confirm_table.add_column("Parameter", style="accent", width=22)
    confirm_table.add_column("Value", style="white")

    confirm_table.add_row("Input file", str(csv_path))
    confirm_table.add_row("SMILES column", smiles_col)
    confirm_table.add_row("Target column", target_col)
    confirm_table.add_row("Mirror molecules", "✅  Yes" if do_mirror else "❌  No")
    confirm_table.add_row("Tautomers", f"✅  Yes (max {max_tautomers})" if do_tautomers else "❌  No")
    confirm_table.add_row("Output file", output_path)

    console.print(Padding(confirm_table, (0, 2)))
    console.print()

    if not Confirm.ask("  [accent]Proceed with augmentation?[/accent]", default=True):
        warn("Augmentation cancelled.")
        return None

    return {
        "csv_path": csv_path,
        "smiles_col": smiles_col,
        "target_col": target_col,
        "do_mirror": do_mirror,
        "do_tautomers": do_tautomers,
        "max_tautomers": max_tautomers,
        "output_path": Path(output_path),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MolAugment — interactive CLI")
    parser.add_argument(
        "--root", type=str, default=None,
        help="Root directory to scan for CSVs (default: project root).",
    )
    args = parser.parse_args()

    # Resolve project root
    if args.root:
        root = Path(args.root).resolve()
    else:
        root = Path(__file__).resolve().parent.parent   # assumes src/ lives under project root

    clear()
    banner()

    # ── scan ──
    with console.status("[brand]Scanning for CSV files …[/brand]", spinner="dots"):
        csvs = scan_csvs(root)

    if not csvs:
        error(f"No CSV files found under {root}")
        sys.exit(1)

    success(f"Found [brand]{len(csvs)}[/brand] CSV file(s) under [muted]{root}[/muted]")

    # ── main loop ──
    selected: Path | None = None

    while True:
        if selected is None:
            selected = menu_select_csv(csvs, root)
            if selected is None:
                console.print("\n  [muted]Goodbye 👋[/muted]\n")
                break

        action = menu_main_action()

        if action == "1":
            inspect_csv(selected)
            pause()

        elif action == "2":
            _, cols, _ = peek_csv(selected)
            config = menu_configure_augmentation(selected, cols)
            if config is not None:
                console.print()
                run_augmentation(**config)
                pause()

        elif action == "3":
            selected = None  # go back to file picker

        elif action == "q":
            console.print("\n  [muted]Goodbye 👋[/muted]\n")
            break


if __name__ == "__main__":
    main()
