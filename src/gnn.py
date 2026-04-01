"""
GNN training wrapper — Chemprop v2 regression.

Exposes:
  train_chemprop(train_csv, save_dir, target_col, epochs,
                 batch_size, seed, val_csv, smiles_col)
  export_embeddings(model_dir, data_csv, out_path)   # placeholder
"""

from pathlib import Path
import subprocess
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────

def check_chemprop_installed():
    """Raise RuntimeError if the chemprop CLI is not on PATH."""
    try:
        subprocess.run(
            ["chemprop", "--help"], check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        raise RuntimeError(
            "Chemprop CLI not found. Activate your env:\n"
            "  conda activate chemprop\n"
            "  pip install chemprop"
        )


def load_and_check_csv(
    csv_path: str,
    smiles_col: str = "smiles",
    target_col: str = "h298",
) -> pd.DataFrame:
    """Load CSV and verify that required columns are present."""
    df = pd.read_csv(csv_path)
    for col in (smiles_col, target_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {csv_path}.\n"
                f"  Available: {list(df.columns)}"
            )
    return df


# ── main training function ─────────────────────────────────────────────────

def train_chemprop(
    train_csv:  str,
    save_dir:   str,
    target_col: str = "h298",
    epochs:     int = 30,
    batch_size: int = 128,
    seed:       int = 42,
    val_csv:    str | None = None,
    test_csv:   str | None = None,
    smiles_col: str = "smiles",
    descriptor_columns: list[str] | None = None,
) -> str:
    """
    Train a Chemprop v2 regression model.

    Parameters
    ----------
    train_csv  : Path to the training CSV (must have smiles_col + target_col).
    save_dir   : Directory where the model checkpoint will be written.
    target_col : Name of the regression target column.
    epochs     : Number of training epochs.
    batch_size : Mini-batch size.
    seed       : Random seed for reproducibility.
    val_csv    : Optional validation CSV.
    test_csv   : Optional test CSV.
    smiles_col : Name of the SMILES column (used for pre-check only).
    descriptor_columns : Optional list of column names to use as extra features.

    Returns
    -------
    save_dir (str) — where the model was saved.
    """
    check_chemprop_installed()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Pre-flight data checks
    load_and_check_csv(train_csv, smiles_col=smiles_col, target_col=target_col)
    if val_csv:
        load_and_check_csv(val_csv, smiles_col=smiles_col, target_col=target_col)
    if test_csv:
        load_and_check_csv(test_csv, smiles_col=smiles_col, target_col=target_col)

    # Combine train and val (and test) if splits are provided
    combined_csv = train_csv
    
    if val_csv or test_csv:
        dfs = []
        df_train = pd.read_csv(train_csv)
        df_train["split_col"] = "train"
        dfs.append(df_train)
        
        if val_csv:
            df_val = pd.read_csv(val_csv)
            df_val["split_col"] = "val"
            dfs.append(df_val)
            
        if test_csv:
            df_test = pd.read_csv(test_csv)
            df_test["split_col"] = "test"
            dfs.append(df_test)
        
        df_combined = pd.concat(dfs, ignore_index=True)
        combined_csv = str(Path(save_dir) / "combined_data.csv")
        df_combined.to_csv(combined_csv, index=False)

    cmd = [
        "chemprop", "train",
        "--data-path",      combined_csv,
        "--target-columns", target_col,
        "--task-type",      "regression",
        "--save-dir",       save_dir,
        "--epochs",         str(epochs),
        "--batch-size",     str(batch_size),
        "--data-seed",      str(seed),
        "--pytorch-seed",   str(seed),
        "--num-workers",    "0",  # Disable dataloader warnings
    ]

    if descriptor_columns:
        cmd.extend(["--descriptors-columns"] + descriptor_columns)
        # Avoid scaling descriptors if they are targets (keep raw physics/units consistent)
        cmd.extend(["--no-descriptor-scaling"])
    
    import re
    import os
    import math
    import time
    import glob
    import psutil

    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich import print as rprint

    # Track how many batches per epoch to calculate hyper-accurate fractional completion
    num_train_samples = len(pd.read_csv(train_csv))
    batches_per_epoch = math.ceil(num_train_samples / batch_size)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    mpl_dir = Path(save_dir).parent.parent / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)

    log_path = Path(save_dir) / "chemprop_run.log"
    log_file = open(log_path, "w", encoding="utf-8")

    process = subprocess.Popen(
        cmd, 
        stdout=log_file, 
        stderr=subprocess.STDOUT, 
        close_fds=True,
        env=env
    )
    
    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(style="magenta", complete_style="cyan", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[yellow]{task.fields[metrics]}"),
        transient=False
    ) as progress:
        task = progress.add_task(f"Training {target_col} (Initializing...)", total=epochs, metrics="")
        
        while process.poll() is None:
            time.sleep(1)
            try:
                metrics_files = glob.glob(str(Path(save_dir) / "**" / "metrics.csv"), recursive=True)
                if not metrics_files:
                    continue
                
                latest_metrics = sorted(metrics_files)[-1] # highest version
                df_metrics = pd.read_csv(latest_metrics)
                if df_metrics.empty or "step" not in df_metrics.columns:
                    continue
                
                latest_step = df_metrics["step"].dropna().iloc[-1]
                current_epoch = df_metrics["epoch"].dropna().iloc[-1] if "epoch" in df_metrics.columns else 0
                
                tr_str = ""
                if "train_loss_step" in df_metrics.columns:
                    series = df_metrics["train_loss_step"].dropna()
                    if not series.empty:
                        tr_str = f"Train Loss: {series.iloc[-1]:.4f}"
                
                val_str = ""
                val_cols = [c for c in df_metrics.columns if "val" in c and "loss" in c]
                if val_cols:
                    val_series = df_metrics[val_cols[0]].dropna()
                    if not val_series.empty:
                        val_str = f"Val Loss: {val_series.iloc[-1]:.4f}"
                
                # Fetch RAM info
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()
                ram_str = f"RAM: {mem.used/(1024**3):.1f}GB | Swap: {swap.used/(1024**3):.1f}GB"
                
                metric_str = " | ".join(filter(None, [ram_str, tr_str, val_str]))
                fractional_epoch = latest_step / batches_per_epoch
                
                progress.update(
                    task,
                    completed=fractional_epoch,
                    total=epochs,
                    description=f"Training {target_col} (Epoch {int(current_epoch)}/{epochs})",
                    metrics=metric_str
                )
            except Exception:
                pass
                
    log_file.close()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    # Scrape final test metrics from log if test set was requested
    if test_csv and log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as lf:
                lines = lf.readlines()
            
            test_lines = []
            in_test_block = False
            for line in lines:
                if "Testing DataLoader" in line or "┏━━━━" in line or "Testing" in line:
                    if not any(c in line for c in ['━━━', 'eta', 'it/s', 'Testing DataLoader']):
                        in_test_block = True
                if in_test_block:
                    if not any(c in line for c in ['━━━', 'eta', 'it/s']):
                        test_lines.append(line.strip())
            
            if test_lines:
                # Remove ANSI colour codes optionally
                clean_lines = [re.sub(r'\x1b\[.*?m', '', l) for l in test_lines]
                rprint(Panel("\n".join(clean_lines), title=f"Test Results: {target_col}", border_style="green"))
        except Exception:
            pass

    # Optionally clean up the temporary combined file
    if (val_csv or test_csv) and Path(combined_csv).exists():
        try:
            Path(combined_csv).unlink()
        except OSError:
            pass

    if test_lines:
        rprint(Panel("\n".join(test_lines), title=f"Test Results: {target_col}", border_style="green"))

    return save_dir


# ── placeholder: embedding export ─────────────────────────────────────────

def export_embeddings(model_dir: str, data_csv: str, out_path: str):
    """Extract molecule embeddings from a trained Chemprop model. (TODO)"""
    # TODO: implement once embedding extraction API is confirmed
    pass
