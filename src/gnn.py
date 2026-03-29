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
    smiles_col: str = "smiles",
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
    val_csv    : Optional validation CSV.  If None, Chemprop uses its own split.
    smiles_col : Name of the SMILES column (used for pre-check only).

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

    # Combine train and val if val_csv is provided
    combined_csv = train_csv
    
    if val_csv:
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        
        df_train["split_col"] = "train"
        df_val["split_col"] = "val"
        
        df_combined = pd.concat([df_train, df_val], ignore_index=True)
        combined_csv = str(Path(save_dir) / "combined_train_val.csv")
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
    ]
    
    if val_csv:
        cmd += ["--splits-column", "split_col"]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)   # raises CalledProcessError on failure

    # Optionally clean up the temporary combined file
    if val_csv and Path(combined_csv).exists():
        try:
            Path(combined_csv).unlink()
        except OSError:
            pass

    return save_dir


# ── placeholder: embedding export ─────────────────────────────────────────

def export_embeddings(model_dir: str, data_csv: str, out_path: str):
    """Extract molecule embeddings from a trained Chemprop model. (TODO)"""
    # TODO: implement once embedding extraction API is confirmed
    pass
