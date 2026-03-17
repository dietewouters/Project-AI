# GNN
from pathlib import Path
import subprocess
import pandas as pd


# Helpers 


# check if chemprop is available
def check_chemprop_installed():
    try:
        subprocess.run(["chemprop", "--help"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        raise RuntimeError(
            "Chemprop CLI not found. Activate your env (conda activate chemprop) "
            "and run: pip install chemprop"
        )


def load_and_check_csv(csv_path : str, smiles_col : str="smiles", target_col : str="h298"):
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Missing column '{smiles_col}' in {csv_path} ")
    if target_col not in df.columns:
        raise ValueError(f"Missing column '{target_col}' in {csv_path} ")
    return df



# Main functions

# Train a chemprop (v2) model (regression baseline)
def train_chemprop(train_csv: str, save_dir: str, target_col: str = "h298",
                   epochs: int = 30, batch_size: int = 128, seed: int = 0):
    check_chemprop_installed()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # sanity check on the csv
    load_and_check_csv(train_csv, smiles_col="smiles", target_col=target_col)

    cmd = [
        "chemprop", "train",
        "--data-path", train_csv,
        "--target-columns", target_col,
        "--task-type", "regression",
        "--save-dir", save_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed)
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True) # check = True => if chemprop fails

    return save_dir


# Placeholder: later we will extract molecule embeddings from the trained model
def export_embeddings(model_dir: str, data_csv: str, out_path: str):
    # TODO
    pass
