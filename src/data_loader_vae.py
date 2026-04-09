import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from config import config_VAE


class MoleculeDatasetVAE(Dataset):
    def __init__(self, target_path: str, noisy_path: str, scaler=None):
        # ---- Load both files ----
        target_df = pd.read_csv(target_path)
        noisy_df = pd.read_csv(noisy_path)

        # ---- Merge exactly like your working dataloader ----
        df = pd.concat([target_df.reset_index(drop=True), noisy_df.reset_index(drop=True)], axis=1)
        df = df[["smiles", "fingerprint", "enthalpy", "h298"]]
        df.columns = ["molecule", "fingerprint", "input", "target"]

        # ---- Optional scaling ----
        if scaler is not None:
            df["input"] = scaler.fit_transform(df[["input"]].values).squeeze()
            df["target"] = scaler.transform(df[["target"]].values).squeeze()

        # ---- Fingerprints to matrix ----
        fp_matrix = np.vstack(
            df["fingerprint"].apply(lambda x: np.array(list(x), dtype=np.uint8))
        )

        # ---- Extra safety check against config ----
        if fp_matrix.shape[1] != config_VAE["fp_input_dim"]:
            raise ValueError(
                f"Fingerprint dimension mismatch: got {fp_matrix.shape[1]}, "
                f"expected {config_VAE['fp_input_dim']}"
            )

        # ---- Store as tensors ----
        self.names = df["molecule"].values
        self.fp = torch.tensor(fp_matrix, dtype=torch.float32)  # shape (N, fp_input_dim)
        self.noisy = torch.tensor(df["input"].values, dtype=torch.float32).unsqueeze(1)
        self.target = torch.tensor(df["target"].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # IMPORTANT:
        # For VAE pipeline we do NOT concatenate here.
        # We return the separate pieces.
        return self.fp[idx], self.noisy[idx], self.target[idx]


def get_dataloaders(
    dataset_path: str,
    noisy_path: str,
):
    scaler = StandardScaler()
    dataset = MoleculeDatasetVAE(dataset_path, noisy_path, scaler=scaler)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return {
        "train": DataLoader(
            train_ds,
            batch_size=config_VAE["batch_size"],
            shuffle=True,
            num_workers=config_VAE["num_workers"],
        ),
        "val": DataLoader(
            val_ds,
            batch_size=config_VAE["batch_size"],
            shuffle=False,
            num_workers=config_VAE["num_workers"],
        ),
    }