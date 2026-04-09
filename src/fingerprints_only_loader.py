import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from config import config


class FingerprintOnlyDataset(Dataset):
    def __init__(self, target_path, noisy_path, scaler=None):
        target_df = pd.read_csv(target_path)
        noisy_df = pd.read_csv(noisy_path)

        df = pd.concat(
            [target_df.reset_index(drop=True), noisy_df.reset_index(drop=True)],
            axis=1
        )
        df = df[["fingerprint", "h298"]]
        df.columns = ["fingerprint", "target"]

        if scaler is not None:
            df["target"] = scaler.fit_transform(df[["target"]].values).squeeze()

        fp_matrix = np.vstack(
            df["fingerprint"].apply(lambda x: np.array(list(x), dtype=np.uint8))
        )

        self.x = torch.tensor(fp_matrix, dtype=torch.float32)
        self.y = torch.tensor(df["target"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_fingerprint_dataloaders(dataset_path, noisy_path):
    scaler = StandardScaler()
    dataset = FingerprintOnlyDataset(dataset_path, noisy_path, scaler=scaler)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return {
        "train": DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        ),
        "val": DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        ),
    }