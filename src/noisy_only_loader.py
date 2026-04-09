import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from config import config


class NoisyOnlyDataset(Dataset):
    def __init__(self, target_path, noisy_path, scaler=None):
        target_df = pd.read_csv(target_path)
        noisy_df = pd.read_csv(noisy_path)

        # neem noisy en clean target
        noisy = noisy_df["h298"].values
        target = target_df["enthalpy"].values

        # reshape
        noisy = noisy.reshape(-1, 1)
        target = target.reshape(-1, 1)

        # scaling (zelfde als andere modellen!)
        if scaler is not None:
            noisy = scaler.fit_transform(noisy)
            target = scaler.transform(target)

        self.x = torch.tensor(noisy, dtype=torch.float32)   # shape (N,1)
        self.y = torch.tensor(target, dtype=torch.float32).squeeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_noisy_dataloaders(dataset_path, noisy_path):
    scaler = StandardScaler()
    dataset = NoisyOnlyDataset(dataset_path, noisy_path, scaler=scaler)

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