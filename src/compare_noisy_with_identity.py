import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from config import config
from train import train


class NoisyOnlyDatasetRaw(Dataset):
    def __init__(self, target_path, noisy_path):
        target_df = pd.read_csv(target_path)
        noisy_df = pd.read_csv(noisy_path)

        if "h298" in target_df.columns:
            target = target_df["h298"].values
        elif "enthalpy" in target_df.columns:
            target = target_df["enthalpy"].values
        else:
            raise ValueError("No clean target column found.")

        if "enthalpy" in noisy_df.columns:
            noisy = noisy_df["enthalpy"].values
        elif "h298" in noisy_df.columns:
            noisy = noisy_df["h298"].values
        else:
            raise ValueError("No noisy column found.")

        self.x = torch.tensor(noisy.reshape(-1, 1), dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def evaluate_identity(loader):
    total_loss = 0.0
    criterion = nn.MSELoss()

    for X, y in loader:
        preds = X.squeeze(1)
        loss = criterion(preds, y)
        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join(
        config["data_path"],
        "dataset",
        "groupadditivity_0.004.csv"
    )
    noisy_path = os.path.join(
        config["data_path"],
        "dataset",
        "noise0.4",
        "groupadditivity_0.004_noise0.4.csv"
    )

    dataset = NoisyOnlyDatasetRaw(dataset_path, noisy_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    loaders = {
        "train": DataLoader(train_ds, batch_size=64, shuffle=True),
        "val": DataLoader(val_ds, batch_size=64, shuffle=False),
    }

    # Identity baseline op exact dezelfde split
    train_identity = evaluate_identity(loaders["train"])
    val_identity = evaluate_identity(loaders["val"])

    print("\nIdentity baseline on SAME SPLIT")
    print(f"Train MSE: {train_identity:.6f}")
    print(f"Val MSE:   {val_identity:.6f}")

    # Klein noisy-only model
    model = SmallMLP().to(device)

    model_config = copy.deepcopy(config)
    model_config["epochs"] = 20
    model_config["lr"] = 1e-4
    model_config["patience"] = 10

    model, history = train(
        model=model,
        loaders=loaders,
        config=model_config,
        device=device,
    )

    print("\nBest noisy-only model losses:")
    print(f"Best train loss: {min(history['train_loss']):.6f}")
    print(f"Best val loss:   {min(history['val_loss']):.6f}")


if __name__ == "__main__":
    main()