import torch
from torch.utils.data import Dataset, DataLoader


class LatentDataset(Dataset):
    def __init__(self, z, noisy, target):
        self.x = torch.cat([z, noisy], dim=1)  # shape: (N, latent_dim + 1)
        self.y = target.squeeze(1)             # shape: (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_latent_dataloaders(
    z_train, noisy_train, y_train,
    z_val, noisy_val, y_val,
    batch_size,
    num_workers,
):
    train_ds = LatentDataset(z_train, noisy_train, y_train)
    val_ds = LatentDataset(z_val, noisy_val, y_val)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    return loaders