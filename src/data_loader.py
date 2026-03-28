# Loading, splitting, preprocessing data

import os
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
import torch
from config import config
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
###### THIS DATALOADER ONLY WORKS ON A SPECIFIC FRACTION OF THE DATA!! ##########
 ###### HAS TO BE UPDATED TO WORK ON THE ENTIRE DATASET!!!!! #####################

# K-NN ADDED -------!!!!!!!!!!!!!!!!!

class MoleculeDataset(Dataset):
    def __init__(self, target_path: str, noisy_path: str, indices_path: str = None, scaler = None):
        # ---- Load both files ----
        target_df = pd.read_csv(target_path)
        noisy_df = pd.read_csv(noisy_path)

        #---- Merge with indices file when full fingerprints ----
        df = pd.concat([target_df.reset_index(drop=True), noisy_df.reset_index(drop=True)], axis=1)
        df = df[["smiles", "fingerprint", "enthalpy", "h298"]]
        df.columns = ["molecule", "fingerprint", "input", "target"]

        # ---- Optional scaling ----
        if scaler is not None:
            df["input"] = scaler.fit_transform(df[["input"]].values).squeeze()
            df["target"] = scaler.transform(df[["target"]].values).squeeze()

        fp_matrix = np.vstack(df["fingerprint"].apply(lambda x: np.array(list(x), dtype=np.uint8)))

        # ---- Store as tensors ----
        self.names = df["molecule"].values
        self.fp = torch.tensor(fp_matrix, dtype=torch.float32)  # shape (N, 2048)
        self.noisy = torch.tensor(df["input"].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(df["target"].values, dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = torch.cat([self.fp[idx], self.noisy[idx]], dim=0)  # (2049,) for now
        return x, self.y[idx]

def get_dataloaders(
        dataset_path: str,
        noisy_path: str,
        indices_dir: str
):
    scaler = StandardScaler()
    dataset = MoleculeDataset(dataset_path, noisy_path, scaler=scaler)
    #val_ds = MoleculeDataset('data/groupadditivity_h298/dataset/groupadditivity_secondarytest.csv', 'data/groupadditivity_h298/dataset/noise0.01/groupadditivity_secondarytest_noise0.01.csv')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    #test_ds = MoleculeDataset('data/groupadditivity_h298/dataset/groupadditivity_test.csv', 'data/groupadditivity_h298/dataset/noise0.01/groupadditivity_test_noise0.01.csv')
    return {
        "train": DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]),
        "val": DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"]),
        #"test": DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"]),
    }







