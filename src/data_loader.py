# Loading, splitting, preprocessing data

import scipy.sparse as sp
import os
from audioop import cross
import glob
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
import torch
from config import config
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
###### THIS DATALOADER ONLY WORKS ON A SPECIFIC FRACTION OF THE DATA!! ##########
 ###### HAS TO BE UPDATED TO WORK ON THE ENTIRE DATASET!!!!! #####################

# K-NN ADDED -------!!!!!!!!!!!!!!!!!

class MoleculeDataset(Dataset):
    def __init__(self, target_path, noisy_path, indices_path, features_dir='data/processed_features', scaler=None):
        # 1. Load CSVs (Smiles and h298 only)
        t_df = pd.read_csv(target_path).rename(columns={"h298": "target", "smiles": "molecule"})
        n_df = pd.read_csv(noisy_path).rename(columns={"h298": "input"})
        df = pd.concat([t_df, n_df[["input"]]], axis=1)

        # 2. Delta & Scaling
        if scaler:
            df["input"] = scaler.fit_transform(df[["input"]].values).squeeze()
            df["target"] = scaler.transform(df[["target"]].values).squeeze()
        df["delta"] = df["input"] - df["target"]

        # 3. Load Features from Chunks using Global Indices
        with open(indices_path, 'r') as f:
            indices = [int(i) for i in f.read().split()]
        
        # Load and stack all chunks into one large sparse bank
        chunks = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
        feature_bank = sp.vstack([sp.load_npz(c) for c in chunks])
        
        # Pull only the fingerprints we need for these specific CSV rows
        fp_matrix = feature_bank[indices].toarray()

        # 4. Final Tensors
        self.fp = torch.tensor(fp_matrix, dtype=torch.float32)
        self.noisy = torch.tensor(df["input"].values, dtype=torch.float32).unsqueeze(1)
        self.y_delta = torch.tensor(df["delta"].values, dtype=torch.float32)
        self.y_true = torch.tensor(df["target"].values, dtype=torch.float32)
        self.names = df["molecule"].values

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.fp[idx], self.noisy[idx], self.y_delta[idx], self.y_true[idx]

class MoleculeDatasetKNN(MoleculeDataset):
    def __init__(self, target_path: str, noisy_path: str, k : int = 5, indices_path: str = None, scaler = None):
        super().__init__(target_path, noisy_path, indices_path, scaler)
        self.k_nn = self.compute_knn_features(k)

    def compute_knn_features(self, k):
        X = self.fp.numpy() # 2048-bit fingerprints
        y = self.y.numpy() # targets

        knn = KNeighborsRegressor(n_neighbors=k, metric='jaccard', n_jobs=-1) # Jaccard is Tanimoto in binary case

        knn_features = cross_val_predict(knn, X, y, cv=5)
        return torch.tensor(knn_features, dtype=torch.float32).unsqueeze(1)

    def __getitem__(self, idx):
        # Combineer: Fingerprint (2048) + Noisy Value (1) + k-NN (1)
        # Dit resulteert in een input vector van (2050,)
        x = torch.cat([self.fp[idx], self.noisy[idx], self.k_nn[idx]], dim=0)

        # Return de gecombineerde input en de target (zonder ruis)
        return x, self.y[idx]

def get_dataloaders(
        dataset_path: str,
        noisy_path: str,
        indices_dir: str
):
    
    # Auto-resolve the specific indices filename based on the dataset CSV
    basename = os.path.basename(dataset_path)
    parts = basename.split('_')
    if len(parts) > 1:
        indices_filename = "indices_" + parts[-1]
    else:
        indices_filename = basename
    indices_path = os.path.join(indices_dir, indices_filename)

    scaler = StandardScaler()
    train_ds = MoleculeDataset(dataset_path, noisy_path, indices_path=indices_path, scaler=scaler)
    val_ds = MoleculeDataset('data/groupadditivity_h298/dataset/groupadditivity_secondarytest.csv', 'data/groupadditivity_h298/dataset/noise0.01/groupadditivity_secondarytest_noise0.01.csv', indices_path='data/groupadditivity_h298/indices/indices_secondarytest.csv', scaler=scaler)
    test_ds = MoleculeDataset('data/groupadditivity_h298/dataset/groupadditivity_test.csv', 'data/groupadditivity_h298/dataset/noise0.01/groupadditivity_test_noise0.01.csv', indices_path='data/groupadditivity_h298/indices/indices_test.csv', scaler=scaler)
    return {
        "train": DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]),
        "val": DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"]),
        "test": DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"]),
    }

def main():
    dataset_path = 'data/groupadditivity_h298/dataset/groupadditivity_0.004.csv'
    noisy_path = 'data/groupadditivity_h298/dataset/noise0.01/groupadditivity_0.004_noise0.01.csv'
    indices_dir = 'data/groupadditivity_h298/indices'
    
    loaders = get_dataloaders(dataset_path, noisy_path, indices_dir)

    for split_name, loader in loaders.items():
        print(f"\n--- {split_name.upper()} DATASET ---")
        
        # Access the custom MoleculeDataset object
        ds = loader.dataset
        
        print(f"Total samples: {len(ds)}")
        
        # Grab the first item (index 0)
        # Based on your class, this returns: fp, noisy, y_delta, y_true
        sample_fp, sample_noisy, sample_delta, sample_true = ds[0]
        
        print(f"Fingerprint shape: {sample_fp.shape}")
        print(f"Fingerprint (first 10 bits): {sample_fp[:10]}")
        print(f"Noisy Input (Scaled): {sample_noisy.item():.4f}")
        print(f"Target Delta (Label): {sample_delta.item():.4f}")
        print(f"True h298: {sample_true.item():.4f}")
        
        # Verify the Delta math: Noisy - True = Delta
        # (Allowing for small float precision differences)
        calc_delta = sample_noisy.item() - sample_true.item()
        print(f"Manual Delta Check: {calc_delta:.4f}")

if __name__ == "__main__":
    main()

def count_sparse_features(features_dir='data/processed_features'):
    # Find all .npz files in the directory
    npz_files = sorted(glob.glob(os.path.join(features_dir, "features_chunk_*.npz")))
    
    if not npz_files:
        print(f"No .npz files found in {features_dir}")
        return

    print(f"{'File Name':<30} | {'Molecules (Rows)':<15} | {'Active Features (nnz)':<20}")
    print("-" * 75)

    total_rows = 0
    total_nnz = 0

    for file_path in npz_files:
        file_name = os.path.basename(file_path)
        
        # Load the sparse matrix
        matrix = sp.load_npz(file_path)
        
        rows = matrix.shape[0]  # Number of rows
        nnz = matrix.nnz        # Number of non-zero elements
        
        total_rows += rows
        total_nnz += nnz
        
        print(f"{file_name:<30} | {rows:<15} | {nnz:<20}")

    print("-" * 75)
    print(f"{'TOTAL':<30} | {total_rows:<15} | {total_nnz:<20}")

if __name__ == "__main__":
    # Adjust this path to where your files are actually stored
    main()
    #count_sparse_features('data/processed_features')





