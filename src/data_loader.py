# Loading, splitting, preprocessing data

import os
from audioop import cross

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
    def __init__(self, target_path: str, noisy_path: str, indices_path: str = None, scaler=None, features_dir: str = 'data/processed_features'):
        target_df = pd.read_csv(target_path)
        noisy_df = pd.read_csv(noisy_path)

        df = pd.concat([target_df.reset_index(drop=True), noisy_df.reset_index(drop=True)], axis=1)
        
        # Check if the CSV still retains the raw text fingerprint for backward compatibility parsing
        if "fingerprint" in df.columns.values or "fingerprint" in df.columns:
            df = df[["smiles", "fingerprint", "enthalpy", "h298"]]
            df.columns = ["molecule", "fingerprint", "input", "target"]
        else:
            df = df[["smiles", "enthalpy", "h298"]]
            df.columns = ["molecule", "input", "target"]

        if scaler is not None:
            df["input"] = scaler.fit_transform(df[["input"]].values).squeeze()
            df["target"] = scaler.transform(df[["target"]].values).squeeze()

        # ---- Delta target ----
        df["delta"] = df["input"] - df["target"]

        # Parse from the extremely lightweight .npz chunks dynamically based on explicit global indices!
        if indices_path is not None and os.path.exists(indices_path) and os.path.exists(features_dir):
            import glob
            import scipy.sparse as sp
            
            with open(indices_path, 'r') as f:
                content = f.read().strip()
                global_indices = [int(idx) for idx in content.split()]
                
            npz_files = sorted(glob.glob(os.path.join(features_dir, "features_chunk_*.npz")))
            if not npz_files:
                raise FileNotFoundError(f"No sparse arrays found in {features_dir}")
                
            # Statically vstacking to memory is ~3GB RAM (handled easily by M-series Macs & GPULabs)
            all_fp_sparse = sp.vstack([sp.load_npz(f) for f in npz_files])
            
            # Select exactly the fingerprints required for this subset!
            subset_fp_sparse = all_fp_sparse[global_indices]
            fp_matrix = subset_fp_sparse.toarray()
        else:
            # Fallback legacy pipeline: converting long CSV strings
            fp_matrix = np.vstack(df["fingerprint"].apply(lambda x: np.array(list(x), dtype=np.uint8)))

        self.names = df["molecule"].values
        self.fp = torch.tensor(fp_matrix, dtype=torch.float32)
        self.noisy = torch.tensor(df["input"].values, dtype=torch.float32).unsqueeze(1)
        self.y_delta = torch.tensor(df["delta"].values, dtype=torch.float32)
        self.y_true = torch.tensor(df["target"].values, dtype=torch.float32)

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        #x = torch.cat([self.fp[idx], self.noisy[idx]], dim=0)
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
    import os
    
    # Auto-resolve the specific indices filename based on the dataset CSV
    basename = os.path.basename(dataset_path)
    parts = basename.split('_')
    if len(parts) > 1:
        indices_filename = "indices_" + parts[-1]
    else:
        indices_filename = basename
    indices_path = os.path.join(indices_dir, indices_filename)

    scaler = StandardScaler()
    dataset = MoleculeDataset(dataset_path, noisy_path, indices_path=indices_path, scaler=scaler)
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







