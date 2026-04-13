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
class DynamicMoleculeDataset(Dataset):
    def __init__(self, dataset_config, features_dir='data/processed_features'):
        self.mode = dataset_config.get("mode", "precomputed")
        total_molecules = int(dataset_config.get("total_molecules", 0))
        
        target_path = dataset_config["target_path"]
        indices_path = dataset_config["indices_path"]
        
        print(f"  [>] Master Target Dictionary: {os.path.basename(target_path)}")
        print(f"  [>] Master Index Dictionary: {os.path.basename(indices_path)}")
        
        t_df = pd.read_csv(target_path).rename(columns={"h298": "target", "smiles": "molecule"})
        with open(indices_path, 'r') as f:
            base_indices = [int(i) for i in f.read().split()]
        t_df["original_index"] = base_indices
        
        if self.mode == "precomputed":
            blocks = dataset_config.get("blocks", [])
            all_dfs = []
            
            print("  [>] Noisy Fractional Files Incorporated:")
            for block in blocks:
                noisy_p = block["noisy_path"]
                
                if os.path.exists(noisy_p):
                    print(f"      - {os.path.basename(noisy_p)}")
                    
                    n_df = pd.read_csv(noisy_p).rename(columns={"h298": "input", "smiles": "molecule"})
                    
                    # Merge on molecule perfectly maps the noisy value directly onto the CLEAN file's correct targets and `.npz` indices
                    merged = pd.merge(n_df, t_df, on="molecule", how="inner")
                    all_dfs.append(merged)
                    
            if not all_dfs:
                raise ValueError("No valid precomputed blocks found!")
                
            final_df = pd.concat(all_dfs, ignore_index=True)
            
            print(f"  [+] Dropping cross-noise duplicates natively...")
            final_df = final_df.drop_duplicates(subset=["molecule"]).reset_index(drop=True)
            
            if total_molecules > 0 and total_molecules < len(final_df):
                # Sample specifically down to the required explicit bound if tracking caused overfetching
                final_df = final_df.sample(n=total_molecules, random_state=None).reset_index(drop=True)
            else:
                final_df = final_df.sample(frac=1).reset_index(drop=True)
                
            print(f"  [=] TOTAL Final Unique Molecules: {len(final_df)}")
            aligned_indices = final_df["original_index"].tolist()
            self.noisy = torch.tensor(final_df["input"].values, dtype=torch.float32).unsqueeze(1)
            
        elif self.mode == "on-the-fly":
            if total_molecules > 0 and total_molecules < len(t_df):
                final_df = t_df.sample(n=total_molecules, random_state=None).reset_index(drop=True)
            else:
                final_df = t_df.sample(frac=1).reset_index(drop=True)
                
            print(f"  [>] Noisy Fractional Files Incorporated: Dynamically Generating Gaussian Variance")
            print(f"  [=] TOTAL Final Unique Molecules: {len(final_df)}")
            self.noise_std = dataset_config.get("noise_std", 0.01)
            self.noisy = None
            aligned_indices = final_df["original_index"].tolist()
            self.clean_target_for_noise = torch.tensor(final_df["target"].values, dtype=torch.float32).unsqueeze(1)
            
        chunks = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
        if chunks:
            print(f"  [+] Fetching sparse features from {len(chunks)} feature bank chunk(s)...")
            feature_bank = sp.vstack([sp.load_npz(c) for c in chunks])
            fp_matrix = feature_bank[aligned_indices].toarray()
            self.fp = torch.tensor(fp_matrix, dtype=torch.float32)
            print(f"  [✔] Sparse fingerprint array built: {self.fp.shape}")
        else:
            self.fp = torch.zeros((len(final_df), 1))  # Fallback
        
        self.y_true = torch.tensor(final_df["target"].values, dtype=torch.float32)
        self.names = final_df["molecule"].values

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self._get_items(idx)

    def _get_items(self, idx):
        fp = self.fp[idx]
        y_true = self.y_true[idx]
        
        if self.mode == "on-the-fly":
            # Add Gaussian noise with given standard deviation on the fly
            clean_target = self.clean_target_for_noise[idx]
            noisy = clean_target + torch.randn_like(clean_target) * self.noise_std
        else:
            noisy = self.noisy[idx]
            
        return fp, noisy, y_true


class DynamicMoleculeDatasetKNN(DynamicMoleculeDataset):
    def __init__(self, dataset_config, k: int = 5, features_dir='data/processed_features'):
        super().__init__(dataset_config, features_dir)
        self.k_nn = self.compute_knn_features(k)

    def compute_knn_features(self, k):
        X = self.fp.numpy() 
        y = self.y_true.numpy()

        knn = KNeighborsRegressor(n_neighbors=k, metric='jaccard', n_jobs=-1) 
        knn_features = cross_val_predict(knn, X, y, cv=5)
        return torch.tensor(knn_features, dtype=torch.float32).unsqueeze(1)

    def _get_items(self, idx):
        fp, noisy, y_true = super()._get_items(idx)
        x = torch.cat([fp, noisy, self.k_nn[idx]], dim=0)
        return x, y_true

class CloudMoleculeDataset(Dataset):
    """
    A lightweight detached dataset that purely consumes pre-extracted Tensors.
    This completely bypasses pandas, CSVs, and .npz parsing for Cloud/Kaggle environments.
    """
    def __init__(self, tensor_dict):
        self.fp = tensor_dict["fp"]
        self.noisy = tensor_dict["noisy"]
        self.y_true = tensor_dict["y_true"]

    def __len__(self):
        return len(self.fp)

    def __getitem__(self, idx):
        fp = self.fp[idx]
        noisy = self.noisy[idx]
        y_true = self.y_true[idx]
        return fp, noisy, y_true

def export_to_cloud_bundle(loaders, export_path):
    """ Extracts base PyTorch tensors from instantiated datasets and seals them in a single .pt """
    import os
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    bundle = {}
    for split, loader in loaders.items():
        ds = loader.dataset
        # Regardless of mode (precomputed or on-the-fly), the dataset caches these static tensors
        bundle[split] = {
            "fp": ds.fp,
            "noisy": ds.noisy if ds.noisy is not None else ds.clean_target_for_noise, # Simplified logic if pre-generated
            "y_true": ds.y_true
        }
    torch.save(bundle, export_path)

def get_dataloaders(loaders_config, features_dir='data/processed_features'):
    """
    loaders_config: dict with 'train', 'val', 'test' keys.
    Each contains a dataset_config dictionary compatible with DynamicMoleculeDataset.
    """
    datasets = {}
    dataset_class = DynamicMoleculeDataset
    
    for split in ["train", "val", "test"]:
        if split in loaders_config:
            print(f"\n[+] Executing Dataset Parser for --> {split.upper()} <-- split")
            datasets[split] = dataset_class(loaders_config[split], features_dir=features_dir)
            
    loaders = {}
    for split in ["train", "val", "test"]:
        if split in datasets:
            loaders[split] = DataLoader(datasets[split], batch_size=config.get("batch_size", 64), shuffle=(split=="train"), num_workers=config.get("num_workers", 0))
            
    return loaders

def get_cloud_dataloaders(bundle_path):
    """
    Instantiates DataLoader objects directly off a pre-extracted Kaggle/Cloud .pt payload
    """
    bundle = torch.load(bundle_path)
        
    datasets = {}
    for split, tensor_dict in bundle.items():
        if split == "metadata":
            continue
        print(f"  [+] Unpacking Cloud payload block -> {split.upper()}")
        datasets[split] = CloudMoleculeDataset(tensor_dict)
        
    loaders = {}
    for split in ["train", "val", "test"]:
        if split in datasets:
            loaders[split] = DataLoader(datasets[split], batch_size=config.get("batch_size", 64), shuffle=(split=="train"), num_workers=config.get("num_workers", 0))
            
    return loaders

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
        # Based on your class, this returns: fp, noisy, y_true
        sample_fp, sample_noisy, sample_true = ds[0]
        
        print(f"Fingerprint shape: {sample_fp.shape}")
        print(f"Fingerprint (first 10 bits): {sample_fp[:10]}")
        print(f"Noisy Input (Scaled): {sample_noisy.item():.4f}")
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





