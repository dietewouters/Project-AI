import scipy.sparse as sp
import os
import glob
import pandas as pd
import numpy as np
import torch
from config import config
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import cross_val_predict
from model.scaler import PropertyScaler

# Unified Data Loader containing both Dynamic and Scaffold-specific datasets

# PropertyScaler has been moved to model/scaler.py
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
            
            # Shuffle BEFORE dropping duplicates to ensure balanced noise distribution
            # This randomizes which noisy version of a molecule is kept during de-duplication
            final_df = final_df.sample(frac=1, random_state=None).reset_index(drop=True)
            
            print(f"  [+] Dropping cross-noise duplicates natively...")
            final_df = final_df.drop_duplicates(subset=["molecule"]).reset_index(drop=True)
            
            if total_molecules > 0 and total_molecules < len(final_df):
                # Sample down to the required explicit bound
                final_df = final_df.sample(n=total_molecules, random_state=None).reset_index(drop=True)
            
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
            # Add Fully Random (Uniform) noise on the fly: range [-noise_std, noise_std]
            clean_target = self.clean_target_for_noise[idx]
            noise = (torch.rand_like(clean_target) - 0.5) * 2 * self.noise_std
            noisy = clean_target + noise
        else:
            noisy = self.noisy[idx]
            
        return fp, noisy, y_true

    def get_scaler(self, mode="delta"):
        """ Returns a fitted PropertyScaler for this dataset """
        scaler = PropertyScaler(mode=mode)
        
        # Determine noisy values for fitting
        if self.noisy is not None:
            noisy_vals = self.noisy
        else:
            # Handle on-the-fly mode: generate a sample of noisy values
            print("  [Scaler] Generating on-the-fly sample for fitting...")
            noise = torch.randn_like(self.y_true) * self.noise_std
            noisy_vals = self.y_true + noise
            
        scaler.fit(noisy_vals, self.y_true)
        return scaler


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
        # SIMPLE LOADING: No bit-unpacking needed anymore.
        # If the tensor is half-precision (float16), we return it as is.
        # The model/train loop will handle casting to float32.
        return self.fp[idx], self.noisy[idx], self.y_true[idx]

    def get_scaler(self, mode="delta"):
        """ Returns a fitted PropertyScaler for this cloud dataset payload """
        scaler = PropertyScaler(mode=mode)
        scaler.fit(self.noisy, self.y_true)
        return scaler

def export_to_cloud_bundle(loaders, export_path, scaler=None):
    """ Extracts base PyTorch tensors from instantiated datasets and seals them in a single .pt """
    import os
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    bundle = {}
    
    # Use provided scaler for metadata if available
    scaler_dict = None
    if scaler is not None:
        scaler_dict = scaler.to_dict()
        bundle["metadata"] = scaler_dict

    for split, loader in loaders.items():
        ds = loader.dataset
        
        # Convert fingerprints to Half-Precision (Float16)
        # This reduces storage by 2x compared to float32 without the complexity of bit-packing
        print(f"  [>] Converting {split} to Float16 for storage...")
        fp_final = ds.fp.to(torch.float16)
        
        bundle[split] = {
            "fp": fp_final,
            "noisy": ds.noisy if ds.noisy is not None else ds.clean_target_for_noise,
            "y_true": ds.y_true
        }
    torch.save(bundle, export_path)
    if scaler_dict:
        print(f"✅ Exported bundle with scaling metadata: {scaler_dict}")

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
    Instantiates DataLoader objects directly off a pre-extracted Kaggle/Cloud .pt payload.
    Uses mmap=True to keep RAM usage minimal even for 16GB datasets.
    """
    # Use mmap=True to ensure the 16GB Float16 tensor doesn't flood RAM
    bundle = torch.load(bundle_path, mmap=True, weights_only=False)
        
    datasets = {}
    for split, tensor_dict in bundle.items():
        if split == "metadata":
            continue
        print(f"  [+] Loading Cloud payload block -> {split.upper()}")
        datasets[split] = CloudMoleculeDataset(tensor_dict)
        
    loaders = {}
    for split in ["train", "val", "test"]:
        if split in datasets:
            loaders[split] = DataLoader(datasets[split], batch_size=config.get("batch_size", 64), shuffle=(split=="train"), num_workers=config.get("num_workers", 0))
            
    return loaders

def main():
    # Example usage / Debugging entry point
    pass

if __name__ == "__main__":
    main()


class SimpleScaffoldDataset(Dataset):
    """
    A simplified dataset specifically for scaffold-split experiments with structural noise.
    Directly subsets CSVs and Fingerprints by provided indices.
    """
    def __init__(self, target_csv_path, noisy_csv_path, indices_path, features_dir, index_offset=0):
        print(f"  [Scaffold] Initializing... ")
        print(f"  [Scaffold] Target: {os.path.basename(target_csv_path)}")
        print(f"  [Scaffold] Noisy:  {os.path.basename(noisy_csv_path)}")
        print(f"  [Scaffold] Indices: {os.path.basename(indices_path)}")

        # 1. Load Indices (Global positions)
        with open(indices_path, 'r') as f:
            self.indices = [int(i) for i in f.read().split()]
        
        # Calculate local indices for the quarterly noisy file
        local_indices = [i - index_offset for i in self.indices]
        print(f"  [Scaffold] Loaded {len(self.indices)} indices (Offset: {index_offset}).")

        # 2. Load and Subset target CSV (Full File -> Use Global Indices)
        t_df = pd.read_csv(target_csv_path)
        t_df = t_df.rename(columns={"h298": "target", "smiles": "molecule"})
        t_df_subset = t_df.iloc[self.indices].copy()

        # 3. Load and Subset noisy CSV (Partial File -> Use Local Indices)
        n_df = pd.read_csv(noisy_csv_path)
        n_df = n_df.rename(columns={"h298": "noisy_input", "smiles": "molecule"})
        n_df_subset = n_df.iloc[local_indices].copy()

        # 4. Extract Tensors
        self.y_true = torch.tensor(t_df_subset["target"].values, dtype=torch.float32)
        self.noisy = torch.tensor(n_df_subset["noisy_input"].values, dtype=torch.float32).unsqueeze(1)
        self.molecules = t_df_subset["molecule"].values

        # 5. Load Fingerprints (subsetted from the global bank)
        chunks = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
        if chunks:
            print(f"  [Scaffold] Loading {len(chunks)} fingerprint chunks...")
            feature_bank = sp.vstack([sp.load_npz(c) for c in chunks])
            
            # BIT-PACKING: Load and compress immediately to save RAM
            fp_subset = feature_bank[self.indices].toarray().astype(np.uint8)
            packed_fp = np.packbits(fp_subset, axis=-1)
            self.fp = torch.from_numpy(packed_fp)
            print(f"  [Scaffold] Packed fingerprints ready: {self.fp.shape} (uint8)")
        else:
            print("  [Scaffold] ⚠️ Warning: No fingerprints found in features_dir!")
            self.fp = torch.zeros((len(self.indices), 1))

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        # SIMPLE LOADING: Directly return the stored tensors.
        return self.fp[idx], self.noisy[idx], self.y_true[idx]

    def get_scaler(self, mode="delta"):
        """ Fitting a PropertyScaler manually since this loader is detached from the main config """
        # PropertyScaler is already imported at top level in this file
        scaler = PropertyScaler(mode=mode)
        scaler.fit(self.noisy, self.y_true)
        return scaler

def get_scaffold_dataloaders(splits_config, features_dir, batch_size=64):
    """
    splits_config = {
        'train': {'target': '...', 'noisy': '...', 'indices': '...'},
        'val':   {...},
        'test':  {...}
    }
    """
    loaders = {}
    for split, paths in splits_config.items():
        # Extract the start row for index offsetting
        # We assume the last integer in the noisy filename or directory might hint at the quarter,
        # but the safest way is to let the Stage 4 script provide it.
        # Here we look for the quarter index in the config if provided.
        index_offset = paths.get('index_offset', 0)
        
        dataset = SimpleScaffoldDataset(
            target_csv_path=paths['target'],
            noisy_csv_path=paths['noisy'],
            indices_path=paths['indices'],
            features_dir=features_dir,
            index_offset=index_offset
        )
        loaders[split] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(split == 'train'),
            num_workers=0
        )
    return loaders
class ARSDataset(Dataset):
    """
    A specialized dataset for Adaptive Residual Scaling.
    Handles bit-unpacking for fingerprints automatically to ensure
    the model always receives 1024-bit float features.
    """
    def __init__(self, fp, noisy, y_true):
        self.fp = fp
        self.noisy = noisy
        self.y_true = y_true

    def __len__(self):
        return len(self.fp)

    def __getitem__(self, idx):
        fp = self.fp[idx]
        
        # Handle bit-unpacking if stored as packed uint8
        if fp.dtype == torch.uint8:
            # np.unpackbits is faster than manual logic
            fp_np = fp.numpy() if not isinstance(fp, np.ndarray) else fp
            fp_unpacked = torch.from_numpy(np.unpackbits(fp_np).astype(np.float32))
        else:
            fp_unpacked = fp.float()

        return fp_unpacked, self.noisy[idx], self.y_true[idx]


