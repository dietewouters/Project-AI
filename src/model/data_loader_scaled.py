# model/data_loader_scaled.py

import os
import glob
import scipy.sparse as sp
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict

from config import config


# --------------------------------------------------
# Scaling wrapper
# --------------------------------------------------
class TargetScaler:
    def __init__(self, scaling_type="none"):
        self.scaling_type = scaling_type.lower()
        self.scaler = None

        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

        if self.scaling_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaling_type == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unsupported scaling_type: {scaling_type}")

    def fit(self, values):
        if self.scaler is not None:
            self.scaler.fit(values)

            if self.scaling_type == "standard":
                self.mean_ = float(self.scaler.mean_[0])
                self.std_ = float(self.scaler.scale_[0])

            elif self.scaling_type == "minmax":
                self.min_ = float(self.scaler.data_min_[0])
                self.max_ = float(self.scaler.data_max_[0])

    def transform(self, values):
        if self.scaler is None:
            return values
        return self.scaler.transform(values)

    def inverse_transform(self, values):
        if self.scaler is None:
            return values
        return self.scaler.inverse_transform(values)

    def inverse_transform_tensor(self, tensor):
        if self.scaling_type == "none":
            return tensor

        if self.scaling_type == "standard":
            return tensor * self.std_ + self.mean_

        elif self.scaling_type == "minmax":
            return tensor * (self.max_ - self.min_) + self.min_

        else:
            raise ValueError(f"Unsupported scaling_type: {self.scaling_type}")

# --------------------------------------------------
# Base dataset with optional scaling
# --------------------------------------------------
class DynamicMoleculeDatasetScaled(Dataset):
    def __init__(self, dataset_config, scaler=None, features_dir='data/processed_features'):
        self.mode = dataset_config.get("mode", "precomputed")
        self.scaler = scaler
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
                    merged = pd.merge(n_df, t_df, on="molecule", how="inner")
                    all_dfs.append(merged)

            if not all_dfs:
                raise ValueError("No valid precomputed blocks found!")

            final_df = pd.concat(all_dfs, ignore_index=True)

            print(f"  [+] Dropping cross-noise duplicates natively...")
            final_df = final_df.drop_duplicates(subset=["molecule"]).reset_index(drop=True)

            if total_molecules > 0 and total_molecules < len(final_df):
                final_df = final_df.sample(n=total_molecules, random_state=None).reset_index(drop=True)
            else:
                final_df = final_df.sample(frac=1).reset_index(drop=True)

            print(f"  [=] TOTAL Final Unique Molecules: {len(final_df)}")
            aligned_indices = final_df["original_index"].tolist()

            noisy_values = final_df["input"].values.reshape(-1, 1).astype(np.float32)
            target_values = final_df["target"].values.reshape(-1, 1).astype(np.float32)

            if self.scaler is not None:
                noisy_values = self.scaler.transform(noisy_values)
                target_values = self.scaler.transform(target_values)

            self.noisy = torch.tensor(noisy_values, dtype=torch.float32)
            self.y_true = torch.tensor(target_values.squeeze(1), dtype=torch.float32)

        elif self.mode == "on-the-fly":
            if total_molecules > 0 and total_molecules < len(t_df):
                final_df = t_df.sample(n=total_molecules, random_state=None).reset_index(drop=True)
            else:
                final_df = t_df.sample(frac=1).reset_index(drop=True)

            print(f"  [>] Noisy Fractional Files Incorporated: Dynamically Generating Gaussian Variance")
            print(f"  [=] TOTAL Final Unique Molecules: {len(final_df)}")

            self.noise_std = dataset_config.get("noise_std", 0.01)
            aligned_indices = final_df["original_index"].tolist()

            clean_values = final_df["target"].values.reshape(-1, 1).astype(np.float32)

            # Bewaar ook originele clean values indien nodig
            self.clean_target_original = torch.tensor(clean_values, dtype=torch.float32)

            if self.scaler is not None:
                clean_values_scaled = self.scaler.transform(clean_values)
            else:
                clean_values_scaled = clean_values

            self.clean_target_for_noise = torch.tensor(clean_values_scaled, dtype=torch.float32)
            self.noisy = None
            self.y_true = torch.tensor(clean_values_scaled.squeeze(1), dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        chunks = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
        if chunks:
            print(f"  [+] Fetching sparse features from {len(chunks)} feature bank chunk(s)...")
            feature_bank = sp.vstack([sp.load_npz(c) for c in chunks])
            fp_matrix = feature_bank[aligned_indices].toarray()
            self.fp = torch.tensor(fp_matrix, dtype=torch.float32)
            print(f"  [✔] Sparse fingerprint array built: {self.fp.shape}")
        else:
            self.fp = torch.zeros((len(final_df), 1), dtype=torch.float32)

        self.names = final_df["molecule"].values

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self._get_items(idx)

    def _get_items(self, idx):
        fp = self.fp[idx]
        y_true = self.y_true[idx]

        if self.mode == "on-the-fly":
            # let op: noise_std zit hier dan in de GESCHAALDE ruimte
            clean_target = self.clean_target_for_noise[idx]
            noisy = clean_target + torch.randn_like(clean_target) * self.noise_std
        else:
            noisy = self.noisy[idx]

        return fp, noisy, y_true


# --------------------------------------------------
# KNN variant
# --------------------------------------------------
class DynamicMoleculeDatasetScaledKNN(DynamicMoleculeDatasetScaled):
    def __init__(self, dataset_config, scaler=None, k: int = 5, features_dir='data/processed_features'):
        super().__init__(dataset_config, scaler=scaler, features_dir=features_dir)
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


# --------------------------------------------------
# Cloud dataset
# --------------------------------------------------
class CloudMoleculeDataset(Dataset):
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


# --------------------------------------------------
# Build scaler from TRAIN split only
# --------------------------------------------------
def build_scaler_from_train(train_config):
    scaling_type = train_config.get("scaling", "none")
    scaler = TargetScaler(scaling_type=scaling_type)

    if scaling_type == "none":
        return scaler

    target_path = train_config["target_path"]
    t_df = pd.read_csv(target_path)

    values = t_df["h298"].values.reshape(-1, 1).astype(np.float32)
    scaler.fit(values)

    print(f"  [✔] Fitted {scaling_type} scaler on TRAIN targets only.")
    return scaler


# --------------------------------------------------
# Main dataloader factory
# --------------------------------------------------
def get_dataloaders_scaled(loaders_config, dataset_type="base", features_dir='data/processed_features'):
    """
    dataset_type:
        - "base"
        - "knn"

    loaders_config contains 'train', 'val', 'test'
    scaling is expected inside loaders_config['train']['scaling']
    """
    datasets = {}

    # fit scaler only on training split
    train_scaler = build_scaler_from_train(loaders_config["train"])

    if dataset_type == "base":
        dataset_class = DynamicMoleculeDatasetScaled
    elif dataset_type == "knn":
        dataset_class = DynamicMoleculeDatasetScaledKNN
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    for split in ["train", "val", "test"]:
        if split in loaders_config:
            print(f"\n[+] Executing Dataset Parser for --> {split.upper()} <-- split")

            if dataset_type == "knn":
                datasets[split] = dataset_class(
                    loaders_config[split],
                    scaler=train_scaler,
                    k=loaders_config[split].get("k", 5),
                    features_dir=features_dir
                )
            else:
                datasets[split] = dataset_class(
                    loaders_config[split],
                    scaler=train_scaler,
                    features_dir=features_dir
                )

    loaders = {}
    for split in ["train", "val", "test"]:
        if split in datasets:
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=config.get("batch_size", 64),
                shuffle=(split == "train"),
                num_workers=config.get("num_workers", 0)
            )

    return loaders, train_scaler


# --------------------------------------------------
# Cloud dataloaders
# --------------------------------------------------
def get_cloud_dataloaders(bundle_path):
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
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=config.get("batch_size", 64),
                shuffle=(split == "train"),
                num_workers=config.get("num_workers", 0)
            )

    return loaders