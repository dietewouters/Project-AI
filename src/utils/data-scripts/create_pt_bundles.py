import os
import sys
import torch

# Add the Project-AI/src directory to sys.path to import local modules
project_src = os.path.join(os.getcwd(), 'Project-AI', 'src')
if project_src not in sys.path:
    sys.path.append(project_src)

from model.data_loader import get_dataloaders, export_to_cloud_bundle
from model.scaler import PropertyScaler

# Paths
DATA_ROOT = os.path.join(os.getcwd(), 'data', 'groupadditivity_h298')
TARGET_CSV = os.path.join(DATA_ROOT, 'dataset', 'groupadditivity.csv')
STRUCTURAL_CSV = os.path.join(DATA_ROOT, 'dataset', 'noise_structural', 'groupadditivity_structural.csv')
INDICES_DIR = os.path.join(DATA_ROOT, 'indices', 'scaffold_split')

def create_bundles():
    splits = ["train", "val", "test"]
    loaders_config = {}

    for split in splits:
        indices_path = os.path.join(INDICES_DIR, f"{split}_indices.csv")
        
        # Determine number of molecules in this split automatically
        with open(indices_path, 'r') as f:
            n_mols = len(f.read().split())
            
        loaders_config[split] = {
            "mode": "precomputed",
            "target_path": TARGET_CSV,
            "indices_path": indices_path,
            "total_molecules": n_mols,
            "blocks": [
                {
                    "clean_path": TARGET_CSV,
                    "noisy_path": STRUCTURAL_CSV,
                    "indices_path": indices_path
                }
            ]
        }

    print("Instantiating DataLoaders for all splits...")
    # Map back to feature bank in 'data/processed_features'
    loaders = get_dataloaders(loaders_config, features_dir=os.path.join(os.getcwd(), 'data', 'processed_features'))

    # Train/Val Bundle
    print("\nScaling and Exporting Train/Val Bundle...")
    train_val_loaders = {"train": loaders["train"], "val": loaders["val"]}
    
    # Fit scaler using training data
    scaler = loaders["train"].dataset.get_scaler(mode="delta")
    
    export_path_tr = os.path.join(os.getcwd(), 'results', 'cloud_datasets', 'structural_train_val.pt')
    export_to_cloud_bundle(train_val_loaders, export_path_tr, scaler=scaler)
    print(f"Exported: {export_path_tr}")

    # Test Bundle
    print("\nExporting Test Bundle...")
    test_loaders = {"test": loaders["test"]}
    export_path_te = os.path.join(os.getcwd(), 'results', 'cloud_datasets', 'structural_test.pt')
    export_to_cloud_bundle(test_loaders, export_path_te, scaler=scaler)
    print(f"Exported: {export_path_te}")

if __name__ == "__main__":
    create_bundles()
