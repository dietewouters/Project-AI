import os
import sys
import torch

# ==========================================================
# 1. KAGGLE PATH CONFIGURATION
# ==========================================================
# Note: These are based on your current Kaggle environment structure.
KAGGLE_USERNAME = "christiancicirelli"
DATASET_SLUG    = "project-data"
SRC_SLUG        = "src-model" # Adjust if your source code slug is different

# Roots
KAGGLE_INPUT   = f"/kaggle/input/datasets/{KAGGLE_USERNAME}"
DATA_ROOT      = f"{KAGGLE_INPUT}/{DATASET_SLUG}/data/groupadditivity_h298"
SRC_ROOT       = f"{KAGGLE_INPUT}/{SRC_SLUG}"

# Inputs from Stage 1 & 3 (Working Directory)
SC_INDICES_DIR = "/kaggle/working/indices/scaffold_split"
NOISY_CSV      = "/kaggle/working/dataset/noise_structural/groupadditivity_structural.csv"

# Inputs from Kaggle Datasets (Read-Only)
TARGET_CSV     = f"{DATA_ROOT}/dataset/groupadditivity.csv"
FEATURES_DIR   = f"{DATA_ROOT}/dataset/processed_features" # Path to your .npz chunks

# Output Directory
OUT_DIR        = "/kaggle/working/results/cloud_datasets"

# ==========================================================
# 2. BUNDLING EXECUTION
# ==========================================================
# Add Source to Path to find the 'model' package
if os.path.exists(SRC_ROOT):
    sys.path.insert(0, SRC_ROOT)
    # Search for the actual package root if nested
    for root, dirs, files in os.walk(SRC_ROOT):
        if 'model' in dirs and '__init__.py' in os.listdir(os.path.join(root, 'model')):
            if root not in sys.path:
                sys.path.insert(0, root)
                print(f"✅ Source package found at: {root}")
            break

try:
    from model.data_loader import get_scaffold_dataloaders, export_to_cloud_bundle
except ImportError as e:
    print(f"❌ Error: {e}.")
    print(f"   Ensure your '{SRC_SLUG}' dataset is attached and contains the 'model' folder.")
    sys.exit(1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Verify inputs exist before heavy processing
    paths_to_check = {
        "Train Indices": f"{SC_INDICES_DIR}/train_indices.csv",
        "Val Indices":   f"{SC_INDICES_DIR}/val_indices.csv",
        "Test Indices":  f"{SC_INDICES_DIR}/test_indices.csv",
        "Target CSV":    TARGET_CSV,
        "Noisy CSV":     NOISY_CSV,
        "Features Dir":  FEATURES_DIR
    }
    
    missing = False
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"❌ Missing {name}: {path}")
            missing = True
    
    if missing:
        print("\nAborting. Please ensure Stage 1 and Stage 3 have finished successfully.")
        return

    splits_config = {
        "train": {'target': TARGET_CSV, 'noisy': NOISY_CSV, 'indices': paths_to_check["Train Indices"]},
        "val":   {'target': TARGET_CSV, 'noisy': NOISY_CSV, 'indices': paths_to_check["Val Indices"]},
        "test":  {'target': TARGET_CSV, 'noisy': NOISY_CSV, 'indices': paths_to_check["Test Indices"]}
    }

    print(f"\n🚀 Loading Data through Scaffold Dataloader...")
    # This might take a few minutes as it loads 7.9M mapping
    loaders = get_scaffold_dataloaders(splits_config, features_dir=FEATURES_DIR)

    # 1. Fit scaler on Training split
    scaler = None
    if "train" in loaders:
        print("\n⚖️  Fitting 'delta' mode scaler to Training set...")
        scaler = loaders["train"].dataset.get_scaler(mode="delta")

    # 2. Export Bundles
    print(f"\n📦 Exporting Bundles to {OUT_DIR}...")
    
    # Train/Val Bundle
    tv_loaders = {k: loaders[k] for k in ["train", "val"] if k in loaders}
    if tv_loaders:
        export_to_cloud_bundle(tv_loaders, os.path.join(OUT_DIR, "scaffold_train_val.pt"), scaler=scaler)

    # Test Bundle
    if "test" in loaders:
        export_to_cloud_bundle({"test": loaders["test"]}, os.path.join(OUT_DIR, "scaffold_test.pt"), scaler=scaler)

    print(f"\n✅ SUCCESS! PT bundles generated in: {OUT_DIR}")

if __name__ == "__main__":
    main()
