import os
import sys
import torch

# ==========================================================
# 1. SETUP YOUR PATHS HERE
# ==========================================================
# Point this to the folder that CONTAINS the 'model' package.
# This script will attempt to find your local path automatically.
KAGGE_SRC      = "/kaggle/input/dataset/username/dataset-slug"
LOCAL_SRC      = os.path.join(os.getcwd(), "Project-AI/src")

# Use local path if it exists, otherwise fallback to Kaggle placeholder
SRC_PATH = LOCAL_SRC if os.path.exists(LOCAL_SRC) else KAGGE_SRC

# Paths to your 3 scaffold index files (update these to your local paths)
SC_TRAIN_IDX  = "path/to/scaffold_split/train_indices.csv"
SC_VAL_IDX    = "path/to/scaffold_split/val_indices.csv"
SC_TEST_IDX   = "path/to/scaffold_split/test_indices.csv"

# Paths to the raw molecular data (clean vs noisy)
TARGET_CSV    = "path/to/groupadditivity.csv"
NOISY_CSV     = "path/to/groupadditivity_structural_or_0.6.csv"

# Directory containing your processed fingerprint .npz chunks
FEATURES_DIR  = "path/to/processed_features"

# Where to save the final .pt bundles
OUT_DIR       = "path/to/results/cloud_datasets"


# ==========================================================
# 2. BUNDLING LOGIC (No changes needed below)
# ==========================================================
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH) # Insert at 0 to prioritize this path

try:
    import model.data_loader as dl
    # Debug: Print where the module is being loaded from
    print(f"📡 Module Source: {dl.__file__}")
    
    from model.data_loader import get_scaffold_dataloaders, export_to_cloud_bundle
except ImportError as e:
    print(f"❌ Error: {e}")
    print(f"   Current SRC_PATH: {SRC_PATH}")
    print(f"   Searching in: {sys.path[:3]}...")
    sys.exit(1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Mapping placeholder variables to the loader config
    splits_map = {
        "train": SC_TRAIN_IDX,
        "val":   SC_VAL_IDX,
        "test":  SC_TEST_IDX
    }
    
    # Verify paths exist BEFORE starting the heavy loading
    for split, idx_path in splits_map.items():
        if not os.path.exists(idx_path):
            print(f"⚠️  Warning: {split} indices NOT found at {idx_path}")
    
    if not os.path.exists(TARGET_CSV) or not os.path.exists(NOISY_CSV):
        print("❌ Error: TARGET_CSV or NOISY_CSV path is invalid.")
        return

    # Prepare specialized config for the scaffold loader
    splits_config = {}
    for split, idx_path in splits_map.items():
        if os.path.exists(idx_path):
            splits_config[split] = {
                'target': TARGET_CSV,
                'noisy':  NOISY_CSV,
                'indices': idx_path
            }

    if not splits_config:
        print("❌ Error: No valid index files found. Check your paths.")
        return

    print(f"🚀 Loading Data through SIMPLE SCAFFOLD LOADER...")
    loaders = get_scaffold_dataloaders(splits_config, features_dir=FEATURES_DIR)

    # 1. Fit scaler on Training split if available
    scaler = None
    if "train" in loaders:
        print("\n⚖️  Fitting scaler on Training set...")
        scaler = loaders["train"].dataset.get_scaler(mode="delta")
    else:
        print("\n⚠️  No training split found. Exporting without new scaling metadata.")

    # 2. Export Train/Val Bundle
    present_tr_val = [s for s in ["train", "val"] if s in loaders]
    if present_tr_val:
        print(f"📦 Exporting Train/Val Bundle ({', '.join(present_tr_val)})...")
        train_val_loaders = {k: loaders[k] for k in present_tr_val}
        export_to_cloud_bundle(train_val_loaders, os.path.join(OUT_DIR, "scaffold_train_val.pt"), scaler=scaler)

    # 3. Export Test Bundle
    if "test" in loaders:
        print("📦 Exporting Test Bundle...")
        test_loader_dict = {"test": loaders["test"]}
        export_to_cloud_bundle(test_loader_dict, os.path.join(OUT_DIR, "scaffold_test.pt"), scaler=scaler)

    print(f"\n✅ Done! PT files are in: {OUT_DIR}")

if __name__ == "__main__":
    main()
