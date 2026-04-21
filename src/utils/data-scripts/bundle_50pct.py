import os
import sys
import torch
import argparse

def find_model_package(search_root):
    """
    Recursively search for the directory that CONTAINS the 'model' folder.
    Returns the parent path that should be added to sys.path.
    """
    print(f"🔍 Searching for 'model' package in: {search_root}...")
    for root, dirs, files in os.walk(search_root):
        # We look for a directory named 'model' 
        if 'model' in dirs:
            model_path = os.path.join(root, 'model')
            # Verify it's actually our package by checking for python files
            try:
                content = os.listdir(model_path)
                if any(f.endswith('.py') for f in content):
                    print(f"✅ Found 'model' package parent at: {root}")
                    return root
            except PermissionError:
                continue
    return None

def main():
    parser = argparse.ArgumentParser(description="Bundle 50% subset into PT files.")
    parser.add_argument("--src_root", type=str, help="Path to project source code (containing 'model' pkg)")
    parser.add_argument("--target_csv", type=str, help="Path to clean groupadditivity.csv")
    parser.add_argument("--noisy_csv", type=str, help="Path to 50% noisy CSV")
    parser.add_argument("--indices_dir", type=str, help="Path to 50% split indices")
    parser.add_argument("--features_dir", type=str, help="Path to .npz feature chunks")
    parser.add_argument("--out_dir", type=str, help="Where to save .pt bundles")
    args = parser.parse_args()

    # --- ENVIRONMENT DETECTION & DEFAULTS ---
    IS_KAGGLE = os.path.exists('/kaggle/input')
    cwd = os.getcwd()

    src_root = args.src_root
    target_csv = args.target_csv
    noisy_csv = args.noisy_csv
    indices_dir = args.indices_dir
    features_dir = args.features_dir
    out_dir = args.out_dir

    if IS_KAGGLE:
        print("Detected Kaggle Environment")
        # Auto-discovery on Kaggle
        if not src_root:
            # Look for common source slugs
            for candidate in ["src-model", "project-ai", "Project-AI"]:
                path = f"/kaggle/input/{candidate}"
                if os.path.exists(path):
                    src_root = path
                    break
            if not src_root: src_root = "/kaggle/input" # fallback search root

        if not target_csv:
            for root, dirs, files in os.walk('/kaggle/input'):
                if "groupadditivity.csv" in files:
                    target_csv = os.path.join(root, "groupadditivity.csv")
                    break
        
        if not features_dir:
            for root, dirs, files in os.walk('/kaggle/input'):
                if "features_chunk_0000.npz" in files:
                    features_dir = root
                    break

        if not noisy_csv: noisy_csv = "/kaggle/working/dataset/groupadditivity_structural_0.5.csv"
        if not indices_dir: indices_dir = "/kaggle/working/indices/scaffold_split_0.5"
        if not out_dir: out_dir = "/kaggle/working/results/cloud_datasets"
    else:
        # Local defaults
        if not src_root: src_root = os.path.join(cwd, 'Project-AI', 'src')
        if not target_csv: target_csv = "data/groupadditivity_h298/dataset/groupadditivity.csv"
        if not noisy_csv: noisy_csv = "data/groupadditivity_h298/dataset/groupadditivity_structural_0.5.csv"
        if not indices_dir: indices_dir = "data/groupadditivity_h298/indices/scaffold_split_0.5"
        if not features_dir: features_dir = "data/processed_features"
        if not out_dir: out_dir = "results/cloud_datasets"

    # --- SETUP SOURCE CODE ---
    pkg_root = find_model_package(src_root)
    if pkg_root:
        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)
            print(f"✅ Source package found at: {pkg_root}")
    else:
        print(f"❌ Error: Could not find 'model' package in {src_root}")
        sys.exit(1)

    try:
        from model.data_loader import get_dataloaders, export_to_cloud_bundle
    except ImportError as e:
        print(f"❌ Error importing model modules: {e}")
        sys.exit(1)

    # --- BUNDLING ---
    print(f"\n🚀 Bundling Config:")
    print(f"  Target:   {target_csv}")
    print(f"  Noisy:    {noisy_csv}")
    print(f"  Indices:  {indices_dir}")
    print(f"  Features: {features_dir}")
    print(f"  Output:   {out_dir}")

    # Verify paths
    for p in [target_csv, noisy_csv, indices_dir, features_dir]:
        if not os.path.exists(p):
            print(f"❌ Error: Path not found -> {p}")
            sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    loaders_config = {}

    for split in splits:
        indices_path = os.path.join(indices_dir, f"{split}_indices.csv")
        with open(indices_path, 'r') as f:
            n_mols = len(f.read().split())
            
        loaders_config[split] = {
            "mode": "precomputed",
            "target_path": target_csv,
            "indices_path": indices_path,
            "total_molecules": n_mols,
            "blocks": [
                {
                    "clean_path": target_csv,
                    "noisy_path": noisy_csv,
                    "indices_path": indices_path
                }
            ]
        }

    print("\n📦 Instantiating DataLoaders...")
    loaders = get_dataloaders(loaders_config, features_dir=features_dir)

    print("\n⚖️  Fitting 'delta' mode scaler to Training set...")
    scaler = loaders["train"].dataset.get_scaler(mode="delta")
    
    print("\n📦 Exporting Train/Val Bundle...")
    tv_loaders = {k: loaders[k] for k in ["train", "val"] if k in loaders}
    export_to_cloud_bundle(tv_loaders, os.path.join(out_dir, "scaffold_0.5_train_val.pt"), scaler=scaler)

    if "test" in loaders:
        print("\n📦 Exporting Test Bundle...")
        export_to_cloud_bundle({"test": loaders["test"]}, os.path.join(out_dir, "scaffold_0.5_test.pt"), scaler=scaler)

    print(f"\n✅ SUCCESS! PT bundles generated in: {out_dir}")

if __name__ == "__main__":
    main()
