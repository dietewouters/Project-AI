import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    # --- DEFAULT PATH DETECTION (overridable via args) ---
    LOCAL_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Fallback for Kaggle dataset structure
    KAGGLE_PATH = "/kaggle/input/datasets/christiancicirelli/project-data/data/groupadditivity_h298"
    DEFAULT_DATA_ROOT = KAGGLE_PATH if os.path.exists(KAGGLE_PATH) else LOCAL_SCRIPT_DIR
    
    IS_KAGGLE = os.path.exists('/kaggle/input')
    DEFAULT_OUTPUT_BASE = "/kaggle/working" if IS_KAGGLE else DEFAULT_DATA_ROOT

    # --- CLI ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Generate bit-packed .pt part for a quarter.")
    
    # Required
    parser.add_argument("--quarter", type=int, choices=[0, 1, 2, 3], required=True, help="Which quarter to process (0-3)")
    
    # Optional Path Overrides
    parser.add_argument("--target_csv", type=str, 
                        default=os.path.join(DEFAULT_DATA_ROOT, "dataset", "groupadditivity.csv"),
                        help="Path to clean groupadditivity.csv")
    parser.add_argument("--noisy_dir", type=str, 
                        default=os.path.join(DEFAULT_OUTPUT_BASE, "dataset", "noise_structural_parts"),
                        help="Folder containing noisy_part_*.csv")
    parser.add_argument("--indices_dir", type=str, 
                        default=os.path.join(DEFAULT_OUTPUT_BASE, "indices", "scaffold_split"),
                        help="Folder containing split indices")
    parser.add_argument("--features_dir", type=str,
                        default=os.path.join(DEFAULT_DATA_ROOT, "dataset", "processed_features"),
                        help="Folder containing .npz feature chunks")
    parser.add_argument("--src_root", type=str,
                        default="/kaggle/input/datasets/christiancicirelli/src-model",
                        help="Folder containing the 'model' package")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(DEFAULT_OUTPUT_BASE, "dataset", "pt_parts"),
                        help="Where to save the quarterly .pt files")

    args = parser.parse_args()

    # --- RESOLVE PATHS ---
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Add source to path for model.data_loader
    if os.path.exists(args.src_root):
        sys.path.insert(0, args.src_root)
        # Handle recursive search for the model package
        for root, dirs, files in os.walk(args.src_root):
             if 'model' in dirs:
                sys.path.insert(0, root)
                print(f"✅ Added {root} to sys.path")
                break

    try:
        from model.data_loader import SimpleScaffoldDataset, export_to_cloud_bundle
    except ImportError:
        print(f"❌ Error: Could not import model.data_loader from {args.src_root}. Check your --src_root.")
        sys.exit(1)

    print(f"🚀 STAGE 4: Generating Bundles for Quarter {args.quarter}")
    
    # 1. Determine Quarter Range
    total_mols = 7906814 
    chunk_size = (total_mols + 3) // 4
    start_row = args.quarter * chunk_size
    end_row = min((args.quarter + 1) * chunk_size, total_mols)
    quarter_range = set(range(start_row, end_row))
    
    print(f"   Quarter {args.quarter} covers rows {start_row} to {end_row}")

    # 2. Check Input Files
    noisy_csv = os.path.join(args.noisy_dir, f"noisy_part_{args.quarter}.csv")
    if not os.path.exists(noisy_csv):
        print(f"❌ Error: Noisy partial file not found at {noisy_csv}")
        sys.exit(1)

    splits = ["train", "val", "test"]
    quarter_splits_config = {}

    for split in splits:
        idx_path = os.path.join(args.indices_dir, f"{split}_indices.csv")
        if not os.path.exists(idx_path):
            print(f"⚠️  Warning: {split} index file missing at {idx_path}. Skipping.")
            continue

        with open(idx_path, 'r') as f:
            global_indices = [int(i) for i in f.read().split()]
        
        # Filter for molecules belonging to this physical quarter
        q_indices = [i for i in global_indices if i in quarter_range]
        
        if q_indices:
            # Create a temporary local indices file
            temp_idx_path = os.path.join(args.out_dir, f"temp_{split}_q{args.quarter}.csv")
            with open(temp_idx_path, "w") as f:
                f.write(" ".join(map(str, q_indices)))
            
            quarter_splits_config[split] = {
                'target': args.target_csv,
                'noisy': noisy_csv,
                'indices': temp_idx_path,
                'index_offset': start_row
            }
            print(f"   - {split.upper()}: {len(q_indices)} molecules in this quarter.")

    # 3. Create loader and pack bits
    from torch.utils.data import DataLoader
    
    loaders = {}
    for split, paths in quarter_splits_config.items():
        print(f"\n📦 Processing {split.upper()} for Quarter {args.quarter}...")
        dataset = SimpleScaffoldDataset(
            target_csv_path=paths['target'],
            noisy_csv_path=paths['noisy'],
            indices_path=paths['indices'],
            features_dir=args.features_dir,
            index_offset=paths['index_offset']
        )
        loaders[split] = DataLoader(dataset, batch_size=1, num_workers=0)

    # 4. Save the quarterly bundle
    out_pt = os.path.join(args.out_dir, f"bundle_q{args.quarter}.pt")
    print(f"\n💾 Saving bit-packed quarterly bundle to {out_pt}...")
    export_to_cloud_bundle(loaders, out_pt)
    
    print(f"✅ STAGE 4 (Quarter {args.quarter}) COMPLETE!")
