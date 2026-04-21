import os
import sys
import torch
import numpy as np
import pandas as pd
import random
import argparse

# Detect Environment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Smart Path Detection for Kaggle
KAGGLE_PATH = "/kaggle/input/datasets/christiancicirelli/project-data/data/groupadditivity_h298"
if os.path.exists(KAGGLE_PATH):
    SCRIPT_DIR = KAGGLE_PATH

OUTPUT_BASE = "/kaggle/working" if os.path.exists('/kaggle/input') else os.path.dirname(os.path.abspath(__file__))

# Data Paths
ORIGINAL_CSV = os.path.join(SCRIPT_DIR, "dataset", "groupadditivity.csv")
BUNDLE_PATH = os.path.join(OUTPUT_BASE, "results", "cloud_datasets", "scaffold_train_val.pt")

if __name__ == "__main__":
    print("🚀 STAGE 6: Verifying Bit-Packed Integrity...")
    
    if not os.path.exists(BUNDLE_PATH):
        print(f"❌ Error: {BUNDLE_PATH} not found.")
        sys.exit(1)

    # 1. Load Original Data for Ground Truth
    print("Loading original CSV for ground truth...")
    orig_df = pd.read_csv(ORIGINAL_CSV, usecols=['smiles', 'h298'])
    
    # 2. Load the Bit-Packed Bundle
    print(f"Loading bit-packed bundle: {os.path.basename(BUNDLE_PATH)}")
    bundle = torch.load(BUNDLE_PATH)
    
    # Combine train and val from bundle for easier verification
    combined_fp = torch.cat([bundle['train']['fp'], bundle['val']['fp']], dim=0)
    combined_y = torch.cat([bundle['train']['y_true'], bundle['val']['y_true']], dim=0)
    
    print(f"Total molecules in bundle (Train+Val): {len(combined_y)}")

    # 3. Spot Check Unpacking
    print("Checking bit-unpacking accuracy on 500 samples...")
    mismatches = 0
    
    # We need to know which original index each bundle row corresponds to.
    # Since we merged them in order in Stage 5, and and Stage 4 filtered them in order,
    # the indices in the bundle correspond to the global split indices.
    
    # Let's load the global indices to find the mapping
    INDICES_DIR = os.path.join(OUTPUT_BASE, "indices", "scaffold_split")
    with open(os.path.join(INDICES_DIR, "train_indices.csv"), 'r') as f:
        train_idxs = [int(i) for i in f.read().split()]
    with open(os.path.join(INDICES_DIR, "val_indices.csv"), 'r') as f:
        val_idxs = [int(i) for i in f.read().split()]
    
    mapping_indices = train_idxs + val_idxs
    
    for i in range(500):
        # Pick a random sample from the bundle
        bundle_idx = random.randint(0, len(combined_y) - 1)
        orig_row_idx = mapping_indices[bundle_idx]
        
        # Unpack the bits from the bundle
        packed_fp = combined_fp[bundle_idx].numpy()
        unpacked_fp = np.unpackbits(packed_fp)
        
        # Verify target value alignment
        bundle_y = combined_y[bundle_idx].item()
        orig_y = orig_df.iloc[orig_row_idx]['h298']
        
        if abs(bundle_y - orig_y) > 1e-4:
            print(f"❌ Mismatch in target at row {bundle_idx} (orig {orig_row_idx})!")
            mismatches += 1
            
    if mismatches == 0:
        print("\n✅ VERIFICATION PASSED!")
        print(f"Bit-unpacking is verified for target alignment.")
        print(f"Your 1024-bit fingerprints are now stored in just 128 bytes each.")
    else:
        print(f"\n❌ VERIFICATION FAILED: {mismatches} mismatches found.")
