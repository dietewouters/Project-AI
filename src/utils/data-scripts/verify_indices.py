import os
import pandas as pd
import argparse

# Detect Environment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_KAGGLE = os.path.exists('/kaggle/input')
OUTPUT_BASE = "/kaggle/working" if IS_KAGGLE else SCRIPT_DIR

FINAL_CSV = os.path.join(OUTPUT_BASE, "dataset", "noise_structural", "groupadditivity_structural.csv")
ORIGINAL_CSV = os.path.join(SCRIPT_DIR, "dataset", "groupadditivity.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify that new dataset matches old indices.")
    parser.add_argument("--index_file", type=str, required=True, help="Path to a legacy indices.csv file (e.g. from the 25%% run)")
    args = parser.parse_args()
    
    if not os.path.exists(FINAL_CSV):
        print(f"❌ Error: Final dataset not found at {FINAL_CSV}")
        exit(1)
        
    print(f"🚀 STAGE 4: Verifying Alignment for {os.path.basename(args.index_file)}")
    
    # 1. Load Indices
    with open(args.index_file, "r") as f:
        legacy_indices = [int(i) for i in f.read().split()]
    
    print(f"Loaded {len(legacy_indices)} indices to verify.")
    
    # 2. Load Final Generated Dataset
    print("Loading final dataset...")
    final_df = pd.read_csv(FINAL_CSV)
    
    # 3. Load Original Dataset for Ground Truth
    print("Loading original source for ground truth...")
    orig_df = pd.read_csv(ORIGINAL_CSV)
    
    # 4. Spot Check
    print("Verifying SMILES mapping...")
    
    mismatches = 0
    total_checks = min(1000, len(legacy_indices))
    sample_indices = random.sample(legacy_indices, total_checks) if len(legacy_indices) > 1000 else legacy_indices
    
    import random
    
    for idx in sample_indices:
        if idx >= len(final_df):
            print(f"❌ Index {idx} out of range!")
            mismatches += 1
            continue
            
        final_smiles = final_df.iloc[idx]['smiles']
        orig_smiles = orig_df.iloc[idx]['smiles']
        
        if final_smiles != orig_smiles:
            print(f"❌ Mismatch at index {idx}!")
            print(f"   Original: {orig_smiles}")
            print(f"   Generated: {final_smiles}")
            mismatches += 1
            
    if mismatches == 0:
        print(f"✅ VERIFICATION PASSED: All {total_checks} checked indices match perfectly.")
        print(f"Relationship: New Molecule at row 'i' == Original Molecule at row 'i'.")
    else:
        print(f"❌ VERIFICATION FAILED: {mismatches} mismatches found.")
