import os
import pandas as pd

# Detect Environment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_KAGGLE = os.path.exists('/kaggle/input')
OUTPUT_BASE = "/kaggle/working" if IS_KAGGLE else SCRIPT_DIR

PARTS_DIR = os.path.join(OUTPUT_BASE, "dataset", "noise_structural_parts")
FINAL_OUT_DIR = os.path.join(OUTPUT_BASE, "dataset", "noise_structural")
FINAL_OUT_CSV = os.path.join(FINAL_OUT_DIR, "groupadditivity_structural.csv")

os.makedirs(FINAL_OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("🚀 STAGE 3: Merging Quarters...")
    
    parts = []
    for i in range(4):
        part_path = os.path.join(PARTS_DIR, f"noisy_part_{i}.csv")
        if os.path.exists(part_path):
            print(f"Loading {part_path}...")
            parts.append(pd.read_csv(part_path))
        else:
            print(f"❌ Error: {part_path} is missing!")
            
    if len(parts) < 4:
        print("❌ Error: Missing parts. Please run Stage 2 for all 4 quarters.")
    else:
        print("Concatenating and sorting by original index...")
        full_df = pd.concat(parts).sort_values('original_index').reset_index(drop=True)
        
        print(f"Final Count: {len(full_df)}")
        
        # Original row check
        ORIGINAL_CSV = os.path.join(SCRIPT_DIR, "dataset", "groupadditivity.csv")
        original_df = pd.read_csv(ORIGINAL_CSV, usecols=['smiles'])
        
        if len(full_df) == len(original_df):
            print("✅ Count Matches Original.")
            # Sample check for alignment
            print("Running alignment check on 1000 random rows...")
            sample_indices = full_df.sample(1000).index
            merged_smiles = full_df.loc[sample_indices, 'smiles'].values
            orig_smiles = original_df.loc[sample_indices, 'smiles'].values
            
            if (merged_smiles == orig_smiles).all():
                print("✅ Alignment Verified!")
            else:
                print("❌ Alignment ERROR! Indices do not match original row positions.")
        else:
            print(f"❌ Count Mismatch! Expected {len(original_df)}, got {len(full_df)}")

        print(f"Saving final combined CSV to {FINAL_OUT_CSV}...")
        full_df[['smiles', 'h298']].to_csv(FINAL_OUT_CSV, index=False)
        print("✅ STAGE 3 COMPLETE!")
