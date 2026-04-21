import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import random
from multiprocessing import Pool, cpu_count

# --- RDKit Installation (Kaggle Support) ---
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, AllChem, rdFingerprintGenerator
except ImportError:
    print("RDKit not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit"])
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, AllChem, rdFingerprintGenerator

# Silence RDKit noise
RDLogger.DisableLog('rdApp.*')

# Configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Detect Environment
SCRIPT_DIR = "/kaggle/input/datasets/christiancicirelli/project-data/data/groupadditivity_h298"
IS_KAGGLE = os.path.exists('/kaggle/input')
OUTPUT_BASE = "/kaggle/working" if IS_KAGGLE else SCRIPT_DIR
INPUT_CSV = os.path.join(SCRIPT_DIR, "dataset", "groupadditivity.csv")
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "dataset", "noise_structural_parts")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Morgan Generator
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def get_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        mol_wt = Descriptors.MolWt(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        arom_rings = Descriptors.NumAromaticRings(mol)
        fp_arr = MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
        
        return {
            "MolWt": mol_wt,
            "RotatableBonds": rot_bonds,
            "AromaticRings": arom_rings,
            "FP": fp_arr
        }
    except:
        return None

def process_chunk(smiles_list):
    return [get_features(s) for s in smiles_list]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate structural noise for a specific quarter of the dataset.")
    parser.add_argument("--quarter", type=int, choices=[0, 1, 2, 3], required=True, help="Which quarter to process (0-3)")
    args = parser.parse_args()

    print(f"🚀 STAGE 2: Generating Noise for Quarter {args.quarter}")
    
    # 1. Load Sliced Data
    print(f"Reading {INPUT_CSV}...")
    # Load everything to ensure correct indexing, then slice
    df_full = pd.read_csv(INPUT_CSV)
    df_full['original_index'] = df_full.index
    
    n_total = len(df_full)
    chunk_size = (n_total + 3) // 4
    start_idx = args.quarter * chunk_size
    end_idx = min((args.quarter + 1) * chunk_size, n_total)
    
    df = df_full.iloc[start_idx:end_idx].reset_index(drop=True)
    print(f"Processing slice: {start_idx} to {end_idx} ({len(df)} molecules)")
    del df_full # Free memory

    # 2. RDKit Feature Extraction (Multiprocessing)
    num_cores = cpu_count()
    CHUNK_SIZE = 10000
    smiles_vals = df['smiles'].values
    chunks = [smiles_vals[i:i + CHUNK_SIZE] for i in range(0, len(smiles_vals), CHUNK_SIZE)]
    
    print(f"Extracting features using {num_cores} cores ({len(chunks)} chunks)...")
    with Pool(num_cores) as p:
        chunk_results = list(tqdm(p.imap(process_chunk, chunks), total=len(chunks), desc=f"Q{args.quarter} Features"))
    
    print("Flattening results...")
    results = [res for chunk in chunk_results for res in chunk]
    
    # Cleanup rows with failed RDKit parsing
    valid_mask = [res is not None for res in results]
    df = df[valid_mask].reset_index(drop=True)
    results = [res for res in results if res is not None]
    print(f"Molecules successfully processed: {len(df)}")

    # Convert to arrays
    fp_matrix = np.stack([res['FP'] for res in results])
    descriptors = pd.DataFrame([{
        "MolWt": res['MolWt'],
        "RotatableBonds": res['RotatableBonds'],
        "AromaticRings": res['AromaticRings']
    } for res in results])

    # 3. Noise Model
    # Re-using the exact same logic as before
    class NoiseMixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1024 + 3, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.1)
                    nn.init.constant_(m.bias, 0)
        def forward(self, x):
            return self.net(x)

    mixer = NoiseMixer()
    # Normalize descriptors (Local normalization per quarter - with 2M molecules, this is stable)
    desc_norm = (descriptors - descriptors.mean()) / (descriptors.std() + 1e-6)
    input_tensor = torch.cat([
        torch.from_numpy(fp_matrix),
        torch.from_numpy(desc_norm.values.astype(np.float32))
    ], dim=1)

    print("Generating structural bias...")
    with torch.no_grad():
        structural_bias = mixer(input_tensor).numpy().flatten()
    
    structural_bias = structural_bias * (0.5 / (np.std(structural_bias) + 1e-6))
    
    complexity = desc_norm.mean(axis=1).values
    complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-6)
    per_molecule_sigma = 0.5 + 1.0 * complexity
    
    pure_noise = np.random.normal(0, per_molecule_sigma)
    total_delta = structural_bias + pure_noise
    
    df['h298_noisy'] = df['h298'] + total_delta
    df['delta'] = total_delta

    # 4. Save Quarter
    out_file = os.path.join(OUTPUT_DIR, f"noisy_part_{args.quarter}.csv")
    print(f"Saving quarter to {out_file}...")
    # Keep original_index for perfect merging later
    df[['original_index', 'smiles', 'h298_noisy']].rename(columns={'h298_noisy': 'h298'}).to_csv(out_file, index=False)
    
    print(f"✅ STAGE 2 (Q{args.quarter}) COMPLETE!")
