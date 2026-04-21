import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# Configuration
SEED = 42

def process_chunk(smiles_list):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    results = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                results.append(None)
                continue
            results.append({
                "MolWt": Descriptors.MolWt(mol),
                "RotatableBonds": Descriptors.NumRotatableBonds(mol),
                "AromaticRings": Descriptors.NumAromaticRings(mol),
                "FP": gen.GetFingerprintAsNumPy(mol).astype(np.float32).tolist()
            })
        except:
            results.append(None)
    return results

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

def main():
    parser = argparse.ArgumentParser(description="Generate 50% structural noise subset.")
    parser.add_argument("--input_csv", type=str, help="Path to groupadditivity.csv")
    parser.add_argument("--indices_dir", type=str, help="Path to existing scaffold_split indices")
    parser.add_argument("--out_indices_dir", type=str, help="Where to save 50% indices")
    parser.add_argument("--out_csv", type=str, help="Where to save 50% noisy CSV")
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size for noise generation")
    args = parser.parse_args()

    # --- ENVIRONMENT DETECTION & DEFAULTS ---
    IS_KAGGLE = os.path.exists('/kaggle/input')
    cwd = os.getcwd()

    input_csv = args.input_csv
    indices_dir = args.indices_dir
    out_indices_dir = args.out_indices_dir
    out_csv = args.out_csv

    if IS_KAGGLE:
        print("Detected Kaggle Environment")
        # Find datasets automatically if not provided
        if not input_csv:
            for root, dirs, files in os.walk('/kaggle/input'):
                if "groupadditivity.csv" in files:
                    input_csv = os.path.join(root, "groupadditivity.csv")
                    break
        if not indices_dir:
            for root, dirs, files in os.walk('/kaggle/input'):
                if "train_indices.csv" in files and "scaffold_split" in root:
                    indices_dir = root
                    break
        
        # Working directory defaults
        if not out_indices_dir: out_indices_dir = "/kaggle/working/indices/scaffold_split_0.5"
        if not out_csv: out_csv = "/kaggle/working/dataset/groupadditivity_structural_0.5.csv"
    else:
        # Local defaults
        if not input_csv: input_csv = "data/groupadditivity_h298/dataset/groupadditivity.csv"
        if not indices_dir: indices_dir = "data/groupadditivity_h298/indices/scaffold_split"
        if not out_indices_dir: out_indices_dir = "data/groupadditivity_h298/indices/scaffold_split_0.5"
        if not out_csv: out_csv = "data/groupadditivity_h298/dataset/groupadditivity_structural_0.5.csv"

    # Seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Input CSV:   {input_csv}")
    print(f"Indices Dir: {indices_dir}")
    print(f"Output Dir:  {out_indices_dir}")
    print(f"Output CSV:  {out_csv}")

    os.makedirs(out_indices_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # 1. SUBSAMPLE INDICES
    print("\n🚀 Subsampling indices (50% per split)...")
    smiles_indices = []
    for split in ["train", "val", "test"]:
        path = os.path.join(indices_dir, f"{split}_indices.csv")
        if not os.path.exists(path):
            print(f"⚠️ Error: Missing {split} indices at {path}")
            continue
        
        with open(path, 'r') as f:
            indices = [int(i) for i in f.read().split()]
        
        kept = sorted(random.sample(indices, len(indices) // 2))
        with open(os.path.join(out_indices_dir, f"{split}_indices.csv"), 'w') as f:
            f.write(" ".join(map(str, kept)))
        
        print(f"  [+] {split}: {len(indices)} -> {len(kept)}")
        smiles_indices.extend(kept)

    smiles_indices = sorted(smiles_indices)

    # 2. LOAD DATA
    print(f"\nReading clean data ({len(smiles_indices)} molecules)...")
    df_full = pd.read_csv(input_csv, usecols=['smiles', 'h298'])
    df = df_full.iloc[smiles_indices].reset_index(drop=True)
    del df_full

    # 3. EXTRACT DESCRIPTORS
    print("Extracting descriptors...")
    num_cores = cpu_count()
    smiles_list = df['smiles'].tolist()
    
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    def get_descriptors(s):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None: return None
            return {
                "MolWt": Descriptors.MolWt(mol),
                "RotatableBonds": Descriptors.NumRotatableBonds(mol),
                "AromaticRings": Descriptors.NumAromaticRings(mol)
            }
        except:
            return None

    with Pool(num_cores) as p:
        desc_results = list(tqdm(p.imap(get_descriptors, smiles_list), total=len(smiles_list), desc="Descriptors"))
    
    valid_mask = [res is not None for res in desc_results]
    df = df[valid_mask].reset_index(drop=True)
    desc_results = [res for res in desc_results if res is not None]
    smiles_list = df['smiles'].tolist()

    descriptors = pd.DataFrame(desc_results)
    desc_norm = (descriptors - descriptors.mean()) / (descriptors.std() + 1e-6)
    desc_tensor_full = torch.tensor(desc_norm.values.tolist(), dtype=torch.float32)

    # 4. BATCHED BIAS GENERATION
    print("\nGenerating structural bias in batches...")
    mixer = NoiseMixer()
    structural_bias = np.zeros(len(df))
    
    for i in range(0, len(smiles_list), args.batch_size):
        end = min(i + args.batch_size, len(smiles_list))
        batch_smiles = smiles_list[i : end]
        batch_desc = desc_tensor_full[i : end]
        
        sub_chunks = [batch_smiles[j:j+1000] for j in range(0, len(batch_smiles), 1000)]
        with Pool(num_cores) as p:
            batch_fps_nested = p.map(process_chunk, sub_chunks)
        
        batch_fps_list = [fp for sub in batch_fps_nested for fp in sub]
        batch_fps_tensor = torch.tensor(batch_fps_list, dtype=torch.float32)

        X = torch.cat([batch_fps_tensor, batch_desc], dim=1)
        with torch.no_grad():
            bias_list = mixer(X).flatten().tolist()
        
        structural_bias[i : end] = bias_list
        print(f"  Processed {end} / {len(smiles_list)}")

    # 5. NOISE CALCULATION & SAVE
    print("\nFinalizing noise...")
    bias_std = np.std(structural_bias) + 1e-6
    structural_bias = structural_bias * (0.5 / bias_std)

    complexity = desc_norm.mean(axis=1).values
    complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-6)
    per_molecule_sigma = 0.5 + 1.0 * complexity

    pure_noise = np.random.normal(0, per_molecule_sigma)
    total_delta = structural_bias + pure_noise

    df['h298_noisy'] = df['h298'] + total_delta

    print(f"Saving to {out_csv}...")
    df[['smiles', 'h298_noisy']].rename(columns={'h298_noisy': 'h298'}).to_csv(out_csv, index=False)

    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
