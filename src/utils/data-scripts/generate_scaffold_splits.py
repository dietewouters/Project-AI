import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- RDKit Installation (Kaggle Support) ---
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    print("RDKit not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit"])
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

# Silence RDKit noise
RDLogger.DisableLog('rdApp.*')

# Configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "dataset", "groupadditivity.csv")

# Detect Kaggle for output redirection
IS_KAGGLE = os.path.exists('/kaggle/input')
OUTPUT_BASE = "/kaggle/working" if IS_KAGGLE else SCRIPT_DIR
INDICES_DIR = os.path.join(OUTPUT_BASE, "indices", "scaffold_split")

os.makedirs(INDICES_DIR, exist_ok=True)

def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return ""

def process_chunk(smiles_list):
    return [get_scaffold(s) for s in smiles_list]

if __name__ == "__main__":
    print(f"🚀 STAGE 1: Generating Scaffold Splits for {INPUT_CSV}")
    
    # 1. Load SMILES only (memory efficient)
    print("Loading SMILES from CSV...")
    df = pd.read_csv(INPUT_CSV, usecols=['smiles'])
    df['original_index'] = df.index
    n_total = len(df)
    print(f"Total molecules: {n_total}")

    # 2. Extract Scaffolds (Multiprocessing)
    num_cores = cpu_count()
    CHUNK_SIZE = 10000
    smiles_vals = df['smiles'].values
    chunks = [smiles_vals[i:i + CHUNK_SIZE] for i in range(0, len(smiles_vals), CHUNK_SIZE)]
    
    print(f"Extracting scaffolds using {num_cores} cores ({len(chunks)} chunks)...")
    with Pool(num_cores) as p:
        chunk_results = list(tqdm(p.imap(process_chunk, chunks), total=len(chunks), desc="Scaffold Extraction"))
    
    print("Flattening results...")
    df['scaffold'] = [s for chunk in chunk_results for s in chunk]
    
    # 3. Perform Scaffold Split
    print("Grouping by scaffold...")
    scaffold_groups = df.groupby('scaffold').groups
    scaffold_list = list(scaffold_groups.keys())
    random.shuffle(scaffold_list)
    
    n_scaffolds = len(scaffold_list)
    print(f"Unique scaffolds found: {n_scaffolds}")
    
    train_cutoff = int(0.8 * n_scaffolds)
    val_cutoff = int(0.9 * n_scaffolds)
    
    train_scaffolds = scaffold_list[:train_cutoff]
    val_scaffolds = scaffold_list[train_cutoff:val_cutoff]
    test_scaffolds = scaffold_list[val_cutoff:]
    
    print("Assigning indices to splits (optimizing access)...")
    original_indices_arr = df['original_index'].values
    
    train_indices = [original_indices_arr[idx] for s in train_scaffolds for idx in scaffold_groups[s]]
    val_indices = [original_indices_arr[idx] for s in val_scaffolds for idx in scaffold_groups[s]]
    test_indices = [original_indices_arr[idx] for s in test_scaffolds for idx in scaffold_groups[s]]
    
    # 4. Save Results
    print(f"Saving splits to {INDICES_DIR}...")
    with open(os.path.join(INDICES_DIR, "train_indices.csv"), "w") as f:
        f.write(" ".join(map(str, sorted(train_indices))))
    with open(os.path.join(INDICES_DIR, "val_indices.csv"), "w") as f:
        f.write(" ".join(map(str, sorted(val_indices))))
    with open(os.path.join(INDICES_DIR, "test_indices.csv"), "w") as f:
        f.write(" ".join(map(str, sorted(test_indices))))
        
    print(f"✅ STAGE 1 COMPLETE!")
    print(f"Counts: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    print(f"Sum matches total: {len(train_indices)+len(val_indices)+len(test_indices) == n_total}")
