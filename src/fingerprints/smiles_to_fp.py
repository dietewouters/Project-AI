import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import scipy.sparse as sp
import multiprocessing as mp
import argparse
from pathlib import Path
from tqdm import tqdm
import os

def generate_fingerprint(smiles, radius=2, n_bits=1024):
    """
    Given a SMILES string, return a Morgan fingerprint.
    Returns a zero-array if the SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.int8)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        # In case of rare RDKit exceptions
        return np.zeros(n_bits, dtype=np.int8)

def process_smiles_chunk(chunk_smiles):
    """
    Processes a list/series of SMILES strings into a sparse matrix.
    """
    fps = [generate_fingerprint(s) for s in chunk_smiles]
    # Convert list of dense arrays to a sparse CSR matrix
    return sp.csr_matrix(fps)

def main():
    parser = argparse.ArgumentParser(description="Convert SMILES to Morgan Fingerprints efficiently.")
    parser.add_argument('--input', type=str, default='data/groupadditivity_h298/dataset/groupadditivity.csv', help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='data/processed_features', help='Directory to save processed chunks')
    parser.add_argument('--chunk_size', type=int, default=100000, help='Number of rows per chunk')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of processes to use (-1 for all available cores)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_jobs = mp.cpu_count() if args.n_jobs == -1 else args.n_jobs
    print(f"Using {n_jobs} cores for multiprocessing.")
    
    # 1. Count rows to initialize tqdm (optional but nice for tracking)
    # Using python wc is fast enough for ~8M rows
    print("Counting rows in dataset...")
    total_rows = sum(1 for _ in open(input_path, 'r')) - 1  # -1 for header
    total_chunks = (total_rows // args.chunk_size) + (1 if total_rows % args.chunk_size != 0 else 0)
    print(f"Total SMILES: {total_rows} | Total Chunks: {total_chunks}")

    # 2. Process in chunks
    csv_reader = pd.read_csv(input_path, usecols=['smiles', 'h298'], chunksize=args.chunk_size)
    
    # Initialize multiprocessing pool
    pool = mp.Pool(n_jobs)
    
    all_targets = []
    chunk_idx = 0
    
    for chunk in tqdm(csv_reader, total=total_chunks, desc="Processing chunks"):
        # We can split the chunk itself into smaller sub-chunks for the pool Map
        # or just map the smiles_to_fp function over the chunk. 
        # Using pool.imap is very efficient here.
        
        # Parallel Fingerprint generation
        chunk_smiles = chunk['smiles'].tolist()
        
        # Pool map preserves order
        fps_dense = pool.map(generate_fingerprint, chunk_smiles)
        
        # Convert to sparse matrix
        fp_sparse_chunk = sp.csr_matrix(fps_dense)
        
        # Save this chunk's features to disk
        out_file = output_dir / f"features_chunk_{chunk_idx:04d}.npz"
        sp.save_npz(out_file, fp_sparse_chunk)
        
        # Collect target values to save together at the end
        all_targets.append(chunk['h298'].values)
        
        chunk_idx += 1
        
    pool.close()
    pool.join()
    
    # 3. Concatenate and save all target values
    targets_array = np.concatenate(all_targets)
    targets_file = output_dir / "targets.npy"
    np.save(targets_file, targets_array)
    
    print(f"\nProcessing complete!")
    print(f"Saved {chunk_idx} feature chunks to {output_dir}")
    print(f"Saved targets to {targets_file}")
    
    print("\nNext steps in your data_loader.py:")
    print("To load the features without running out of memory, you can iterate over the .npz chunks")
    print("or use scipy.sparse.vstack to load them all if you have enough RAM (Sparse uses ~2-4GB for 8M rows).")

if __name__ == '__main__':
    main()
