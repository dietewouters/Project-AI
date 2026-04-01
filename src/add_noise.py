#!/usr/bin/env python3
import numpy as np
import scipy.stats as stats
from scipy.stats import rv_continuous

class cosh_dis(rv_continuous):
    """Cosh distribution for noise injection."""
    def __init__(self, loc):
        super().__init__(a=-loc, b=loc)
        self.scale_p = 2 * np.sinh(loc)
    def _pdf(self, x):
        return np.cosh(x) / self.scale_p

def add_noise(target_vals, smiles_vals, noise_type="normal", noise_scale=0.0):
    """
    Generate noise for a set of values based on SMILES strings and distribution type.
    
    Parameters:
    -----------
    target_vals : array-like
        The original Clean target values.
    smiles_vals : array-like
        SMILES strings (used for conditional noise like 'nitrogen').
    noise_type : str
        Type of noise distribution: 'normal', 'cosh', 'uniform', 'bimodal', 'half_and_half', 'nitrogen'.
    noise_scale : float
        The scale (e.g. std dev) for the noise.
        
    Returns:
    --------
    noise_arr : np.ndarray
        Array of the same shape as target_vals containing the generated noise.
    """
    n_size = len(target_vals)
    if noise_scale <= 0:
        return np.zeros(n_size)

    if noise_type == "normal":
        return np.random.normal(scale=noise_scale, size=n_size)
    
    elif noise_type == "cosh":
        # Value 1.543... corresponds to the parameter used in the original script
        dist = cosh_dis(1.543404638418213)
        return dist.rvs(size=n_size)
    
    elif noise_type == "uniform":
        dist = stats.uniform(-np.sqrt(3), 2 * np.sqrt(3))
        return dist.rvs(size=n_size)
    
    elif noise_type == "bimodal":
        # Combined normal and discrete shift
        return np.random.normal(scale=0.866025, size=n_size) + np.random.choice([-0.5, 0.5], size=n_size)
    
    elif noise_type == "half_and_half":
        # High noise for positive values, low for negative (default 1/10th)
        return np.where(target_vals > 0,
                         np.random.normal(scale=noise_scale, size=n_size),
                         np.random.normal(scale=noise_scale/10.0, size=n_size))
                         
    elif noise_type == "nitrogen":
        # High noise for nitrogen-containing molecules
        has_n = np.array(["N" in str(s).upper() for s in smiles_vals])
        return np.where(has_n,
                         np.random.normal(scale=noise_scale, size=n_size),
                         np.random.normal(scale=noise_scale/10.0, size=n_size))
    
    # Default to normal
    return np.random.normal(scale=noise_scale, size=n_size)


if __name__ == "__main__":
    # Original script functionality preserved for backwards compatibility
    import os
    import csv
    
    DATA_DIR = 'dataset'
    if not os.path.exists(DATA_DIR):
        print(f"Directory '{DATA_DIR}' not found. Skipping script execution.")
    else:
        files = [i for i in os.listdir(DATA_DIR) if i.endswith('.csv')]
        
        # Example: Normal noise
        for noise in [0.01, 0.02, 0.08, 0.2, 0.4, 1]:
            noise_dir = os.path.join(DATA_DIR, f'noise{noise}')
            os.makedirs(noise_dir, exist_ok=True)
            for file in files:
                with open(os.path.join(DATA_DIR, file), 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    data = list(reader)
                    
                smiles = [r[0] for r in data]
                targets = np.array([float(r[1]) for r in data])
                noise_arr = add_noise(targets, smiles, "normal", noise)
                
                new_file = f"{file[:-4]}_noise{noise}.csv"
                with open(os.path.join(noise_dir, new_file), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for i, row in enumerate(data):
                        writer.writerow([row[0], targets[i] + noise_arr[i]])
        print("Script execution finished.")