import torch 
from model.data_loader import get_cloud_dataloaders
from scipy.stats import pearsonr
import numpy as np


loader = get_cloud_dataloaders("results/cloud_datasets/bundle_1776377287.pt")
train_loader = loader['train']
ds = train_loader.dataset

# Flattening tensors to ensure 1D arrays for pearsonr
y_true = ds.y_true.numpy().flatten()
noisy = ds.noisy.numpy().flatten()
delta = noisy - y_true
X = ds.fp.numpy()

# 1. Correlation between Target and Noisy input
r_base, _ = pearsonr(y_true, noisy)
print(f"Correlation (Target vs Noisy): {r_base:.4f}")

# 2. Correlation between Fingerprint Bits and Delta
# Using a loop as requested, but with a guard for bits that have zero variance
print("Calculating bit-wise correlations with Delta...")
correlations = np.array([
    pearsonr(X[:, i], delta)[0] if np.std(X[:, i]) > 0 else 0 
    for i in range(X.shape[1])
])

print(f"Max bit correlation with delta: {np.max(np.abs(correlations)):.4f}")
print(f"Mean abs correlation: {np.abs(correlations).mean():.4f}")