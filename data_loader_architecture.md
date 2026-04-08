# Data Loader Architecture

This document briefly explains the architecture of `MoleculeDataset` and how the various data components interact natively in PyTorch to train our models cleanly.

## Overview
Because storing the fingerprints for 8 million molecules in raw text (`.csv`) causes catastrophic memory/storage bloating, we fully decoupled the structural Fingerprints from the target variables. 

The data loading pipeline now bridges these concepts efficiently through `indices`.

### 1. `processed_features/`
This directory holds the globally computed Morgan Fingerprints for the **entire 8 million row dataset** saved as mathematical sparse matrices (`.npz`). Crucially, the row index of this sparse matrix maps perfectly to the entire dataset (Index `42` is exactly Molecule `42`).

### 2. `indices/`
These files (e.g., `indices_0.004.csv`) are simply a space-separated string of integers. They tell the model exactly which specific global index positions map functionally to a given subset.

### 3. The Target / Noisy Datasets (`dataset/`)
The smaller CSV files (like `groupadditivity_0.004.csv`) only hold the SMILES string and the numeric variables (like `h298` and noisy `input`). They no longer need to contain the massively redundant text fingerprints.

## How `MoleculeDataset` works
When you initialize `MoleculeDataset` inside `data_loader.py`, here is exactly what happens:

1. **Load Numbers:** It opens the specific slice datasets (`target_df` and `noisy_df`) to read the target and noisy features.
2. **Find Indices:** It matches the current CSV name (e.g., `0.004`) to find the matching `indices_0.004.csv` file. 
3. **Load Sparse Vectors:** It quickly loads all the `.npz` chunks from `processed_features/` into one big sparse memory block.
4. **Sub-Select Data:** It uses the numbers from the `indices` file to slice out *only* the specific sparse fingerprints needed for this subset iteration.
5. **Convert:** It converts that specific fingerprint matrix slice back into a dense `torch.tensor` perfectly aligned with your `target` and `noisy` variable dataframes ready for GPU training!
