import os
from data_loader import MoleculeDataset
import pandas as pd
from fingerprints_for_small_subset import fingerprint_to_string
import numpy as np


def test_fingerprints():
    dataset_path = 'data/groupadditivity_h298/dataset/groupadditivity_8e-05.csv'
    noisy_path = 'data/groupadditivity_h298/dataset/noise0.01/groupadditivity_8e-05_noise0.01.csv'
    indices_path = 'data/groupadditivity_h298/indices/indices_8e-05.csv'
    fingerprint_path = "data/groupadditivity_h298/groupadditivity_8e-05_fingerprints.csv"

    dataset = MoleculeDataset(dataset_path, noisy_path, indices_path)
    df = pd.read_csv(fingerprint_path)

    mapping = {
        "fp": "fingerprint",
        "y_true": "enthalpy"
    }

    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    all_pass = True

    for attr, col in mapping.items():
        dataset_vals = getattr(dataset, attr)
        dataset_vals = dataset_vals.numpy() if hasattr(dataset_vals, 'numpy') else dataset_vals

        # Handle fingerprint conversion
        if attr == "fp":
            dataset_vals = np.array([fingerprint_to_string(fp) for fp in dataset_vals])

        column_vals = df[col].values

        # Compare
        if isinstance(column_vals[0], str):
            match = (column_vals == dataset_vals).all()
            mismatches = np.where(column_vals != dataset_vals)[0]
        else:
            match = np.allclose(column_vals, dataset_vals)
            mismatches = np.where(~np.isclose(column_vals, dataset_vals))[0]

        # Print result
        status = "✓" if match else "✗"
        print(f"\n{status} {attr} == df['{col}']")

        if match:
            print(f"  PASS")
        else:
            print(f"  FAIL ({len(mismatches)} / {len(column_vals)} mismatches)")
            print(f"  First 10:\n")

            for i, idx in enumerate(mismatches[:10], 1):
                print(f"    {i}. Index {idx}:")
                print(f"       df['{col}']: {column_vals[idx]}")
                print(f"       dataset.{attr}: {dataset_vals[idx]}")

        all_pass = all_pass and match

    print("\n" + "=" * 60)
    print(f"Overall Result: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return all_pass





if __name__ == '__main__':
    test_fingerprints()