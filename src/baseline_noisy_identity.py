import os
import pandas as pd
import numpy as np


def compute_identity_metrics(dataset_path, noisy_path):
    target_df = pd.read_csv(dataset_path)
    noisy_df = pd.read_csv(noisy_path)

    # kies de juiste kolommen
    if "h298" in target_df.columns:
        clean = target_df["h298"].to_numpy(dtype=float)
    elif "enthalpy" in target_df.columns:
        clean = target_df["enthalpy"].to_numpy(dtype=float)
    else:
        raise ValueError("No clean target column found in target file.")

    if "enthalpy" in noisy_df.columns:
        noisy = noisy_df["enthalpy"].to_numpy(dtype=float)
    elif "h298" in noisy_df.columns:
        noisy = noisy_df["h298"].to_numpy(dtype=float)
    else:
        raise ValueError("No noisy column found in noisy file.")

    if len(clean) != len(noisy):
        raise ValueError(f"Length mismatch: clean={len(clean)}, noisy={len(noisy)}")

    diff = noisy - clean
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)

    print("First 10 clean values: ", clean[:10])
    print("First 10 noisy values: ", noisy[:10])
    print("First 10 diffs:        ", diff[:10])

    print("\nIdentity baseline on RAW values")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")

    return mse, rmse, mae


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data", "groupadditivity_h298")

    dataset_path = os.path.join(
        data_path,
        "dataset",
        "fingerprints/groupadditivity_0.004_fingerprints.csv"
    )
    noisy_path = os.path.join(
        data_path,
        "dataset",
        "noise0.4",
        "groupadditivity_0.004_noise0.4.csv"
    )

    compute_identity_metrics(dataset_path, noisy_path)