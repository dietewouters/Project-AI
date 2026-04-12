import pandas as pd
import numpy as np
from fingerprints import smiles_to_fingerprint


def fingerprint_to_string(fp: np.ndarray) -> str:
    return "".join(map(str, fp.astype(int)))


def convert_csv_to_fingerprint_csv(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    if "smiles" not in df.columns or "h298" not in df.columns:
        raise ValueError("CSV does not contain the correct columns. Required: 'smiles' and 'h298'.")

    fingerprints = []
    enthalpies = []

    for _, row in df.iterrows():
        smiles = row["smiles"]
        h298 = row["h298"]

        try:
            fp = smiles_to_fingerprint(smiles, length=1024)
            fp_str = fingerprint_to_string(fp)   #make string
            fingerprints.append(fp_str)
            enthalpies.append(h298)
        except Exception as e:
            print(f"Error '{smiles}': {e}")

    out_df = pd.DataFrame({
        "fingerprint": fingerprints,
        "enthalpy": enthalpies
    })

    # save
    out_df.to_csv(output_path, index=False)
    print(f"New csv file saved: {output_path}")
    
#input_file = "data/groupadditivity_h298/dataset/groupadditivity_8e-05.csv"
#output_file = "data/groupadditivity_h298/groupadditivity_8e-05_fingerprints.csv"
#
##input_file = "data/groupadditivity_h298/dataset/noise0.4/groupadditivity_0.004_noise0.4.csv"
##output_file = "data/groupadditivity_h298/dataset/fingerprints/noise0.4/groupadditivity_0.004_noise0.4_fingerprints.csv"
#
#convert_csv_to_fingerprint_csv(input_file, output_file)