#Start on the model that works with fingerprints 

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import pandas as pd


#Function that transform a given smiles string into a ECFP fingerprint of specified radius and length. The output is a numpy array of floats
#Parameters: default values are common
def smiles_to_fingerprint(smiles: str, radius: int = 2, length: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=length)
    fingerprint = generator.GetFingerprint(mol)

    arr = np.zeros((length,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr

#Help function to put fingerprint in string format
def fingerprint_to_string(fp: np.ndarray) -> str:
    return "".join(map(str, fp.astype(int)))

#Function that converts the complete csv into a csv containing fingerprints with enthalpy value. For this csv the input for the model is already encoded. Model can read line by line
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
            fp = smiles_to_fingerprint(smiles)  
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
  
    
def build_mlp_model(input) -> int:
    pass
    #return clean_enthalpy


#Function that takes a SMILES string and a noisy enthalpy value, encodes the SMILES into a fingerprint, and concatenates it with the noisy enthalpy to create an input feature vector for the model. The output is a numpy array of floats that can be fed into the MLP model.
#def encode_input(smiles:str, noisy_enthalpy:int):
 #   fingerprint = smiles_to_fingerprint(smiles)
  #  noisy_enthalpy_array = np.array([noisy_enthalpy], dtype=np.float32)
   # input_features = np.concatenate([fingerprint, noisy_enthalpy_array])
    #return input_features



