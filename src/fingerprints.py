#Start on the model that works with fingerprints 

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np


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


#Function that takes a SMILES string and a noisy enthalpy value, encodes the SMILES into a fingerprint, and concatenates it with the noisy enthalpy to create an input feature vector for the model. The output is a numpy array of floats that can be fed into the MLP model.
def encode_input(smiles:str, noisy_enthalpy:int):
    fingerprint = smiles_to_fingerprint(smiles)
    noisy_enthalpy_array = np.array([noisy_enthalpy], dtype=np.float32)
    input_features = np.concatenate([fingerprint, noisy_enthalpy_array])
    return input_features


def build_mlp_model(input) -> int:
    pass
    #return clean_enthalpy

