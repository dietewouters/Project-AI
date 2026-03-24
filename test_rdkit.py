#Je kan deze eens runnen om te zien of rdkit werkt 

from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles("CCO")
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

print("Fingerprint length:", len(fp))