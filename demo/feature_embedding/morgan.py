# features/morgan.py
# This module is responsible for generating Morgan Fingerprints, a type of
# molecular fingerprint used for representing chemical structures.

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 1024) -> torch.Tensor:
    """Generates a Morgan Fingerprint vector for a given RDKit molecule."""
    # Ensure the input is a valid RDKit molecule object.
    if not isinstance(mol, Chem.Mol):
        raise TypeError("Input must be a valid RDKit molecule.")
    
    # Generate the Morgan Fingerprint as a bit vector using RDKit.
    fp_bits = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    
    # Convert the bit vector to a NumPy array and then to a PyTorch tensor.
    return torch.tensor(np.array(fp_bits), dtype=torch.float32)