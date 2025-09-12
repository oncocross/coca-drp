# feature_embedding/gin.py
# This module is responsible for generating GIN (Graph Isomorphism Network)
# embeddings from RDKit molecule objects.

import torch
from rdkit import Chem
from molfeat.trans.pretrained import PretrainedDGLTransformer

# Initialize the transformer once when the module is loaded.
# This avoids re-loading the pre-trained model on every function call, improving efficiency.
GIN_TRANSFORMER = PretrainedDGLTransformer(kind='gin_supervised_infomax', dtype=float)

def generate_gin_embedding(mol: Chem.Mol) -> torch.Tensor:
    """Generates a GIN embedding for a given RDKit molecule."""
    # Validate the input type.
    if not isinstance(mol, Chem.Mol):
        raise TypeError("Input must be a valid RDKit molecule.")
        
    # Generate the embedding using the pre-loaded transformer.
    embedding = GIN_TRANSFORMER(mol)
    
    # Return the embedding as a PyTorch tensor.
    return torch.tensor(embedding[0], dtype=torch.float32)