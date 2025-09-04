# utils/feature_generator.py

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from molfeat.trans.pretrained import PretrainedDGLTransformer

# Imports related to the GCADTI model.
# Paths may need adjustment depending on the GCADTI folder structure.
from GCADTI.run_GCADTI import Model as GcaDtiModel
from GCADTI.DTIDataset import DTIDataset, graph_collate_func


class DtiPredictor:
    """A wrapper class for the GCADTI model to perform Drug-Target Interaction (DTI) prediction."""
    def __init__(self, model_path, prot_list_path, device):
        """
        Initializes the DTI predictor.

        Args:
            model_path (str): Path to the pre-trained GCADTI model weights.
            prot_list_path (str): Path to the CSV file containing the list of target proteins.
            device (str): The computing device to use ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        # Initialize the model wrapper from the GCADTI library.
        self.model_wrapper = GcaDtiModel(modeldir='./dataloader/GCADTI', device=device)
        # Load the pre-trained weights.
        self.model_wrapper.load_pretrained(model_path, device)
        # Load the list of target protein sequences against which predictions will be made.
        self.prot_list = pd.read_csv(prot_list_path, index_col=0)['sequence'].unique()

    def predict_for_smiles(self, smiles: str) -> torch.Tensor:
        """
        Predicts the DTI vector for a single SMILES string against the predefined list of proteins.

        Args:
            smiles (str): The SMILES string of the drug.

        Returns:
            torch.Tensor: A 1D tensor representing the predicted interaction profile.
        """
        # Validate the input SMILES string.
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
            
        # Create a DataFrame pairing the input SMILES with every target protein.
        # This is the required input format for the DTIDataset.
        pred_df = pd.DataFrame({
            'smiles': [smiles] * len(self.prot_list),
            'sequence': self.prot_list,
            'label': [0] * len(self.prot_list) # Dummy labels for inference.
        })

        # Perform prediction without calculating gradients to save memory.
        with torch.no_grad():
            pred_set = DTIDataset(pred_df)
            _, y_pred = self.model_wrapper.predict(pred_set)
        
        # Post-process the predictions.
        pred_df['label'] = y_pred
        # Sort by protein sequence to ensure the output vector has a consistent order.
        df_sorted = pred_df.sort_values(by=['sequence'], ascending=[True])
        # Return the predictions as a PyTorch tensor.
        return torch.tensor(df_sorted['label'].values, dtype=torch.float32)

def generate_drug_features(smiles: str, dti_predictor: DtiPredictor) -> dict:
    """
    Generates all required drug feature vectors from a SMILES string and returns them as a dictionary.

    Args:
        smiles (str): The SMILES string of the drug.
        dti_predictor (DtiPredictor): An initialized DtiPredictor object.

    Returns:
        dict: A dictionary containing the 'morgan', 'gin', and 'dti' feature vectors as PyTorch tensors.
    """
    # Validate the SMILES string before proceeding.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # 1. Generate Morgan Fingerprint
    # This is a widely used, structure-based molecular fingerprint.
    morgan_vec = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    
    # 2. Generate GIN Embedding
    # This uses a pre-trained Graph Isomorphism Network to create a graph-based embedding.
    gin_transformer = PretrainedDGLTransformer(kind='gin_supervised_infomax', dtype=float)
    gin_vec = gin_transformer(mol)
    
    # 3. Generate DTI Vector
    # This predicts the drug's interaction profile across a set of target proteins.
    dti_vec = dti_predictor.predict_for_smiles(smiles)

    # Return all features in a dictionary, converted to PyTorch tensors.
    return {
        'morgan': torch.tensor(morgan_vec, dtype=torch.float32),
        'gin': torch.tensor(gin_vec[0], dtype=torch.float32),
        'dti': dti_vec
    }