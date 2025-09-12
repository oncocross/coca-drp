# feature_embedding/dti.py
# Contains the DtiPredictor class, a wrapper for the GCADTI model
# to predict Drug-Target Interaction (DTI) profiles.

import torch
import numpy as np
import pandas as pd
from rdkit import Chem

# Import modules from the GCADTI library.
from GCADTI.run_GCADTI import Model as GcaDtiModel
from GCADTI.DTIDataset import DTIDataset


class DtiPredictor:
    """A wrapper class for the GCADTI model to perform DTI prediction."""
    def __init__(self, model_path, prot_list_path, device):
        """
        Initializes the DTI predictor.

        Args:
            model_path (str): Path to the pre-trained GCADTI model weights.
            prot_list_path (str): Path to the CSV file with target protein sequences.
            device (str): The computing device to use ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        # Initialize the model wrapper from the GCADTI library.
        self.model_wrapper = GcaDtiModel(modeldir='./dataloader/GCADTI', device=device)
        # Load the pre-trained weights.
        self.model_wrapper.load_pretrained(model_path, device)
        # Load the list of target protein sequences for prediction.
        self.prot_list = pd.read_csv(prot_list_path, index_col=0)['sequence'].unique()

    def predict_for_smiles(self, smiles: str) -> torch.Tensor:
        """
        Predicts the DTI vector for a single SMILES string against the predefined list of proteins.

        Args:
            smiles (str): The SMILES string of the drug.

        Returns:
            torch.Tensor: A 1D tensor representing the predicted interaction profile.
        """
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
            
        # Create a DataFrame pairing the input SMILES with every target protein,
        # which is the required input format for the DTIDataset.
        pred_df = pd.DataFrame({
            'smiles': [smiles] * len(self.prot_list),
            'sequence': self.prot_list,
            'label': [0] * len(self.prot_list) # Dummy labels are used for inference.
        })

        # Perform prediction without gradient calculation for efficiency.
        with torch.no_grad():
            pred_set = DTIDataset(pred_df)
            _, y_pred = self.model_wrapper.predict(pred_set)
        
        # Post-process the predictions.
        pred_df['label'] = y_pred
        # Sort by protein sequence to ensure a consistent output vector order.
        df_sorted = pred_df.sort_values(by=['sequence'], ascending=[True])
        
        # Return the DTI profile as a PyTorch tensor.
        return torch.tensor(df_sorted['label'].values, dtype=torch.float32)