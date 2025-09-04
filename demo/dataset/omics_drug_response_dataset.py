# dataset/dataset.py

import torch
import pandas as pd
from torch.utils.data import Dataset


class OmicsDrugResponseDataset(Dataset):
    """
    A PyTorch Dataset for handling the full drug response data, including omics,
    drug features, and IC50 values. Used primarily for training.
    """
    def __init__(self, weighted_ic50, crispr, pro, exp, meth, cid2morgan, cid2gin, cid2desc, cid2dti):
        """
        Initializes the dataset by aligning and preprocessing all input data.

        Args:
            weighted_ic50 (pd.DataFrame): DataFrame of IC50 values (rows=samples, cols=drug CIDs).
            crispr, pro, exp, meth (pd.DataFrame): DataFrames for various omics features (rows=samples).
            cid2morgan (dict): Dictionary mapping drug CIDs to their Morgan fingerprint.
            cid2gin (dict): Dictionary mapping drug CIDs to their GIN embedding.
            cid2desc (dict): Dictionary mapping drug CIDs to their descriptor vectors.
            cid2dti (dict): Dictionary mapping drug CIDs to their DTI vectors.
        """
        # Find the common cell lines (index) across all omics and IC50 dataframes to ensure alignment.
        common_index = weighted_ic50.index \
                           .intersection(crispr.index) \
                           .intersection(pro.index) \
                           .intersection(exp.index) \
                           .intersection(meth.index)
        
        # Filter all dataframes to keep only the common cell lines.
        self.weighted_ic50 = weighted_ic50.loc[common_index]
        self.omics = {
            'crispr': crispr.loc[common_index],
            'pro': pro.loc[common_index],
            'exp': exp.loc[common_index],
            'meth': meth.loc[common_index]
        }

        # Get the final lists of cell lines and drugs.
        self.cell_lines = list(self.weighted_ic50.index)
        self.drugs = list(self.weighted_ic50.columns)

        # Create a list of all possible (cell line, drug) pairs. This defines the dataset size.
        self.pairs = [
            (cell, drug)
            for cell in self.cell_lines
            for drug in self.drugs
        ]

        # Convert omics dataframes to PyTorch tensors for efficiency.
        self.omics_tensors = {
            k: torch.tensor(v.values, dtype=torch.float32)
            for k, v in self.omics.items()
        }
        # Create a mapping from cell line name to its index for quick lookups.
        self.cell_line_to_idx = {cl: i for i, cl in enumerate(self.cell_lines)}
        
        # Pre-process and store all drug features in a nested dictionary for quick access.
        self.drug_features = {
            cid: {
                'morgan': torch.tensor(cid2morgan[cid], dtype=torch.float32) if cid in cid2morgan else None,
                'gin': torch.tensor(cid2gin[cid], dtype=torch.float32) if cid in cid2gin else None,
                'desc': torch.tensor(cid2desc[cid], dtype=torch.float32) if cid in cid2desc else None,
                'dti': torch.tensor(cid2dti[cid], dtype=torch.float32) if cid in cid2dti else None,
            }
            for cid in self.drugs
        }

    def __len__(self):
        """Returns the total number of samples in the dataset (i.e., the number of cell-drug pairs)."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (a cell-drug pair and its associated data) from the dataset.
        
        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing omics features, drug features, cell line name, drug CID, and the target IC50 value.
        """
        # Get the cell line and drug for the given index.
        cell_line, drug_cid = self.pairs[idx]
        # Find the numerical index for the cell line.
        cell_idx = self.cell_line_to_idx[cell_line]
        # Retrieve the pre-computed omics tensors for this cell line.
        omics_features = {k: v[cell_idx] for k, v in self.omics_tensors.items()}
        # Retrieve the pre-computed drug features for this drug.
        drug_feature = self.drug_features[drug_cid]
        # Retrieve the target IC50 value and convert it to a tensor.
        target_ic50 = torch.tensor(
            self.weighted_ic50.loc[cell_line, drug_cid],
            dtype=torch.float32
        )
        return omics_features, drug_feature, cell_line, drug_cid, target_ic50


# ==============================================================================
# The simplified class below is used during the prediction (inference) stage.
# ==============================================================================
class OmicsDrugResponseInferenceDataset(Dataset):
    """
    A simplified Dataset for inference.
    It combines the features of a single, fixed drug with the omics data of all available cell lines.
    """
    def __init__(self, cell_lines, omics_features_per_cell, drug_features):
        self.cell_lines = cell_lines
        self.omics_features_per_cell = omics_features_per_cell
        self.drug_features = drug_features # Features for ONE drug.

    def __len__(self):
        """The length is the number of cell lines, as we make one prediction per cell line."""
        return len(self.cell_lines)

    def __getitem__(self, idx):
        """
        Retrieves a sample for inference. Each sample consists of the omics data for one cell line
        paired with the same drug features.
        """
        omics_features = self.omics_features_per_cell[idx]
        cell_line_id = self.cell_lines[idx]
        # The drug features are the same for every item in this dataset.
        return omics_features, self.drug_features, cell_line_id


def collate_fn(batch, device='cuda'):
    """
    Custom collate function for the DataLoader.
    It takes a list of samples (from __getitem__) and stacks them into a single batch
    in a dictionary format that the model expects as input.
    """
    # Unzip the list of samples into separate lists.
    omics_list, drug_list, cell_lines = zip(*batch)
    
    # Create a dictionary to hold the batched tensors.
    batch_dict = {
        # Stack individual drug feature tensors into a single batch tensor.
        'morgan': torch.stack([d['morgan'] for d in drug_list]).to(device),
        'gin':    torch.stack([d['gin'] for d in drug_list]).to(device),
        'dti':    torch.stack([d['dti'] for d in drug_list]).to(device),
        
        # Stack individual omics feature tensors into a single batch tensor.
        'crispr': torch.stack([o['crispr'] for o in omics_list]).to(device),
        'pro':    torch.stack([o['pro'] for o in omics_list]).to(device),
        'exp':    torch.stack([o['exp'] for o in omics_list]).to(device),
        'meth':   torch.stack([o['meth'] for o in omics_list]).to(device),
    }
    # The dictionary is now ready to be passed as input to the model.
    return batch_dict