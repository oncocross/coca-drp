# predictor.py

# --- Core Libraries ---
import torch
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader

# --- Custom Application Modules ---
# Import necessary classes and functions from other project modules.
from model.cdr_model import CDRModel
from dataset.omics_drug_response_dataset import OmicsDrugResponseInferenceDataset, collate_fn, OmicsDrugResponseDataset
from utils.feature_generator import generate_drug_features, DtiPredictor

class DrugResponsePredictor:
    """A class to load all models and data, and run the complete prediction pipeline."""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the predictor by loading all necessary models, data, and metadata.
        This setup is done once when the application starts.
        
        Args:
            device (str): The computing device to use ('cuda' for GPU or 'cpu').
        """
        print("Initializing predictor... This may take a moment.")
        
        # Set the computation device (GPU if available, otherwise CPU).
        self.device = device
        print(f"Using device: {self.device}")
        
        # --- Model Loading ---
        # 1. Load the main Cancer Drug Response (CDR) model and its pre-trained weights.
        self.cdr_model = CDRModel().to(self.device)
        cdr_weight_path = './weights/best_model.pt'
        self.cdr_model.load_state_dict(torch.load(cdr_weight_path, map_location=self.device, weights_only=True))
        self.cdr_model.eval() # Set the model to evaluation mode (disables dropout, etc.).
        
        # 2. Initialize the Drug-Target Interaction (DTI) predictor, used for drug feature generation.
        dti_weight_path = './GCADTI/model.pt'
        prot_list_path = './GCADTI/IC_test.csv'
        self.dti_predictor = DtiPredictor(dti_weight_path, prot_list_path, self.device)
        
        # --- Data Loading and Preprocessing ---
        # 3. Load the pre-processed omics data for all cell lines.
        #    The global assignment is a workaround to help joblib unpickle a custom class.
        globals()['OmicsDrugResponseDataset'] = OmicsDrugResponseDataset
        self.cell_lines, self.omics_features = self._extract_unique_cell_omics('./data/omics_drug_dataset.joblib')
        
        # 4. Load internal datasets (used for training and primary similarity analysis).
        self.drug_meta = pd.read_csv('./data/drug_all.csv') # Metadata for known drugs (targets, pathways).
        self.ic50_ref = pd.read_csv('./data/20231023_092657_imputed_drugresponse.csv', index_col=0) # Reference IC50 data for similarity.
        self.cell_all = pd.read_csv('./data/model_list_20250630.csv') # Metadata for cell lines (tissue, cancer type).

        # 5. Load and preprocess external (unseen) datasets for secondary similarity analysis.
        self.drh_ic50_df = pd.read_csv('./data/DRH_drugresponse.csv', index_col=0)
        self.drh_ic50_df.columns = self.drh_ic50_df.columns.str.replace('(pred.)', '', regex=False) # Clean column names.
        self.drh_meta_df = pd.read_csv('./data/DRH_meta.csv')

        # 6. Filter external data to remove any drugs that overlap with the internal dataset.
        #    This ensures the external analysis is performed on truly "unseen" compounds.
        internal_pubchem_ids = self.drug_meta['pubchem'].dropna().unique()
        internal_drug_names_normalized = self.drug_meta['drug_name'].str.lower().str.replace(r'[-_]', '', regex=True).dropna().unique()

        # Create a normalized name column in the external metadata for case-insensitive comparison.
        self.drh_meta_df['pert_iname_normalized'] = self.drh_meta_df['pert_iname'].str.lower().str.replace(r'[-_]', '', regex=True)
        self.drh_meta_df.drop_duplicates(subset=['pert_iname_normalized', 'target', 'moa'], keep='first', inplace=True)
        
        # Identify overlapping drugs by checking both PubChem CIDs and normalized names.
        overlapping_drugs_meta = self.drh_meta_df[
            self.drh_meta_df['pubchem_cid'].isin(internal_pubchem_ids) |
            self.drh_meta_df['pert_iname_normalized'].isin(internal_drug_names_normalized)
        ]
        overlapping_drug_names = overlapping_drugs_meta['pert_iname'].unique()

        # Remove the identified overlapping drugs from the external IC50 and metadata DataFrames.
        self.drh_ic50_df = self.drh_ic50_df.drop(columns=overlapping_drug_names, errors='ignore')
        self.drh_meta_df = self.drh_meta_df[~self.drh_meta_df['pert_iname'].isin(overlapping_drug_names)]
        
        # 7. Final preparation of the external metadata for later use.
        self.drh_meta_df['targets'] = self.drh_meta_df['target'].str.replace('|', ', ', regex=False)
        self.drh_meta_df['moa'] = self.drh_meta_df['moa'].str.replace('|', ', ', regex=False)
        self.drh_meta_df['pert_iname_lower'] = self.drh_meta_df['pert_iname'].str.lower()
        
        print("Predictor initialized successfully!")

    def _extract_unique_cell_omics(self, dataset_path: str):
        """
        Extracts unique cell line omics data from a joblib dataset file.
        The raw data is stored as tensors for each omics type; this function
        reorganizes it into a list where each element represents one cell line.

        Args:
            dataset_path (str): The path to the .joblib dataset file.

        Returns:
            tuple: A tuple containing (list_of_cell_line_names, list_of_omics_feature_dicts).
        """
        dataset_full = joblib.load(dataset_path)
        cell_line_list = dataset_full.cell_lines
        omics_tensor_dict = dataset_full.omics_tensors
        
        # Restructure the omics data into a list of dictionaries, one for each cell line.
        omics_list = []
        for idx, cell_line in enumerate(cell_line_list):
            omics_features_for_cell = {
                omics_type: omics_tensor_dict[omics_type][idx]
                for omics_type in omics_tensor_dict
            }
            omics_list.append(omics_features_for_cell)
        
        return cell_line_list, omics_list

    def _calculate_similarity_scores(self, pred_df: pd.DataFrame, ref_ic50_df: pd.DataFrame):
        """
        Calculates similarity scores by comparing the predicted IC50 profile against a
        reference IC50 data matrix of known drugs.

        The score is a weighted average of normalized RMSE and Pearson correlation.

        Args:
            pred_df (pd.DataFrame): DataFrame with predicted lnIC50 values for the input drug.
            ref_ic50_df (pd.DataFrame): Reference DataFrame where rows are cell lines and columns are known drugs.

        Returns:
            pd.DataFrame: A DataFrame containing drugs and their calculated similarity scores, sorted descending.
        """
        pred_indexed = pred_df.set_index('cell_lines')
        # Find the common cell lines between the prediction and the reference data.
        common_cells = ref_ic50_df.index.intersection(pred_indexed.index)
        
        # Require at least 2 common cell lines to calculate meaningful correlation.
        if len(common_cells) < 2:
            return pd.DataFrame(columns=['drug', 'Score'])

        # Subset both dataframes to only include common cell lines.
        ic50_subset = ref_ic50_df.loc[common_cells]
        pred_subset = pred_indexed.loc[common_cells]
        y_pred = pred_subset['pred_lnIC50'].values
        
        scores = []
        # Iterate through each known drug (column) in the reference data.
        for drug_col in ic50_subset.columns:
            y_true = ic50_subset[drug_col].values
            # Ensure we only compare non-NaN values.
            valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if np.sum(valid_indices) < 2: continue

            y_true_valid, y_pred_valid = y_true[valid_indices], y_pred[valid_indices]
            
            # --- Calculate the custom similarity score ---
            # 1. Normalized Root Mean Squared Error (lower is better).
            rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid)**2))
            y_range = y_true_valid.max() - y_true_valid.min()
            norm_rmse = rmse / y_range if y_range > 0 else 1
            # 2. Pearson Correlation (higher is better). We clip it to [0, 1] to focus on positive correlation.
            pearson_corr = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]
            pearson_clip = np.clip(pearson_corr, 0, 1)
            # 3. Final score combines both metrics.
            score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * pearson_clip
            
            scores.append({
                # Parse the drug name from the column header.
                "drug": drug_col.split(';')[1] if ';' in drug_col else drug_col,
                "Score": score
            })
            
        if not scores:
            return pd.DataFrame(columns=['drug', 'Score'])
            
        # Return the results as a DataFrame, sorted by the highest score.
        return pd.DataFrame(scores).sort_values("Score", ascending=False).reset_index(drop=True)


    def _merge_with_metadata(self, scores_df: pd.DataFrame, meta_df: pd.DataFrame, 
                                 join_key_meta: str, features: list) -> pd.DataFrame:
        """
        Merges a scores DataFrame with a metadata DataFrame to enrich it with additional information.
        Performs a case-insensitive join on the drug name.

        Args:
            scores_df (pd.DataFrame): DataFrame with 'drug' and 'Score' columns.
            meta_df (pd.DataFrame): DataFrame containing metadata.
            join_key_meta (str): The column name in meta_df to join on (e.g., 'drug_name').
            features (list): A list of metadata column names to include in the final result.

        Returns:
            pd.DataFrame: The merged DataFrame with drug, score, and requested features.
        """
        if scores_df.empty:
            return pd.DataFrame(columns=['drug', 'Score'] + features)

        # Select only the necessary columns from the metadata for the merge.
        cols_to_merge = [join_key_meta] + features
        info_to_merge = meta_df[cols_to_merge].copy()
        
        # Create a lowercase version of the join key for case-insensitive matching.
        info_to_merge_lower_key = join_key_meta + '_lower'
        info_to_merge[info_to_merge_lower_key] = info_to_merge[join_key_meta].str.lower()

        # Perform a left merge from scores to metadata.
        merged_df = pd.merge(
            scores_df,
            info_to_merge,
            left_on=scores_df['drug'].str.lower(),
            right_on=info_to_merge_lower_key,
            how='left'
        )
        # Return only the essential columns.
        return merged_df[['drug', 'Score'] + features]
        
    def predict(self, smiles: str):
        """
        Executes the full prediction pipeline for a given SMILES string.
        
        Args:
            smiles (str): The SMILES string of the compound to analyze.

        Returns:
            tuple: A tuple of three DataFrames:
                   1. pred_df: Predicted IC50 values for the input drug.
                   2. internal_sim_df: Similar drugs from the internal reference dataset.
                   3. external_sim_df: Similar drugs from the external (unseen) reference dataset.
        """
        # Validate the input SMILES string.
        if not smiles or not isinstance(smiles, str):
            raise ValueError("Please provide a valid SMILES string.")

        try:
            with torch.no_grad():
                # --- Step 1: Generate features for the input drug ---
                drug_features = generate_drug_features(smiles, self.dti_predictor)
                
                # --- Step 2: Prepare data for the model ---
                # Create a custom dataset and DataLoader for batch processing.
                dataset = OmicsDrugResponseInferenceDataset(self.cell_lines, self.omics_features, drug_features)
                loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=lambda b: collate_fn(b, device=self.device))
                
                # --- Step 3: Run model prediction ---
                # Process the data in batches to manage memory usage.
                all_preds = []
                for batch in loader:
                    output, _, _ = self.cdr_model(batch)
                    all_preds.append(output.cpu()) # Move predictions to CPU
                
                # --- Step 4: Format prediction results ---
                # Concatenate predictions from all batches and create a DataFrame.
                merged_preds = torch.cat(all_preds).numpy().flatten()
                pred_df = pd.DataFrame({'cell_lines': self.cell_lines, 'pred_lnIC50': merged_preds})

                # --- Step 5: Merge predictions with cell line metadata ---
                # Add human-readable info like tissue and cancer type to the results.
                pred_df = pred_df.merge(
                    self.cell_all[['model_id', 'tissue', 'cancer_type']],
                    how='left', left_on='cell_lines', right_on='model_id'
                )
                pred_df = pred_df[['cell_lines', 'tissue', 'cancer_type', 'pred_lnIC50']]
                
                # --- Step 6: Calculate Similarity with Internal Drugs ---
                # Compare the prediction profile against the internal reference dataset.
                internal_scores = self._calculate_similarity_scores(pred_df, self.ic50_ref)
                internal_sim_df = self._merge_with_metadata(
                    scores_df=internal_scores,
                    meta_df=self.drug_meta,
                    join_key_meta='drug_name',
                    features=['targets', 'pathway_name']
                )
        
                # --- Step 7: Calculate Similarity with External (Unseen) Drugs ---
                # Compare the prediction profile against the external reference dataset.
                external_scores = self._calculate_similarity_scores(pred_df, self.drh_ic50_df)
                external_sim_df = self._merge_with_metadata(
                    scores_df=external_scores,
                    meta_df=self.drh_meta_df,
                    join_key_meta='pert_iname',
                    features=['targets', 'moa']
                )
                
                return pred_df, internal_sim_df, external_sim_df
                
        except Exception as e:
            # Handle any exceptions that occur during the prediction process.
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            raise e