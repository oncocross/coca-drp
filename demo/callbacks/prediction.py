# callbacks/prediction.py
# This module contains the core callback functions related to the main
# prediction workflow, from SMILES input to generating all results.

import gradio as gr
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, inchi

# Import global data and the main predictor instance.
from app_setup import PREDICTOR, IS_PREDICTOR_LOADED, DF_IC50

# Import visualization functions.
from visualizations import (
    create_drug_ic50_correlation_plot,
    create_cell_line_ic50_comp_plot,
    create_cell_line_ic50_dist_plot,
    create_drug_similarity_dist_plot,
    create_placeholder_fig
)

# Import helper functions from within this package.
from ._helpers import (
    _format_dataframe_for_display, 
    _create_drug_info_text, 
    _prepare_pagination_updates
)


def handle_molecular_description(smiles: str, close_accordion: bool = False) -> tuple:
    """Calculates and displays molecular properties for the input SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        error_msg = "Error: Invalid SMILES string."
        # The returned tuple's length must match the number of output components.
        ui_updates = (gr.update(visible=True), gr.update(value=None), gr.update(value=error_msg)) + (gr.update(value=""),) * 8
        return (gr.update(open=False),) + ui_updates if close_accordion else ui_updates

    try:
        # Calculate all properties and store them in a dictionary.
        properties = {
            "img": Draw.MolToImage(mol, size=(350, 350)),
            "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
            "can_smiles": Chem.MolToSmiles(mol, canonical=True),
            "logp": f"{Descriptors.MolLogP(mol):.2f}",
            "tpsa": f"{Descriptors.TPSA(mol):.2f} Å²",
            "h_donors": Lipinski.NumHDonors(mol),
            "h_acceptors": Lipinski.NumHAcceptors(mol)
        }
        try:
            ich = inchi.MolToInchi(mol)
            properties["inchikey"] = inchi.InchiToInchiKey(ich)
        except Exception:
            properties["inchikey"] = "N/A"
            
        # Create a tuple of Gradio updates from the properties dictionary.
        ui_updates = (
            gr.update(visible=True), gr.update(value=properties["img"]),
            gr.update(value=properties["formula"]), gr.update(value=properties["weight"]),
            gr.update(value=properties["can_smiles"]), gr.update(value=properties["inchikey"]),
            gr.update(value=properties["logp"]), gr.update(value=properties["tpsa"]),
            gr.update(value=properties["h_donors"]), gr.update(value=properties["h_acceptors"])
        )
        # If requested, prepend an update to close the accordion.
        return (gr.update(open=False),) + ui_updates if close_accordion else ui_updates

    except Exception as e:
        error_msg = f"Error calculating properties: {e}"
        ui_updates = (gr.update(visible=True), gr.update(value=None), gr.update(value=error_msg)) + (gr.update(value=""),) * 8
        return (gr.update(open=False),) + ui_updates if close_accordion else ui_updates


def handle_smiles_input_change(smiles_text: str) -> gr.update:
    """Validates SMILES in real-time to enable/disable the 'Run Prediction' button."""
    if not smiles_text or not smiles_text.strip():
        return gr.update(interactive=False)
    # RDKit's MolFromSmiles returns None for invalid SMILES.
    mol = Chem.MolFromSmiles(smiles_text, sanitize=True)
    return gr.update(interactive=(mol is not None))


def handle_prediction_and_plot_generation(smiles: str, ic50_thresh: float, ic50_group_by: str, sim_thresh: float, sim_group_by: str) -> tuple:
    """
    Main handler for the 'Run Prediction' button. Triggers prediction, processes results,
    and returns updates for all relevant UI components.
    """
    # Define a default empty return value to handle cases of failed prediction.
    placeholder_fig = create_placeholder_fig()
    fail_return = (
        [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        placeholder_fig, "", gr.update(visible=False),
        placeholder_fig, "", gr.update(visible=False),
        placeholder_fig, "", gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), placeholder_fig, *[gr.update(visible=False)] * 5,
        gr.update(visible=False), 1, None, gr.update(value="...", visible=False),
        "Page 1 / 1", gr.update(interactive=False), gr.update(interactive=False)
    )

    if not IS_PREDICTOR_LOADED: raise gr.Error("Predictor not loaded.")
    if not smiles: return fail_return

    # 1. Get prediction results from the core predictor object.
    pred_df, sim_df, external_sim_df = PREDICTOR.predict(smiles)

    # 2. Apply deduplication to the similarity results.
    sim_df = sim_df.drop_duplicates(subset=['drug'], keep='first')
    external_sim_df = external_sim_df.drop_duplicates(subset=['drug'], keep='first')
    
    if pred_df.empty or sim_df.empty:
        gr.Warning("Prediction returned no results.")
        return fail_return

    # 3. Format the raw dataframes for UI display using a helper.
    pred_df_display = _format_dataframe_for_display(pred_df.drop_duplicates(), columns_order=["cell_lines", "tissue", "cancer_type", "pred_lnIC50"])
    pred_df_display['pred_lnIC50'] = pred_df_display['pred_lnIC50'].astype(float).map('{:.3f}'.format)
    sim_df_display = _format_dataframe_for_display(sim_df, columns_map={'drug': 'Drug', 'targets': 'Target', 'pathway_name': 'Pathway'}, columns_order=['Drug', 'Target', 'Pathway', 'Score'])
    external_sim_df_display = _format_dataframe_for_display(external_sim_df, columns_map={'drug': 'Drug', 'targets': 'Target', 'moa': 'MOA'}, columns_order=['Drug', 'Target', 'MOA', 'Score'])
    
    # 4. Generate the initial analysis plots based on default control values.
    fig_ic50_comp, info_ic50_comp = create_cell_line_ic50_comp_plot(pred_df, ic50_thresh, ic50_group_by)
    fig_ic50_dist, info_ic50_dist = create_cell_line_ic50_dist_plot(pred_df, ic50_thresh, ic50_group_by)
    fig_sim_dist, info_sim_dist = create_drug_similarity_dist_plot(sim_df, sim_thresh, sim_group_by)
    
    # 5. Prepare the initial state for the IC50 correlation plot.
    selected_drug = sim_df.iloc[0]['drug']
    fig_scatter = create_drug_ic50_correlation_plot(pred_df, DF_IC50, selected_drug)
    info_scatter = _create_drug_info_text(sim_df.iloc[0], is_internal=True)
    pagination_updates = _prepare_pagination_updates(sim_df, page=1, selected_drug=selected_drug)
    dataset_info_text = "*Note.* **The Genomics of Drug Sensitivity in Cancer (GDSC)** dataset was used as the reference for training."
    
    # 6. Assemble and return the complete tuple of updates in the correct order.
    return (
        pred_df_display.values.tolist(), sim_df_display.values.tolist(), external_sim_df_display.values.tolist(),
        pred_df, sim_df, external_sim_df,
        fig_ic50_comp, info_ic50_comp, gr.update(visible=True),
        fig_ic50_dist, info_ic50_dist, gr.update(visible=True),
        fig_sim_dist, info_sim_dist, gr.update(visible=True), gr.update(visible=True), gr.update(value=dataset_info_text, visible=True),
        gr.update(visible=True), gr.update(value=dataset_info_text, visible=True), fig_scatter, *pagination_updates["buttons"],
        gr.update(visible=True), 1, selected_drug, gr.update(value=info_scatter, visible=True),
        pagination_updates["page_info"], pagination_updates["prev_btn"], pagination_updates["next_btn"]
    )