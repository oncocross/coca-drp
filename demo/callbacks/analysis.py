# callbacks/analysis.py
# This module contains all callback functions that handle events for the
# main analysis plots in the right-hand column of the UI.

import gradio as gr
import pandas as pd
import math

# Import pre-loaded dataframes from the main setup file.
from app_setup import DF_IC50, DF_DRH_IC50

# Import visualization functions.
from visualizations import (
    create_drug_similarity_dist_plot,
    create_drug_ic50_correlation_plot,
    create_placeholder_fig
)

# Import helper functions from within this package.
from ._helpers import(
    _create_drug_info_text, 
    _prepare_pagination_updates
)

GDSC_INFO = "*Note.* **The Genomics of Drug Sensitivity in Cancer** (GDSC) dataset was used as the reference for training."
DRH_INFO = "*Note.* **The Drug Repurposing Hub (DRH)** dataset not seen during training."

def handle_drug_ic50_correlation_source_switch(data_source: str, pred_df: pd.DataFrame, sim_df: pd.DataFrame, external_sim_df: pd.DataFrame) -> tuple:
    """Handles switching between Internal/External data for the IC50 correlation plot."""
    is_internal = "Genomics of Drug Sensitivity in Cancer" in data_source
    df_to_use = sim_df if is_internal else external_sim_df
    data_info_text = GDSC_INFO if is_internal else DRH_INFO

    # Handle cases where the selected data source is empty.
    if df_to_use is None or df_to_use.empty:
        empty_pagination = _prepare_pagination_updates(pd.DataFrame(columns=['drug']), 1)
        return (data_info_text, create_placeholder_fig("No data available."), "No data", *empty_pagination["buttons"], empty_pagination["page_info"], empty_pagination["prev_btn"], empty_pagination["next_btn"], 1, None)

    # Reset plot to show the first drug of the newly selected dataset.
    selected_drug = df_to_use.iloc[0]['drug']
    ref_df_to_use = DF_IC50 if is_internal else DF_DRH_IC50
    fig = create_drug_ic50_correlation_plot(pred_df, ref_df_to_use, selected_drug)
    
    # Update info text and pagination for the new context.
    info_text = _create_drug_info_text(df_to_use.iloc[0], is_internal)
    pagination_updates = _prepare_pagination_updates(df_to_use, page=1, selected_drug=selected_drug)
    
    # Return a tuple of all UI updates.
    return (data_info_text, fig, info_text, *pagination_updates["buttons"], pagination_updates["page_info"], pagination_updates["prev_btn"], pagination_updates["next_btn"], 1, selected_drug)

def handle_drug_ic50_correlation_control_selection(selected_drug: str, data_source: str, pred_df: pd.DataFrame, sim_df: pd.DataFrame, external_sim_df: pd.DataFrame, *all_buttons) -> tuple:
    """Updates the IC50 correlation plot when a specific drug button is clicked."""
    is_internal = "Genomics of Drug Sensitivity in Cancer" in data_source
    df_to_use = sim_df if is_internal else external_sim_df
    
    if not selected_drug or pred_df is None or df_to_use is None:
        raise gr.Error("Required data not available.")
        
    # Regenerate the plot and info text for the selected drug.
    ref_df_to_use = DF_IC50 if is_internal else DF_DRH_IC50
    fig = create_drug_ic50_correlation_plot(pred_df, ref_df_to_use, selected_drug)
    drug_info = df_to_use[df_to_use['drug'] == selected_drug].iloc[0]
    info_text = _create_drug_info_text(drug_info, is_internal)
    
    # Update button variants to highlight the newly selected button.
    button_updates = [gr.update(variant='primary' if btn_val == selected_drug else 'secondary') for btn_val in all_buttons]
    
    return fig, info_text, *button_updates, selected_drug

def handle_drug_ic50_correlation_page_change(action: str, data_source: str, current_page: int, sim_df: pd.DataFrame, external_sim_df: pd.DataFrame, selected_drug: str) -> tuple:
    """Handles the Previous/Next button clicks for the IC50 correlation drug list."""
    is_internal = "Genomics of Drug Sensitivity in Cancer" in data_source
    df_to_use = sim_df if is_internal else external_sim_df

    if df_to_use is None or df_to_use.empty:
        return current_page, *([gr.update(visible=False)]*5), "Page 1/1", gr.update(interactive=False), gr.update(interactive=False)

    # Calculate the new page number based on the action ('prev' or 'next').
    total_pages = math.ceil(len(df_to_use) / 5)
    new_page = current_page
    if action == "next" and current_page < total_pages: new_page += 1
    elif action == "prev" and current_page > 1: new_page -= 1
    
    # Get the updated state for all pagination components.
    pagination_updates = _prepare_pagination_updates(df_to_use, page=new_page, selected_drug=selected_drug)
    
    return (new_page, *pagination_updates["buttons"], pagination_updates["page_info"], pagination_updates["prev_btn"], pagination_updates["next_btn"])

def handle_drug_dist_source_switch(data_source: str) -> gr.update:
    """Updates the 'Group By' choices when the data source for the distribution plot changes."""
    is_internal = "Genomics of Drug Sensitivity in Cancer" in data_source
    data_info_text = GDSC_INFO if is_internal else DRH_INFO
    
    # Use 'Target'/'Pathway' for internal data, 'Target'/'MOA' for external.
    if is_internal:
        return (gr.update(choices=['Target', 'Pathway'], value='Target'), data_info_text)
    else:
        return (gr.update(choices=['Target', 'MOA'], value='Target'), data_info_text)

def handle_drug_dist_plot_update(data_source: str, sim_df: pd.DataFrame, external_sim_df: pd.DataFrame, score_threshold: float, group_by: str) -> tuple:
    """Generates or updates the 'Similar Drug Distribution' plot based on UI controls."""
    is_internal = "Genomics of Drug Sensitivity in Cancer" in data_source
    df_to_use = sim_df if is_internal else external_sim_df
    
    # Correct the 'group_by' value if it's inconsistent with the new data source
    # (e.g., user switches source while 'Pathway' is selected).
    if is_internal and group_by not in ['Target', 'Pathway']:
        group_by = 'Target'
    elif not is_internal and group_by not in ['Target', 'MOA']:
        group_by = 'Target'
        
    # Regenerate the plot with the updated data and parameters.
    fig, info = create_drug_similarity_dist_plot(df_to_use, score_threshold, group_by)
    return fig, info