# callbacks/_helpers.py
# This module contains common helper functions used exclusively by the callback
# handlers. The leading underscore indicates internal use within the 'callbacks' package.

import gradio as gr
import pandas as pd
import math


def _format_dataframe_for_display(df, score_column='Score', columns_map=None, columns_order=None) -> pd.DataFrame:
    """
    A general helper to format DataFrames before displaying them in a Gradio component.
    It formats score columns, renames columns, reorders them, and fills empty cells.
    """
    display_df = df.copy()
    # Format the score column to 3 decimal places.
    if score_column in display_df.columns:
        display_df[score_column] = display_df[score_column].apply(lambda x: f'{x:.3f}')
    # Rename columns for a more user-friendly display.
    if columns_map:
        display_df = display_df.rename(columns=columns_map)
    # Reorder columns to a specific sequence.
    if columns_order:
        display_df = display_df[columns_order]
    # Fill any remaining NaN values with a dash.
    display_df.fillna('-', inplace=True)
    return display_df

def _create_drug_info_text(drug_info: pd.Series, is_internal: bool) -> str:
    """
    Creates the formatted markdown string for displaying info of a selected drug.
    The content varies based on the data source (internal vs. external).
    """
    if is_internal:
        return (f"**Drug:** {drug_info.get('drug', 'N/A')} | "
                f"**Target:** {drug_info.get('targets', 'N/A')} | "
                f"**Pathway:** {drug_info.get('pathway_name', 'N/A')} | "
                f"**Score:** {drug_info.get('Score', 0):.3f}")
    else:
        return (f"**Drug:** {drug_info.get('drug', 'N/A')} | "
                f"**Target:** {drug_info.get('targets', 'N/A')} | "
                f"**MOA:** {drug_info.get('moa', 'N/A')} | "
                f"**Score:** {drug_info.get('Score', 0):.3f}")

def _prepare_pagination_updates(df: pd.DataFrame, page: int, selected_drug: str = None) -> dict:
    """
    Generates a dictionary of all necessary Gradio updates for the pagination component.
    This includes button visibility, text, variants, and page info.
    """
    page_size = 5
    total_pages = math.ceil(len(df) / page_size) if not df.empty else 1
    
    # Slice the dataframe to get drugs for the current page.
    start_index = (page - 1) * page_size
    page_drugs = df.iloc[start_index : start_index + page_size]['drug'].tolist()
    
    # If no drug is actively selected, default to the first one on the page.
    if selected_drug is None and page_drugs:
        selected_drug = page_drugs[0]

    # Create a list of Gradio button updates for the current page of drugs.
    button_updates = [
        gr.update(value=drug, visible=True, variant='primary' if drug == selected_drug else 'secondary')
        for drug in page_drugs
    ]
    # Fill any unused button slots with invisible buttons.
    button_updates += [gr.update(visible=False)] * (page_size - len(page_drugs))
    
    # Return all updates in a dictionary to avoid ordering errors in the main handlers.
    return {
        "buttons": button_updates,
        "page_info": f"Page {page} / {total_pages}",
        "prev_btn": gr.update(interactive=(page > 1)),
        "next_btn": gr.update(interactive=(page < total_pages))
    }