# callbacks/data_overview.py
# This module contains callback functions that handle events for the
# "Data Overview" panel of the UI.

# Import pre-loaded dataframes from the main setup file.
from app_setup import DF_DRUGS, DF_OMICS_UMAP, DF_DRUG_UMAP

# Import the specific plotting functions used by these callbacks.
from visualizations import (
    create_stacked_bar_chart, 
    create_3d_cell_line_umap_plot, 
    create_3d_drug_umap_plot,
)


def handle_load_more_drug_targets(current_n: int, load_more_count=10) -> tuple:
    """Handles the 'Load More' button click in the 'Drug Composition' tab."""
    # Increment the number of items to display.
    new_n = current_n + load_more_count
    # Regenerate the bar chart with the new item count.
    new_plot = create_stacked_bar_chart(DF_DRUGS, 'pathway_name', 'targets', new_n)
    return new_plot, new_n

def handle_umap_color_group_change(color_by: str, data_type: str):
    """Updates the UMAP plots based on the selected coloring group."""
    # Check which UMAP plot needs updating.
    if data_type == "cell_line":
        # Regenerate the cell line UMAP plot with the new color setting.
        return create_3d_cell_line_umap_plot(umap_df=DF_OMICS_UMAP, color_by=color_by, hover_name='model_id')
    elif data_type == "drug":
        # Regenerate the drug UMAP plot with the new color setting.
        return create_3d_drug_umap_plot(umap_df=DF_DRUG_UMAP, color_by=color_by)