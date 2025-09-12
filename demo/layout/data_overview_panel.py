# layout/data_overview_panel.py
# This module builds the "Overview of Internal Dataset" accordion panel,
# which displays visualizations of the training data.

import gradio as gr

# Import pre-loaded data and statistics from the main setup file.
from app_setup import STATS, DF_CELL_LINES, DF_DRUGS, DF_OMICS_UMAP, DF_DRUG_UMAP

# Import plotting functions.
from visualizations import (
    create_interactive_cell_line_treemap, 
    create_stacked_bar_chart, 
    create_3d_cell_line_umap_plot, 
    create_3d_drug_umap_plot
)


def build_overview_panel() -> dict:
    """Builds the 'Overview of Internal Dataset' accordion and returns its components."""
    with gr.Accordion("Overview of Internal Dataset", open=True) as overview_accordion:
        with gr.Tabs():
            # Tab for cell line composition treemap and summary stats.
            with gr.TabItem("Cell Line Composition"):
                gr.Plot(value=create_interactive_cell_line_treemap(DF_CELL_LINES), show_label=False, elem_id='cell_line_treemap_container')
                with gr.Row():
                    gr.Button(f"Total Samples: {STATS.get('total_samples', 0)}", elem_classes="info-button")
                    gr.Button(f"Total Tissues: {STATS.get('total_tissues', 0)}", elem_classes="info-button")
                    gr.Button(f"Total Cancer Types: {STATS.get('total_cancer_types', 0)}", elem_classes="info-button")

            # Tab for drug composition bar chart and summary stats.
            with gr.TabItem("Drug Composition"):
                pathway_top_n_state = gr.State(value=20)
                pathway_plot = gr.Plot(value=create_stacked_bar_chart(DF_DRUGS, 'pathway_name', 'targets', 20), show_label=False, elem_id='drug_composition_bar_container')
                with gr.Row(elem_id='info-panel-row'):
                    gr.Button(f"Total Drugs: {STATS.get('total_drugs', 0)}", elem_classes="info-button")
                    gr.Button(f"Total Targets: {STATS.get('total_targets', 0)}", elem_classes="info-button")
                    gr.Button(f"Total Pathways: {STATS.get('total_pathways', 0)}", elem_classes="info-button")
                    load_more_btn = gr.Button("Load More", variant="secondary", elem_id="load-more-button")

            # Tab for 3D UMAP visualization of cell line embeddings.
            with gr.TabItem("Cell Line Embedding Analysis"):
                gr.Markdown("#### 3D UMAP of Omics Embeddings (for Erlotinib)")
                gr.Markdown("This plot shows how the model groups cell lines based on their omics data when interacting with Erlotinib. You can rotate the plot by clicking and dragging.")
                cell_line_umap_plot = gr.Plot(value=create_3d_cell_line_umap_plot(DF_OMICS_UMAP, 'tissue', 'model_id'), show_label=False, elem_id='cell_line_umap_container')
                umap_cell_line_color_radio = gr.Radio(choices=['tissue', 'cancer_type'], value='tissue', label="Group by:")
            
            # Tab for 3D UMAP visualization of drug embeddings.
            with gr.TabItem("Drug Embedding Analysis"):
                gr.Markdown("#### 3D UMAP of Drug Embeddings")
                gr.Markdown("This plot shows how the model groups different drugs based on their learned features. You can rotate the plot by clicking and dragging.")
                drug_umap_plot = gr.Plot(value=create_3d_drug_umap_plot(DF_DRUG_UMAP, 'pathway_name'), show_label=False, elem_id='drug_umap_container')
                umap_drug_color_radio = gr.Radio(choices=['pathway_name', 'targets'], value='pathway_name', label="Group by")
                
    # Return a dictionary of all components created in this panel for event registration.
    return {
        "overview_accordion": overview_accordion,
        "pathway_top_n_state": pathway_top_n_state,
        "pathway_plot": pathway_plot,
        "load_more_btn": load_more_btn,
        "cell_line_umap_plot": cell_line_umap_plot,
        "umap_cell_line_color_radio": umap_cell_line_color_radio,
        "drug_umap_plot": drug_umap_plot,
        "umap_drug_color_radio": umap_drug_color_radio
    }