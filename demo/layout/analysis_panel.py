# layout/analysis_panel.py
# This module builds the right-hand column of the UI, which contains all
# the data analysis plots and their corresponding interactive controls.

import gradio as gr

def build_analysis_panel() -> dict:
    """Builds the main right column for analysis plots and their interactive controls."""
    with gr.Column(scale=3, elem_classes=["right-column", "stretch-height"]):
        with gr.Tabs():
            # Tab for Drug IC50 Correlation scatter plot.
            with gr.TabItem("Target/Drug IC50 Correlation", elem_id="unified_ic50_correlation_tab"):
                gr.Markdown("#### Correlation analysis with similar drugs")
                corr_source_radio = gr.Radio(choices=["Genomics of Drug Sensitivity in Cancer", "Drug Repurposing Hub"], value="Genomics of Drug Sensitivity in Cancer", show_label=False, elem_id="scatter_source_selector", visible=False)
                corr_source_info_md = gr.Markdown(visible=False, elem_classes="dataset_info_text")
                corr_plot = gr.Plot(show_label=False, elem_id='unified_ic50_scatter_analysis_container')
                # This column group holds all controls for the scatter plot.
                with gr.Column(visible=False) as corr_controls:
                    corr_selected_drug_info_md = gr.Markdown("...")
                    with gr.Row():
                        corr_top_drug_buttons = [gr.Button(visible=False, scale=1) for _ in range(5)]
                    with gr.Row(elem_classes="pagination-row"):
                        corr_page_state = gr.State(value=1)
                        corr_selected_drug_state = gr.State(value=None)
                        corr_prev_button = gr.Button("◀ Previous", interactive=False, elem_classes="pagination-button", elem_id="ic50_correlation_drug_prev_button")
                        corr_page_info_md = gr.Markdown("Page 1 / 1", elem_id="drug_page_info")
                        corr_next_button = gr.Button("▶ Next", interactive=False, elem_classes="pagination-button", elem_id="ic50_correlation_drug_next_button")
            
            # Tab for Similar Drug Distribution bar chart.
            with gr.TabItem("Similar Drug Distribution", elem_id="drug_sim_dist_analysis_tab"):
                gr.Markdown("#### Distribution of Similar Drugs (Score ≥ threshold)")
                dist_source_radio = gr.Radio(choices=["Genomics of Drug Sensitivity in Cancer", "Drug Repurposing Hub"], value="Genomics of Drug Sensitivity in Cancer", show_label=False, elem_id="drug_sim_source_selector", visible=False)
                dist_source_info_md = gr.Markdown(visible=False, elem_classes="dataset_info_text")

                dist_plot = gr.Plot(show_label=False, elem_id='drug_sim_dist_analysis_container')
                with gr.Column(visible=False) as dist_controls:
                    dist_info_output = gr.Markdown()
                    with gr.Row():
                        dist_slider = gr.Slider(0.0, 1.0, 0.8, step=0.01, label="Score Threshold")
                        dist_group_by_radio = gr.Radio(['Target', 'Pathway'], value='Target', label="Group By")

            # Tab for Sensitive Cell Line Composition donut chart.
            with gr.TabItem("Cell Line Composition"):
                gr.Markdown("#### Composition of Sensitive Cell Lines (lnIC50 ≤ threshold)")
                ic50_comp_plot = gr.Plot(show_label=False, elem_id='cell_line_ic50_comp_analysis_container')
                with gr.Column(visible=False) as ic50_comp_controls:
                    ic50_comp_info_output = gr.Markdown()
                    with gr.Row():
                        ic50_comp_slider = gr.Slider(1e-2, 6.0, 1.0, step=0.1, label="lnIC50 Threshold")
                        ic50_comp_group_by_radio = gr.Radio(['Tissue', 'Cancer type'], value='Tissue', label="Group By")
            
            # Tab for Sensitive Cell Line Distribution violin plot.
            with gr.TabItem("Cell Line Distribution"):
                gr.Markdown("#### Distribution of Sensitive Cell Lines (lnIC50 ≤ threshold)")
                ic50_dist_plot = gr.Plot(show_label=False, elem_id='cell_line_ic50_dist_analysis_container')
                with gr.Column(visible=False) as ic50_dist_controls:
                    ic50_dist_info_output = gr.Markdown()
                    with gr.Row():
                        ic50_dist_slider = gr.Slider(1e-2, 6.0, 1.0, step=0.1, label="lnIC50 Threshold")
                        ic50_dist_group_by_radio = gr.Radio(['Tissue', 'Cancer type'], value='Tissue', label="Group By")
    
    # Return a dictionary of all created components for event registration.
    return {
        "corr_source_radio": corr_source_radio, "corr_source_info_md": corr_source_info_md, "corr_plot": corr_plot, "corr_controls": corr_controls,
        "corr_selected_drug_info_md": corr_selected_drug_info_md, "corr_top_drug_buttons": corr_top_drug_buttons,
        "corr_page_state": corr_page_state, "corr_selected_drug_state": corr_selected_drug_state,
        "corr_prev_button": corr_prev_button, "corr_page_info_md": corr_page_info_md, "corr_next_button": corr_next_button,
        "dist_source_radio": dist_source_radio, "dist_source_info_md": dist_source_info_md, "dist_plot": dist_plot, "dist_controls": dist_controls,
        "dist_info_output": dist_info_output, "dist_slider": dist_slider, "dist_group_by_radio": dist_group_by_radio,
        "ic50_comp_plot": ic50_comp_plot, "ic50_comp_controls": ic50_comp_controls, "ic50_comp_info_output": ic50_comp_info_output,
        "ic50_comp_slider": ic50_comp_slider, "ic50_comp_group_by_radio": ic50_comp_group_by_radio,
        "ic50_dist_plot": ic50_dist_plot, "ic50_dist_controls": ic50_dist_controls, "ic50_dist_info_output": ic50_dist_info_output,
        "ic50_dist_slider": ic50_dist_slider, "ic50_dist_group_by_radio": ic50_dist_group_by_radio
    }