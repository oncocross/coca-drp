# main.py

import gradio as gr

# Required for the data unpickling process to find the custom class definition.
from dataset.omics_drug_response_dataset import OmicsDrugResponseDataset

# Import UI builders, event registration logic, and utility functions.
from visualizations import create_placeholder_fig
from callbacks import register_event_listeners
from layout import (
    build_overview_panel,
    build_main_io_panel,
    build_analysis_panel
)


def create_ui():
    """Constructs the complete Gradio UI by assembling modules and registering events."""

    # Consolidate all CSS files into a single string for injection.
    css_files = [
        "./static/css/_reset.css",
        "./static/css/layout.css",
        "./static/css/components.css"
    ]
    custom_css = ""
    for css_file in css_files:
        try:
            with open(css_file, "r", encoding="utf-8") as f:
                custom_css += f.read() + "\n"
        except FileNotFoundError:
            print(f"Warning: CSS file not found at {css_file}")
            
    # Define the main Gradio Blocks layout.
    with gr.Blocks(theme=gr.themes.Default(), title="Drug Response Predictor", css=custom_css) as demo:
        # Define persistent states.
        states = {
            "pred_data": gr.State(),
            "sim_data": gr.State(),
            "external_sim_data": gr.State()
        }
        
        # dd application header.
        gr.Markdown("""
        # AI-based Anticancer Drug Response and Similar Drug Prediction
        Enter a molecular structure (SMILES), and the AI model will predict the drug response (lnIC50) across various cancer cell lines, 
        and analyze the response pattern similarity to known anticancer drugs.
        """)

        # Build the main UI layout by calling panel builder functions.
        overview_comps = build_overview_panel()
        with gr.Row(elem_id="main_container"):
            main_io_comps = build_main_io_panel()
            analysis_comps = build_analysis_panel()

        # Consolidate all created components into a single dictionary.
        all_components = {**overview_comps, **main_io_comps, **analysis_comps}
        
        # Link all UI components to their respective callback functions.
        register_event_listeners(all_components, states)

        # Initialize plots with placeholders when the app loads.
        demo.load(fn=lambda: create_placeholder_fig(), outputs=[all_components["ic50_comp_plot"]])\
           .then(fn=lambda: create_placeholder_fig(), outputs=[all_components["ic50_dist_plot"]])\
           .then(fn=lambda: create_placeholder_fig(), outputs=[all_components["dist_plot"]])\
           .then(fn=lambda: create_placeholder_fig(), outputs=[all_components["corr_plot"]])
            
    return demo


if __name__ == "__main__":
    # Create and launch the Gradio app.
    app = create_ui()
    app.launch(share=False, server_name="0.0.0.0")