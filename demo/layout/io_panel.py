# layout/io_panel.py
# This module builds the left-hand column of the UI, which handles all
# user inputs (SMILES) and displays the main output tables.

import gradio as gr

def build_main_io_panel() -> dict:
    """Builds the main left column for SMILES input and result dataframes."""
    with gr.Column(scale=2, elem_classes=["left-column", "stretch-height"]):
        # --- User Input Section ---
        smiles_input = gr.Textbox(label="Enter SMILES of the drug to analyze", placeholder="e.g., COCCOC1...", info="Paste the SMILES string of the molecule here.")
        predict_button = gr.Button("🚀 Run Prediction", variant="primary", interactive=False)
        gr.Examples(examples=[["COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"]], inputs=smiles_input, label="Example SMILES")
        
        # --- Output Tabs Section ---
        with gr.Tabs():
            # Tab for displaying detailed molecular properties.
            with gr.TabItem("ℹ️ Molecular Desc."):
                with gr.Column(visible=False) as mol_desc_panel:
                    mol_image_output = gr.Image(show_label=False, show_download_button=False, elem_id="mol_image_container", interactive=False, show_fullscreen_button=False)
                    with gr.Row():
                        # Basic molecular properties.
                        with gr.Column():
                            gr.Markdown("### Basic Properties")
                            formula_output = gr.Textbox(label="Chemical Formula", interactive=False)
                            weight_output = gr.Textbox(label="Molecular Weight", interactive=False)
                            can_smiles_output = gr.Textbox(label="Canonical SMILES", interactive=False)
                            inchikey_output = gr.Textbox(label="InChiKey", interactive=False)
                        # Physicochemical properties.
                        with gr.Column():
                            gr.Markdown("### Physicochemical Properties")
                            logp_output = gr.Textbox(label="LogP (Lipophilicity)", interactive=False)
                            tpsa_output = gr.Textbox(label="Topological Polar Surface Area (TPSA)", interactive=False)
                            hdonors_output = gr.Textbox(label="Hydrogen Bond Donors", interactive=False)
                            hacceptors_output = gr.Textbox(label="Hydrogen Bond Acceptors", interactive=False)
            
            # Tab for the main lnIC50 prediction results.
            with gr.TabItem("📈 IC50 Pred."):
                gr.Markdown("#### Predicted drug sensitivity (lnIC50) for each cancer cell line.")
                ic50_output = gr.DataFrame(headers=["Cell line", "Tissue", "Cancer Type", "Pred. lnIC50"], wrap=True)
            
            # Tab for similar drugs from the internal (GDSC) dataset.
            with gr.TabItem("📊 [GDSC] Similar Drug Pred."):
                gr.Markdown("#### List of existing anticancer drugs with similar response patterns.")
                similarity_output = gr.DataFrame(headers=["Drug", "Target", "Pathway", "Score"], wrap=True)
            
            # Tab for similar drugs from the external (DRH) dataset.
            with gr.TabItem("📊 [DRH] Similar Drug Pred."):
                gr.Markdown("#### Similar drugs from an external dataset not seen during training.")
                external_similarity_output = gr.DataFrame(headers=["Drug", "Target", "MOA", "Score"], wrap=True)
    
    # Return a dictionary of all created components for event registration.
    return {
        "smiles_input": smiles_input, "predict_button": predict_button,
        "mol_desc_panel": mol_desc_panel, "mol_image_output": mol_image_output,
        "formula_output": formula_output, "weight_output": weight_output,
        "can_smiles_output": can_smiles_output, "inchikey_output": inchikey_output,
        "logp_output": logp_output, "tpsa_output": tpsa_output,
        "hdonors_output": hdonors_output, "hacceptors_output": hacceptors_output,
        "ic50_output": ic50_output, "similarity_output": similarity_output,
        "external_similarity_output": external_similarity_output
    }