# app.py
import gradio as gr
import pandas as pd
from predictor import DrugResponsePredictor

from dataset.omics_drug_response_dataset import OmicsDrugResponseInferenceDataset, collate_fn, OmicsDrugResponseDataset

# Assume 'predictor' and 'PREDICTOR_LOADED' are defined globally
try:
    predictor = DrugResponsePredictor()
    PREDICTOR_LOADED = True
except Exception as e:
    PREDICTOR_LOADED = False
    print(f"Failed to load predictor: {e}")

def predict_wrapper(smiles):
    """Function to be called by Gradio. Includes error handling."""
    if not PREDICTOR_LOADED:
        raise gr.Error("Failed to load the model. Please check the server logs.")
    if not smiles:
        gr.Warning("Please enter a SMILES string!")
        return pd.DataFrame(), pd.DataFrame()
    try:
        pred_df, sim_df = predictor.predict(smiles)
        gr.Info("Prediction complete!")
        return pred_df, sim_df
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"An error occurred during prediction: {e}")

# Define the Gradio UI
def create_ui():
    with gr.Blocks(theme=gr.themes.Default(), title="Drug Response Predictor") as demo:
        gr.Markdown(
            """
            # AI-based Anticancer Drug Response and Similar Drug Prediction
            Enter a molecular structure (SMILES), and the AI model will predict the drug response (lnIC50) across various cancer cell lines, 
            and analyze the response pattern similarity to known anticancer drugs.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                smiles_input = gr.Textbox(
                    label="Enter SMILES of the drug to analyze",
                    placeholder="e.g., COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
                    info="Paste the SMILES string of the molecule here."
                )
                
                predict_button = gr.Button("🚀 Run Prediction", variant="primary")
                
                gr.Examples(
                    examples=[
                        ["COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"], # Erlotinib
                    ],
                    inputs=smiles_input,
                    label="Example SMILES"
                )

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("📈 IC50 Prediction per Cell Line"):
                        gr.Markdown("#### Predicted drug sensitivity (lnIC50) for each cancer cell line. (Lower values indicate higher effectiveness)")
                        ic50_output = gr.DataFrame(
                            headers=["cell_lines", "tissue", "cancer_type", "pred_lnIC50"],
                            datatype=["str", "str", "str", "number"],
                            wrap=True,
                        )

                    with gr.TabItem("📊 Similar Drug Analysis"):
                        gr.Markdown("#### List of existing anticancer drugs with the most similar response patterns.")
                        similarity_output = gr.DataFrame(
                            headers=["drug", "Score", "RMSE_norm", "Pearson"],
                            datatype=["str", "number", "number", "number"],
                            wrap=True,
                        )

        # Button click event handler
        predict_button.click(
            fn=predict_wrapper,
            inputs=smiles_input,
            outputs=[ic50_output, similarity_output],
            api_name="predict"
        )
        
    return demo


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0")