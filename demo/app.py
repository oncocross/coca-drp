# app.py

# --- Core and Data Handling Libraries ---
import gradio as gr
import pandas as pd
import math
import joblib
import pickle

# --- Cheminformatics and Utilities ---
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw, inchi
from typing import Tuple, Optional
from PIL import Image  
from urllib.parse import quote

# --- Custom Application Modules ---
from functools import partial                     # Used to fix a number of arguments of a function and generate a new function
from predictor import DrugResponsePredictor       # The main class that encapsulates the prediction logic

from dataset.omics_drug_response_dataset import * # Custom dataset handling module
from utils.data_visualizations import *           # Utility functions for creating data visualizations
from utils.result_visualizations import *         # Utility functions for visualizing prediction results


# --- Predictor Initialization ---
# Attempt to initialize the main DrugResponsePredictor class when the app starts.
# This class handles all model and data loading.
try:
    # Instantiate the predictor, loading all necessary models and data into memory.
    predictor = DrugResponsePredictor()
    # Set a flag to indicate that the predictor has loaded successfully.
    PREDICTOR_LOADED = True
    print("Predictor initialized successfully.")
except Exception as e:
    # If initialization fails, set the flag to False and log a fatal error.
    # The UI will then load with empty components to prevent a crash.
    PREDICTOR_LOADED = False
    print(f"FATAL: Failed to load predictor. The application might not work as expected. Error: {e}")

# --- Data Loading and Summary Statistics ---
# Source all data required for the UI from the 'predictor' object to ensure
# a single source of truth and prevent redundant loading.
if PREDICTOR_LOADED:
    # Retrieve pre-loaded DataFrames from the successfully initialized predictor.
    cell_line_df = predictor.cell_all
    drug_df = predictor.drug_meta
    ic50_df = predictor.ic50_ref
    drh_ic50_df = predictor.drh_ic50_df
    drh_meta_df = predictor.drh_meta_df
    all_drh_drugs = drh_ic50_df.columns.tolist()

    # Calculate summary statistics for display in the 'Train Data Overview' section.
    # Cell line statistics:
    total_samples = len(cell_line_df)
    total_tissues = cell_line_df['tissue'].nunique()
    total_cancer_types = cell_line_df['cancer_type'].nunique()

    # Drug statistics:
    total_targets = drug_df['targets'].nunique()
    total_pathways = drug_df['pathway_name'].nunique()
    total_drugs = drug_df['drug_name'].nunique()
    
else:
    # If the predictor failed to load, initialize all variables with empty/default values.
    # This ensures that the Gradio UI can be built without crashing.
    cell_line_df = pd.DataFrame()
    drug_df = pd.DataFrame()
    ic50_df = pd.DataFrame()
    drh_ic50_df = pd.DataFrame()
    drh_meta_df = pd.DataFrame()
    all_drh_drugs = []
    
    # Set all statistics to zero.
    total_samples, total_tissues, total_cancer_types = 0, 0, 0
    total_targets, total_pathways, total_drugs = 0, 0, 0

# --- UMAP Data Loading ---
# Load pre-computed UMAP embedding results for visualization.
try:
    omics_umap_df = joblib.load('./data/erlotinib_omics_interact_3d_umap.joblib')
    drug_umap_df = joblib.load('./data/drug_3d_umap_with_meta.joblib')
    print("UMAP data loaded successfully.")
except FileNotFoundError:
    # If UMAP files are not found, initialize as None and print a warning.
    # The corresponding UI tabs will display a placeholder message.
    print("Warning: UMAP data file not found. The embedding analysis tab will be empty.")
    omics_umap_df = None
    drug_umap_df = None


def load_more_drug_targets(current_n):
    """
    Handles the 'Load More' button click to show more drug pathways.
    It regenerates the plot with an increased number of items to display.
    """
    # Increment the number of items to display.
    new_n = current_n + LOAD_MORE_COUNT
    
    # Re-create the stacked bar chart with the new item count.
    new_plot = create_stacked_bar_chart(
        drug_df, 
        main_axis='pathway_name', 
        stack_by='targets', 
        top_n_stack=new_n
    )
    
    # Return the new plot and the updated count.
    return new_plot, new_n

def generate_molecular_description(smiles: str) -> Tuple:
    """
    Calculates molecular properties from a SMILES string using RDKit.
    Returns a tuple of Gradio updates to populate the UI with the results.
    """
    # Attempt to create a molecule object from the SMILES string.
    mol = Chem.MolFromSmiles(smiles)

    # Handle invalid SMILES input.
    if not mol:
        error_msg = f"Error: Invalid SMILES string."
        # Return updates to show an error message.
        return (gr.update(visible=True), gr.update(value=None)) + (gr.update(value=error_msg),) + (gr.update(value=""),) * 5

    try:
        # --- Calculate various molecular descriptors ---
        can_smiles = Chem.MolToSmiles(mol, canonical=True)      # Canonical SMILES
        img = Draw.MolToImage(mol, size=(350, 350))             # 2D Molecular structure image
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)     # Molecular Formula
        weight = f"{Descriptors.MolWt(mol):.2f} g/mol"          # Molecular Weight
        logp = f"{Descriptors.MolLogP(mol):.2f}"                 # LogP (lipophilicity)
        tpsa = f"{Descriptors.TPSA(mol):.2f} Å²"                 # Topological Polar Surface Area
        h_donors = Lipinski.NumHDonors(mol)                     # Hydrogen Bond Donors
        h_acceptors = Lipinski.NumHAcceptors(mol)               # Hydrogen Bond Acceptors

        # Generate InChI and InChIKey, handling potential conversion errors.
        try:
            ich = inchi.MolToInchi(mol); inchikey = inchi.InchiToInchiKey(ich)
        except Exception:
            ich, inchikey = 'N/A', 'N/A'
        
        # Return Gradio updates for all molecular property fields.
        return (
            gr.update(visible=True),      # Show the description panel
            gr.update(value=img),         # Update image
            gr.update(value=formula),     # Update formula
            gr.update(value=weight),      # Update weight
            gr.update(value=can_smiles),  # Update canonical SMILES
            gr.update(value=inchikey),    # Update InChIKey
            gr.update(value=logp),        # Update LogP
            gr.update(value=tpsa),        # Update TPSA
            gr.update(value=h_donors),    # Update H-bond donors
            gr.update(value=h_acceptors)  # Update H-bond acceptors
        )
    except Exception as e:
        # Handle any other unexpected errors during property calculation.
        error_msg = f"Error calculating properties: {e}"
        return (gr.update(visible=True), gr.update(value=None)) + (gr.update(value=error_msg),) + (gr.update(value=""),) * 5

def close_accordion_and_generate_desc(smiles):
    """
    A wrapper function that first closes the 'Train Data Overview' accordion,
    then calls the function to generate and display molecular properties.
    """
    # Call the main function to get the updates for molecular properties.
    desc_outputs = generate_molecular_description(smiles)
    
    # Create a Gradio update to close the accordion.
    accordion_update = gr.update(open=False)
    
    # Combine and return all updates.
    return (accordion_update,) + desc_outputs

def update_button_state(smiles_text):
    """
    Validates the SMILES string in real-time to enable or disable the 
    'Run Prediction' button, providing immediate user feedback.
    """
    # Disable button if the input is empty.
    if not smiles_text or not smiles_text.strip():
        return gr.update(interactive=False)

    # Check for chemical validity using RDKit.
    mol = Chem.MolFromSmiles(smiles_text, sanitize=True)

    # Enable the button only if the SMILES string is valid (mol is not None).
    return gr.update(interactive=(mol is not None))


def update_cell_line_umap_plot(color_by):
    """Updates the 3D UMAP plot for cell lines based on the selected color-by option."""
    return create_3d_cell_line_umap_plot(
        umap_df=omics_umap_df,
        color_by=color_by,  # e.g., 'tissue' or 'cancer_type'
        hover_name='model_id',
    )

def update_drug_umap_plot(color_by):
    """Updates the 3D UMAP plot for drugs based on the selected color-by option."""
    return create_3d_drug_umap_plot(
        umap_df=drug_umap_df, 
        color_by=color_by # e.g., 'pathway_name' or 'targets'
    )

def predict_wrapper(smiles):
    """
    A simple wrapper for the main prediction function. Includes basic error handling.
    Note: This function appears to be unused in the final app flow.
    """
    # Check if the model was loaded at startup.
    if not PREDICTOR_LOADED:
        raise gr.Error("Failed to load the model. Please check the server logs.")
    
    # Check for empty input.
    if not smiles:
        gr.Warning("Please enter a SMILES string!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    try:
        # Call the main prediction method.
        pred_df, internal_sim_df, external_sim_df = predictor.predict(smiles)
        gr.Info("Prediction complete!")
        return pred_df, internal_sim_df, external_sim_df
    except Exception as e:
        # Raise a Gradio error to the user if prediction fails.
        raise gr.Error(f"An error occurred during prediction: {e}")

def predict_and_generate_plots(smiles, ic50_threshold, ic50_group_by, sim_score_threshold, sim_group_by):
    """
    Runs the full prediction pipeline and generates all initial plots and data tables.
    This is the main function triggered after the user clicks 'Run Prediction'.
    """
    # 1. Run prediction and perform initial data cleaning.
    pred_df, sim_df, external_sim_df = predictor.predict(smiles)
    pred_df = pred_df.drop_duplicates()
    sim_df = sim_df.drop_duplicates(subset=['drug'], keep='first')
    external_sim_df = external_sim_df.drop_duplicates(subset=['drug'], keep='first')
    
    # 2. Define a default return value for cases where prediction fails.
    placeholder_fig = create_placeholder_fig()
    empty_buttons = [gr.update(visible=False)] * 5
    fail_return = (
        [], [], [], # Display DFs
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), # State DFs
        placeholder_fig, "", gr.update(visible=False), # Plot 1
        placeholder_fig, "", gr.update(visible=False), # Plot 2
        placeholder_fig, "", gr.update(visible=False), # Plot 3
        # Unified Scatter Plot (defaulting to empty internal view)
        "Internal Data (GDSC)", placeholder_fig, *empty_buttons, gr.update(visible=False), 
        1, "Page 1/1", gr.update(value="◀ Previous", interactive=False), gr.update(value="▶ Next", interactive=False), None, "..."
    )

    if pred_df.empty or sim_df.empty:
        return fail_return

    # 3. Format data for display in the Gradio DataFrames.
    pred_df_display = pred_df.copy()
    pred_df_display['pred_lnIC50'] = pred_df_display['pred_lnIC50'].apply(lambda x: f'{x:.3f}')
    pred_df_display.columns = ["Cell line", "Tissue", "Cancer Type", "Pred. lnIC50"]
    pred_df_display.fillna('-', inplace=True)
    
    sim_df_display = sim_df.copy()
    sim_df_display['Score'] = sim_df_display['Score'].apply(lambda x: f'{x:.3f}')
    sim_df_display = sim_df_display.rename(columns={'drug': 'Drug', 'targets': 'Target', 'pathway_name': 'Pathway', 'Score': 'Score'})
    sim_df_display = sim_df_display[['Drug', 'Target', 'Pathway', 'Score']]
    sim_df_display.fillna('-', inplace=True)
    
    external_sim_df_display = external_sim_df.copy()
    if not external_sim_df_display.empty:
        external_sim_df_display['Score'] = external_sim_df_display['Score'].apply(lambda x: f'{x:.3f}')
        external_sim_df_display = external_sim_df_display.rename(columns={'drug': 'Drug', 'targets': 'Target', 'moa': 'MOA', 'Score': 'Score'})
        external_sim_df_display = external_sim_df_display[['Drug', 'Target', 'MOA', 'Score']]
        external_sim_df_display.fillna('-', inplace=True)
    
    # 4. Generate the initial analysis plots.
    fig1, info1 = create_ic50_comp_plot(pred_df, ic50_threshold, ic50_group_by)
    fig2, info2 = create_ic50_dist_plot(pred_df, ic50_threshold, ic50_group_by)
    fig3, info3 = create_drug_similarity_dist_plot(sim_df, sim_score_threshold, sim_group_by)
    
    # 5. Generate the initial state for the unified scatter plot tab (defaulting to internal data).
    top_1_internal_drug = sim_df.iloc[0]['drug']
    fig4 = create_ic50_scatter_plot(pred_df, ic50_df, top_1_internal_drug)
    
    # Prepare button updates for the top 5 similar drugs.
    internal_top_5 = sim_df.head(5)['drug'].tolist()
    button_updates = [gr.update(value=d, visible=True, variant='primary' if i == 0 else 'secondary') for i, d in enumerate(internal_top_5)]
    button_updates += [gr.update(visible=False)] * (5 - len(internal_top_5))

    # Prepare pagination state.
    total_pages = math.ceil(len(sim_df) / 5)
    page_text = f"Page 1 / {total_pages}"
    next_interactive = total_pages > 1
    
    # Prepare drug info text for the selected drug.
    info = sim_df.iloc[0]
    info_text = f"**Drug:** {info['drug']} | **Target:** {info['targets']} | **Pathway:** {info['pathway_name']} | **Score:** {info['Score']:.3f}"

    # 6. Return all the necessary updates for the UI.
    return (
        # DataFrames for display
        pred_df_display.values.tolist(), sim_df_display.values.tolist(), external_sim_df_display.values.tolist(),
        # DataFrames for state
        pred_df, sim_df, external_sim_df,
        # Analysis plots
        fig1, info1, gr.update(visible=True),
        fig2, info2, gr.update(visible=True),
        fig3, info3, gr.update(visible=True), gr.update(visible=True),
        # Initial state for the Unified Scatter Plot Tab
        gr.update(visible=True), 
        fig4,
        *button_updates,
        gr.update(visible=True),
        1, # page_state
        page_text,
        gr.update(value="◀ Previous", interactive=False),
        gr.update(value="▶ Next", interactive=next_interactive),
        top_1_internal_drug, # selected_drug_state
        info_text
    )


def switch_scatter_source(data_source, pred_df, sim_df, external_sim_df):
    """
    Handles switching the data source (Internal vs. External) for the 
    IC50 correlation scatter plot.
    """
    # Determine which dataset to use based on the radio button selection.
    is_internal = "Internal" in data_source
    page = 1
    df_to_use = sim_df if is_internal else external_sim_df
    
    # If the selected dataset is empty, show a placeholder.
    if df_to_use is None or df_to_use.empty:
        return create_placeholder_fig("No data."), "No data", *[gr.update(visible=False)]*5, "Page 1/1", gr.update(interactive=False), gr.update(interactive=False), page, None

    # Get the top drug from the selected dataset to display initially.
    top_1_drug = df_to_use.iloc[0]['drug']
    
    # Generate the appropriate plot and info text.
    if is_internal:
        fig = create_ic50_scatter_plot(pred_df, ic50_df, top_1_drug)
        info = df_to_use.iloc[0]
        info_text = f"**Drug:** {info['drug']} | **Target:** {info['targets']} | **Pathway:** {info['pathway_name']} | **Score:** {info['Score']:.3f}"
    else:
        fig = create_external_ic50_scatter_plot(pred_df, drh_ic50_df, top_1_drug)
        info = df_to_use.iloc[0]
        info_text = f"**Drug:** {info['drug']} | **Target:** {info['targets']} | **MOA:** {info['moa']} | **Score:** {info['Score']:.3f}"

    # Update the top 5 drug buttons and pagination for the new data source.
    top_5_drugs = df_to_use.head(5)['drug'].tolist()
    button_updates = [gr.update(value=d, visible=True, variant='primary' if i == 0 else 'secondary') for i, d in enumerate(top_5_drugs)]
    button_updates += [gr.update(visible=False)] * (5 - len(top_5_drugs))
    total_pages = math.ceil(len(df_to_use) / 5)
    page_text = f"Page 1 / {total_pages}"
    
    # Return all updates to reset the scatter plot tab.
    return fig, info_text, *button_updates, page_text, gr.update(value="◀ Previous", interactive=False), gr.update(value="▶ Next", interactive=total_pages > 1), page, top_1_drug

def update_unified_scatter(selected_drug, data_source, pred_df, sim_df, external_sim_df, *all_buttons):
    """
    Updates the scatter plot when a user clicks on one of the top 5 drug buttons.
    """
    is_internal = "Internal" in data_source
    df_to_use = sim_df if is_internal else external_sim_df
    
    # Check if essential data is available.
    if not selected_drug or pred_df is None or df_to_use is None:
        raise gr.Error("Please run prediction and select a drug.")
    
    # Generate the plot and info text for the specifically selected drug.
    if is_internal:
        fig = create_ic50_scatter_plot(pred_df, ic50_df, selected_drug)
        info = df_to_use[df_to_use['drug'] == selected_drug].iloc[0]
        info_text = f"**Drug:** {info['drug']} | **Target:** {info['targets']} | **Pathway:** {info['pathway_name']} | **Score:** {info['Score']:.3f}"
    else:
        fig = create_external_ic50_scatter_plot(pred_df, drh_ic50_df, selected_drug)
        info = df_to_use[df_to_use['drug'] == selected_drug].iloc[0]
        info_text = f"**Drug:** {info['drug']} | **Target:** {info['targets']} | **MOA:** {info['moa']} | **Score:** {info['Score']:.3f}"

    # Update the button variants to highlight the selected one.
    button_updates = [gr.update(variant='primary' if btn_val == selected_drug else 'secondary') for btn_val in all_buttons]
    
    return fig, info_text, *button_updates, selected_drug

def change_unified_page(action, data_source, current_page, sim_df, external_sim_df, selected_drug):
    """
    Handles pagination for the top similar drug buttons in the scatter plot tab.
    """
    is_internal = "Internal" in data_source
    df_to_use = sim_df if is_internal else external_sim_df
    
    # Handle cases with no data.
    if df_to_use is None or df_to_use.empty:
        return current_page, *([gr.update(visible=False)]*5), "Page 1 / 1", False, False

    # Calculate the new page number.
    total_pages = math.ceil(len(df_to_use) / 5)
    new_page = current_page
    if action == "next" and current_page < total_pages: new_page += 1
    elif action == "prev" and current_page > 1: new_page -= 1
        
    # Get the drugs for the new page.
    start_index = (new_page - 1) * 5
    page_drugs = df_to_use.iloc[start_index : start_index + 5]['drug'].tolist()
    
    # Update button visibility, values, and variants.
    button_updates = [gr.update(value=d, visible=True, variant='primary' if d == selected_drug else 'secondary') for d in page_drugs]
    button_updates += [gr.update(visible=False)] * (5 - len(page_drugs))
        
    page_text = f"Page {new_page} / {total_pages}"
    
    # Return updates for the page number, buttons, and page info text.
    return new_page, *button_updates, page_text, gr.update(value="◀ Previous", interactive=new_page > 1), gr.update(value="▶ Next", interactive=new_page < total_pages)


def switch_drug_sim_source(data_source):
    """
    Updates the 'Group By' radio choices when the data source for the
    'Similar Drug Distribution' plot is changed.
    """
    # Use 'Target' and 'Pathway' for internal data.
    if "Internal" in data_source:
        return gr.update(choices=['Target', 'Pathway'], value='Target')
    # Use 'Target' and 'MOA' for external data.
    else: # External
        return gr.update(choices=['Target', 'MOA'], value='Target')

def update_drug_sim_dist_plot(data_source, sim_df, external_sim_df, score_threshold, group_by):
    """
    Generates or updates the 'Similar Drug Distribution' plot based on the
    selected data source and filtering/grouping options.
    """
    # Determine which dataset to use.
    is_internal = "Internal" in data_source
    df_to_use = sim_df if is_internal else external_sim_df
    
    # Correct the 'group_by' value if it's inconsistent with the data source
    # (e.g., user switches source while 'Pathway' is selected).
    if is_internal and group_by not in ['Target', 'Pathway']:
        group_by = 'Target'
    elif not is_internal and group_by not in ['Target', 'MOA']:
        group_by = 'Target'
        
    # Re-use the main plot creation function with the correct data and parameters.
    fig, info = create_drug_similarity_dist_plot(df_to_use, score_threshold, group_by)
        
    return fig, info


# --- UI Configuration Constants ---
INITIAL_TOP_N = 20      # Initial number of items in the drug composition chart.
LOAD_MORE_COUNT = 10    # Number of additional items to load when 'Load More' is clicked.

def create_ui():
    """
    Constructs and returns the complete Gradio UI, including layout and event handlers.
    """
    # Load custom CSS for styling.
    with open("./style.css", "r", encoding="utf-8") as f:
        custom_css = f.read()
            
    # --- UI Layout Definition ---
    # Use gr.Blocks for a custom and flexible layout.
    with gr.Blocks(theme=gr.themes.Default(), title="Drug Response Predictor", css=custom_css) as demo:
        # --- Persistent State Management ---
        # Use gr.State to store data across interactions, avoiding re-computation.
        pred_data_state = gr.State()         # Stores the main prediction DataFrame.
        sim_data_state = gr.State()          # Stores the internal similarity DataFrame.
        external_sim_data_state = gr.State() # Stores the external similarity DataFrame.
        
        # --- Application Header ---
        gr.Markdown(
            """
            # AI-based Anticancer Drug Response and Similar Drug Prediction
            Enter a molecular structure (SMILES), and the AI model will predict the drug response (lnIC50) across various cancer cell lines, 
            and analyze the response pattern similarity to known anticancer drugs.
            """
        )

        # --- Section 1: Training Data Overview ---
        # An expandable accordion to show visualizations of the training data.
        with gr.Accordion("Train Data Overview", open=True) as data_overview_accordion:
            with gr.Tabs():
                # --- Tab 1.1: Cell Line Data Composition ---
                with gr.TabItem("Cell Line Composition"):
                    with gr.Row():
                        # Display an interactive treemap of cell lines.
                        gr.Plot(value=create_interactive_cell_line_treemap(cell_line_df), show_label=False, elem_id='cell_line_treemap_container')
                    
                    with gr.Row():
                        # Display summary statistics for cell line data.
                        with gr.Column():
                            gr.Button(f"Total Sample: {total_samples}", elem_classes="info-button")
                        with gr.Column():
                            gr.Button(f"Total Tissue Types: {total_tissues}", elem_classes="info-button")
                        with gr.Column():
                            gr.Button(f"Total Disease Types: {total_cancer_types}", elem_classes="info-button")
                
                # --- Tab 1.2: Drug Data Composition ---
                with gr.TabItem("Drug Composition"):
                    with gr.Row():
                        # State to track the number of items shown in the chart.
                        pathway_top_n_state = gr.State(value=INITIAL_TOP_N)
                    
                        # Display a stacked bar chart of drug pathways and targets.
                        pathway_plot = gr.Plot(
                            value=create_stacked_bar_chart(
                                drug_df, 
                                main_axis='pathway_name', 
                                stack_by='targets',   
                                top_n_stack=INITIAL_TOP_N
                            ),
                            show_label=False,
                            elem_id='drug_composition_bar_container'
                        )
                    
                    with gr.Row(elem_id='info-panel-row'):
                        # Display summary statistics for drug data.
                        gr.Button(f"Total Unique Drugs: {total_drugs}", elem_classes="info-button")
                        gr.Button(f"Total Unique Targets: {total_targets}", elem_classes="info-button")
                        gr.Button(f"Total Unique Pathways: {total_pathways}", elem_classes="info-button")
                        
                        # Button to load more items into the chart.
                        load_more_pathways_btn = gr.Button("Load More", variant="secondary", elem_id="load-more-button")

                # --- Tab 1.3: Cell Line Embedding Visualization ---
                with gr.TabItem("Cell Line Embedding Analysis"):
                    gr.Markdown("#### 3D UMAP of Omics Embeddings (for Erlotinib)")
                    gr.Markdown("This plot shows how the model groups cell lines based on their omics data when interacting with Erlotinib. You can rotate the plot by clicking and dragging.")
                    with gr.Row():
                        # Display the initial 3D UMAP plot for cell lines.
                        initial_cell_line_umap_plot = create_3d_cell_line_umap_plot(
                            umap_df=omics_umap_df,
                            color_by='tissue',
                            hover_name='model_id',
                        )
                        cell_line_umap_plot = gr.Plot(value=initial_cell_line_umap_plot, show_label=False, elem_id='cell_line_umap_container')
                    
                    with gr.Row():
                        # Radio buttons to change the color encoding of the plot.
                        cell_line_umap_color_radio = gr.Radio(
                            choices=['tissue', 'cancer_type'], 
                            value='tissue', 
                            label="Color points by:"
                        )

                # --- Tab 1.4: Drug Embedding Visualization ---
                with gr.TabItem("Drug Embedding Analysis"):
                    gr.Markdown("#### 3D UMAP of Drug Embeddings")
                    gr.Markdown("This plot shows how the model groups different drugs based on their learned features. You can rotate the plot by clicking and dragging.")
                    with gr.Row():
                        # Display the initial 3D UMAP plot for drugs.
                        initial_drug_umap_plot = create_3d_drug_umap_plot(
                            umap_df=drug_umap_df, 
                            color_by='pathway_name'
                        )
                        drug_umap_plot = gr.Plot(value=initial_drug_umap_plot, show_label=False, elem_id='drug_umap_container')
                    
                    with gr.Row():
                        # Radio buttons to change the color encoding of the plot.
                        drug_umap_color_radio = gr.Radio(
                            choices=['pathway_name', 'targets'],
                            value='pathway_name',
                            label="Color by"
                        )
                        
        # --- Section 2: Main Prediction Interface ---
        with gr.Row(elem_id="main_container"):
            # --- Left Column: Input and Tabulated Results ---
            with gr.Column(scale=2, elem_classes=["left-column", "stretch-height"]):
                # Textbox for user to enter a SMILES string.
                smiles_input = gr.Textbox(
                    label="Enter SMILES of the drug to analyze",
                    placeholder="e.g., COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
                    info="Paste the SMILES string of the molecule here."
                )
                
                # Main button to trigger prediction, initially disabled.
                predict_button = gr.Button("🚀 Run Prediction", variant="primary", interactive=False)
                
                # Clickable examples for user convenience.
                gr.Examples(
                    examples=[
                        ["COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"], # Erlotinib
                    ],
                    inputs=smiles_input,
                    label="Example SMILES"
                )

                # Tabbed interface for molecular info and result tables.
                with gr.Tabs():
                    # --- Tab 2.1: Molecular Properties ---
                    with gr.TabItem("ℹ️ Molecular Desc."):
                        with gr.Column(visible=False) as mol_desc_panel:
                            with gr.Row():
                                # Display the 2D image of the molecule.
                                mol_image_output = gr.Image(
                                    show_label=False, 
                                    show_download_button=False,
                                    elem_id="mol_image_container",
                                    interactive=False,
                                    show_fullscreen_button=False
                                )
                            with gr.Row():
                                # Display basic molecular properties.
                                with gr.Column():
                                    gr.Markdown("### Basic Properties")
                                    formula_output = gr.Textbox(label="Chemical Formula", interactive=False)
                                    weight_output = gr.Textbox(label="Molecular Weight", interactive=False)
                                    can_smiles_output = gr.Textbox(label="Canonical SMILES", interactive=False)
                                    inchikey_output = gr.Textbox(label="InChiKey", interactive=False)
                                # Display physicochemical properties.
                                with gr.Column():
                                    gr.Markdown("### Physicochemical Properties")
                                    logp_output = gr.Textbox(label="LogP (Lipophilicity)", interactive=False)
                                    tpsa_output = gr.Textbox(label="Topological Polar Surface Area (TPSA)", interactive=False)
                                    hdonors_output = gr.Textbox(label="Hydrogen Bond Donors", interactive=False)
                                    hacceptors_output = gr.Textbox(label="Hydrogen Bond Acceptors", interactive=False)
                    
                    # --- Tab 2.2: IC50 Prediction Table ---
                    with gr.TabItem("📈 IC50 Pred."):
                        gr.Markdown("#### Predicted drug sensitivity (lnIC50) for each cancer cell line.")
                        ic50_output = gr.DataFrame(
                            headers=["Cell line", "Tissue", "Cancer Type", "Pred. lnIC50"],
                            datatype=["str", "str", "str", "str"],
                            wrap=True
                        )

                    # --- Tab 2.3: Similar Drugs (Internal) Table ---
                    with gr.TabItem("📊 [GDSC] Similar Drug Pred."):
                        gr.Markdown("#### List of existing anticancer drugs with similar response patterns.")
                        similarity_output = gr.DataFrame(
                            headers=["Drug", "Target", "Pathway", "Score"],
                            datatype=["str", "str", "str", "str"],
                            wrap=True
                        )

                    # --- Tab 2.4: Similar Drugs (External) Table ---
                    with gr.TabItem("📊 [DRH] Similar Drug Pred."):
                        gr.Markdown("#### Similar drugs from an external dataset not seen during training.")
                        external_similarity_output = gr.DataFrame(
                            headers=["Drug", "Target", "MOA", "Score"],
                            datatype=["str", "str", "str", "str"],
                            wrap=True
                        )

            # --- Right Column: Detailed Analysis and Visualizations ---
            with gr.Column(scale=3, elem_classes=["right-column", "stretch-height"]):
                with gr.Tabs() as analysis_tabs_group:
                    # --- Tab 3.1: IC50 Correlation Scatter Plot ---
                    with gr.TabItem("Target/Drug IC50 Correlation", elem_id="unified_ic50_correlation_tab"):
                        gr.Markdown("#### Correlation analysis with similar drugs")
                        
                        scatter_source_radio = gr.Radio(
                            choices=["Internal Data (GDSC)", "External Data (DRH)"],
                            value="Internal Data (GDSC)",
                            show_label=False,
                            elem_id="scatter_source_selector",
                            visible=False
                        )
                        
                        scatter_plot = gr.Plot(show_label=False, elem_id='unified_ic50_scatter_analysis_container')
                        
                        with gr.Column(visible=False) as scatter_controls:
                            selected_drug_info_md = gr.Markdown("Click on a drug to display detailed information here.")
                            
                            top_drug_buttons = []
                            with gr.Row():
                                for _ in range(5):
                                    btn = gr.Button(visible=False, scale=1)
                                    top_drug_buttons.append(btn)
                            
                            with gr.Row(elem_classes="pagination-row"):
                                page_state = gr.State(value=1)
                                selected_drug_state = gr.State(value=None)
                                
                                prev_button = gr.Button("◀ Previous", interactive=False, elem_classes="pagination-button", elem_id="ic50_correlation_drug_prev_button")
                                page_info_markdown = gr.Markdown("Page 1 / 1", elem_id="drug_page_info")
                                next_button = gr.Button("▶ Next", interactive=False, elem_classes="pagination-button", elem_id="ic50_correlation_drug_next_button")
                    
                    # --- Tab 3.2: Similar Drug Distribution ---
                    with gr.TabItem("Similar Drug Distribution", elem_id="drug_sim_dist_analysis_tab") as drug_tab:
                        gr.Markdown("#### Distribution of Similar Drugs (Score ≥ threshold)")
                    
                        drug_sim_source_radio = gr.Radio(
                            choices=["Internal Data (GDSC)", "External Data (DRH)"],
                            value="Internal Data (GDSC)",
                            show_label=False,
                            elem_id="drug_sim_source_selector",
                            visible=False
                        )
                    
                        with gr.Row():
                            drug_sim_dist_plot = gr.Plot(show_label=False, elem_id='drug_sim_dist_analysis_container')
                        
                        with gr.Column(visible=False) as drug_sim_dist_controls:
                            with gr.Row():
                                drug_sim_dist_info_output = gr.Markdown()
                            with gr.Row():
                                drug_sim_dist_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.8, label="Score Threshold")
                                drug_sim_group_by_radio = gr.Radio(choices=['Target', 'Pathway'], value='Target', label="Group By")
                                
                    # --- Tab 3.3: Sensitive Cell Line Composition ---
                    with gr.TabItem("Cell Line Composition"):
                        gr.Markdown("#### Composition of Sensitive Cell Lines (lnIC50 <= threshold)")
                        with gr.Row():
                            ic50_comp_plot = gr.Plot(show_label=False, elem_id='cell_line_ic50_comp_analysis_container')
                        
                        with gr.Column(visible=False) as ic50_comp_controls:
                            with gr.Row():
                                ic50_comp_info_output = gr.Markdown()
                            with gr.Row():
                                ic50_comp_slider = gr.Slider(minimum=1e-2, maximum=6.0, step=0.1, value=1.0, label="lnIC50 Threshold")
                                ic50_comp_group_by_radio = gr.Radio(choices=['Tissue', 'Cancer type'], value='Tissue', label="Group By")
                                
                    # --- Tab 3.4: Sensitive Cell Line Distribution ---
                    with gr.TabItem("Cell Line Distribution"):
                        gr.Markdown("#### Distribution of Sensitive Cell Lines (lnIC50 <= threshold)")
                        with gr.Row():
                            ic50_dist_plot = gr.Plot(show_label=False, elem_id='cell_line_ic50_dist_analysis_container')
                        
                        with gr.Column(visible=False) as ic50_dist_controls:
                            with gr.Row():
                                ic50_dist_info_output = gr.Markdown()
                            with gr.Row():
                                ic50_dist_slider = gr.Slider(minimum=1e-2, maximum=6.0, step=0.1, value=1.0, label="lnIC50 Threshold")
                                ic50_dist_group_by_radio = gr.Radio(choices=['Tissue', 'Cancer type'], value='Tissue', label="Group By")
        
        ## ------------------ Event Listeners and Handlers ------------------ ##
        
        # --- Event Handlers for "Train Data Overview" Section ---
        load_more_pathways_btn.click(
            fn=load_more_drug_targets,
            inputs=[pathway_top_n_state],
            outputs=[pathway_plot, pathway_top_n_state]
        )

        cell_line_umap_color_radio.change(
            fn=update_cell_line_umap_plot,
            inputs=cell_line_umap_color_radio,
            outputs=cell_line_umap_plot
        )

        drug_umap_color_radio.change(
            fn=update_drug_umap_plot,
            inputs=drug_umap_color_radio,
            outputs=drug_umap_plot
        )

        # --- Event Handlers for Main Prediction Flow ---
        # Validate SMILES input in real-time to enable/disable the predict button.
        smiles_input.change(
            fn=update_button_state,
            inputs=smiles_input,
            outputs=predict_button,
            trigger_mode="always_last" # Fires only after the user stops typing.
        )

        # Chain of events triggered by the 'Run Prediction' button.
        predict_button.click(
            fn=close_accordion_and_generate_desc, # First, close overview and show molecular info.
            inputs=smiles_input,
            outputs=[
                data_overview_accordion,
                mol_desc_panel, mol_image_output, 
                formula_output, weight_output, can_smiles_output, inchikey_output,
                logp_output, tpsa_output, hdonors_output, hacceptors_output
            ]
        ).then(
            fn=predict_and_generate_plots, # Then, run prediction and update all results.
            inputs=[smiles_input, 
                    ic50_comp_slider, ic50_comp_group_by_radio,
                    drug_sim_dist_slider, drug_sim_group_by_radio], 
            outputs=[
                ic50_output, similarity_output, external_similarity_output,
                pred_data_state, sim_data_state, external_sim_data_state,
                ic50_comp_plot, ic50_comp_info_output, ic50_comp_controls,
                ic50_dist_plot, ic50_dist_info_output, ic50_dist_controls,
                drug_sim_dist_plot, drug_sim_dist_info_output, drug_sim_dist_controls, drug_sim_source_radio,
                scatter_source_radio, scatter_plot, *top_drug_buttons, scatter_controls,
                page_state, page_info_markdown, prev_button, next_button,
                selected_drug_state, selected_drug_info_md
            ]
        )
        
        # --- Event Handlers for Interactive Analysis Plots ---
        # Link controls to the IC50 composition plot.
        ic50_comp_controls_list = [ic50_comp_slider, ic50_comp_group_by_radio]
        for control in ic50_comp_controls_list:
            event_method = control.release if isinstance(control, gr.Slider) else control.change
            event_method(
                fn=create_ic50_comp_plot,
                inputs=[pred_data_state, ic50_comp_slider, ic50_comp_group_by_radio],
                outputs=[ic50_comp_plot, ic50_comp_info_output]
            )

        # Link controls to the IC50 distribution plot.
        ic50_dist_controls_list = [ic50_dist_slider, ic50_dist_group_by_radio]
        for control in ic50_dist_controls_list:
            event_method = control.release if isinstance(control, gr.Slider) else control.change
            event_method(
                fn=create_ic50_dist_plot,
                inputs=[pred_data_state, ic50_dist_slider, ic50_dist_group_by_radio],
                outputs=[ic50_dist_plot, ic50_dist_info_output]
            )

        # --- [Modified] Event Handlers for "Similar Drug Distribution" ---
        # 1. Event when the data source radio button is changed.
        drug_sim_source_radio.change(
            fn=switch_drug_sim_source,
            inputs=[drug_sim_source_radio],
            outputs=[drug_sim_group_by_radio]
        ).then(
            # Then, immediately update the plot with the new source.
            fn=update_drug_sim_dist_plot,
            inputs=[drug_sim_source_radio, sim_data_state, external_sim_data_state, drug_sim_dist_slider, drug_sim_group_by_radio],
            outputs=[drug_sim_dist_plot, drug_sim_dist_info_output]
        )
        
        # 2. Event when the slider or group_by radio is changed.
        drug_sim_dist_controls_list = [drug_sim_dist_slider, drug_sim_group_by_radio]
        for control in drug_sim_dist_controls_list:
            event_method = control.release if isinstance(control, gr.Slider) else control.change
            event_method(
                fn=update_drug_sim_dist_plot, 
                inputs=[drug_sim_source_radio, sim_data_state, external_sim_data_state, drug_sim_dist_slider, drug_sim_group_by_radio],
                outputs=[drug_sim_dist_plot, drug_sim_dist_info_output]
            )

        # --- Event Handlers for IC50 Correlation Scatter Plot and Pagination ---
        scatter_source_radio.change(
            fn=switch_scatter_source,
            inputs=[scatter_source_radio, pred_data_state, sim_data_state, external_sim_data_state],
            outputs=[
                scatter_plot, selected_drug_info_md, *top_drug_buttons, page_info_markdown, 
                prev_button, next_button, page_state, selected_drug_state
            ]
        )

        # Event for clicking one of the top 5 drug buttons.
        for btn in top_drug_buttons:
            btn.click(
                fn=update_unified_scatter,
                inputs=[btn, scatter_source_radio, pred_data_state, sim_data_state, external_sim_data_state] + top_drug_buttons,
                outputs=[scatter_plot, selected_drug_info_md, *top_drug_buttons, selected_drug_state]
            )
        
        # Events for pagination buttons (Previous/Next).
        prev_button.click(
            fn=partial(change_unified_page, "prev"),
            inputs=[scatter_source_radio, page_state, sim_data_state, external_sim_data_state, selected_drug_state],
            outputs=[page_state, *top_drug_buttons, page_info_markdown, prev_button, next_button]
        )
        next_button.click(
            fn=partial(change_unified_page, "next"),
            inputs=[scatter_source_radio, page_state, sim_data_state, external_sim_data_state, selected_drug_state],
            outputs=[page_state, *top_drug_buttons, page_info_markdown, prev_button, next_button]
        )

        # --- UI Initialization on Page Load ---
        # Initialize all plot areas with placeholder figures.
        demo.load(
            fn=lambda: create_placeholder_fig(),
            inputs=None,
            outputs=[ic50_comp_plot]
        ).then(
            fn=lambda: create_placeholder_fig(),
            inputs=None,
            outputs=[ic50_dist_plot]
        ).then(
            fn=lambda: create_placeholder_fig(),
            inputs=None,
            outputs=[drug_sim_dist_plot]
        ).then(
            fn=lambda: create_placeholder_fig(),
            inputs=None,
            outputs=[scatter_plot]
        )
            
    return demo


if __name__ == '__main__':
    app = create_ui()
    app.launch(share=True, server_name="0.0.0.0")