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
# Attempt to load the DrugResponsePredictor class at app startup.
# This object encapsulates all models and data loading/processing logic.
try:
    # Instantiate the main predictor class. This will load all necessary models and data into memory.
    predictor = DrugResponsePredictor()
    PREDICTOR_LOADED = True
    print("Predictor initialized successfully.")
except Exception as e:
    # If initialization fails, set a flag and print an error message.
    # The UI will then be populated with empty components to prevent crashing.
    PREDICTOR_LOADED = False
    print(f"FATAL: Failed to load predictor. The application might not work as expected. Error: {e}")

# --- Data Loading and Summary Statistics ---
# All data required for the UI is sourced from the central 'predictor' object
# to ensure a single source of truth and prevent redundant data loading.
if PREDICTOR_LOADED:
    # Retrieve pre-loaded and pre-processed DataFrames from the predictor instance.
    cell_line_df = predictor.cell_all
    drug_df = predictor.drug_meta
    ic50_df = predictor.ic50_ref
    drh_ic50_df = predictor.drh_ic50_df
    drh_meta_df = predictor.drh_meta_df
    all_drh_drugs = drh_ic50_df.columns.tolist()

    # Calculate summary statistics for display in the "Train Data Overview" section.
    # Cell line statistics
    total_samples = len(cell_line_df)
    total_tissues = cell_line_df['tissue'].nunique()
    total_cancer_types = cell_line_df['cancer_type'].nunique()

    # Drug statistics
    total_targets = drug_df['targets'].nunique()
    total_pathways = drug_df['pathway_name'].nunique()
    total_drugs = drug_df['drug_name'].nunique()
    
else:
    # If the predictor failed to load, initialize all variables with empty/default values.
    # This ensures that the Gradio UI can still be built without crashing.
    cell_line_df = pd.DataFrame()
    drug_df = pd.DataFrame()
    ic50_df = pd.DataFrame()
    drh_ic50_df = pd.DataFrame()
    drh_meta_df = pd.DataFrame()
    all_drh_drugs = []
    
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
    Loads more drug targets by creating a new chart with an increased number of items.

    Args:
        current_n (int): The current number of items being displayed.

    Returns:
        tuple: A tuple containing the new plot object and the updated number of items.
    """
    # Increment the current count by a predefined constant.
    new_n = current_n + LOAD_MORE_COUNT
    # Create a new stacked bar chart with the updated number of items to display.
    new_plot = create_stacked_bar_chart(
        drug_df, 
        main_axis='pathway_name', 
        stack_by='targets', 
        top_n_stack=new_n
    )
    # Return the newly generated plot and the updated item count.
    return new_plot, new_n

def generate_molecular_description(smiles: str) -> Tuple:
    """
    Generates molecular information from a SMILES string and returns objects
    to update Gradio UI components.

    Args:
        smiles (str): The SMILES string of the molecule.

    Returns:
        tuple: A tuple of Gradio update objects for various UI components.
    """
    # Create a molecule object from the SMILES string using RDKit.
    mol = Chem.MolFromSmiles(smiles)

    # If the SMILES string is invalid, the molecule object will be None.
    if not mol:
        # In case of an error, make the info panel visible and display an error message.
        error_msg = f"Error: Invalid SMILES string."
        return (gr.update(visible=True), gr.update(value=None)) + (gr.update(value=error_msg),) + (gr.update(value=""),) * 5

    try:
        # --- Calculate molecular properties using RDKit ---
        can_smiles = Chem.MolToSmiles(mol, canonical=True)   # Generate the canonical SMILES string
        img = Draw.MolToImage(mol, size=(350, 350))          # Create a 2D image of the molecule
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)  # Calculate the molecular formula
        weight = f"{Descriptors.MolWt(mol):.2f} g/mol"       # Calculate molecular weight
        logp = f"{Descriptors.MolLogP(mol):.2f}"             # Calculate the octanol-water partition coefficient (LogP)
        tpsa = f"{Descriptors.TPSA(mol):.2f} Å²"             # Calculate the Topological Polar Surface Area
        h_donors = Lipinski.NumHDonors(mol)                  # Count the number of hydrogen bond donors
        h_acceptors = Lipinski.NumHAcceptors(mol)            # Count the number of hydrogen bond acceptors

        try:
            # Generate InChI and InChIKey, with error handling for molecules that might fail conversion.
            ich = inchi.MolToInchi(mol); inchikey = inchi.InchiToInchiKey(ich)
        except Exception:
            ich, inchikey = 'N/A', 'N/A'
        
        return (
            gr.update(visible=True),      # mol_desc_panel: Make the description panel visible
            gr.update(value=img),         # mol_image_output: Display the molecule image
            gr.update(value=formula),     # formula_output: Display the molecular formula
            gr.update(value=weight),      # weight_output: Display the molecular weight
            gr.update(value=can_smiles),  # can_smiles_output: Display the canonical SMILES
            gr.update(value=inchikey),    # inchikey_output: Display the InChIKey
            gr.update(value=logp),        # logp_output: Display the LogP value
            gr.update(value=tpsa),        # tpsa_output: Display the TPSA value
            gr.update(value=h_donors),    # hdonors_output: Display the H-bond donor count
            gr.update(value=h_acceptors)  # hacceptors_output: Display the H-bond acceptor count
        )
    except Exception as e:
        # Handle any other unexpected errors during property calculation.
        error_msg = f"Error calculating properties: {e}"
        return (gr.update(visible=True), gr.update(value=None)) + (gr.update(value=error_msg),) + (gr.update(value=""),) * 5

def close_accordion_and_generate_desc(smiles):
    """
    A wrapper function to generate molecular description and close an accordion UI element.

    Args:
        smiles (str): The SMILES string of the molecule.

    Returns:
        tuple: A tuple of Gradio update objects, including one for the accordion.
    """
    # 1. Call the original function to get the Gradio updates for the molecular description.
    desc_outputs = generate_molecular_description(smiles)
    
    # 2. Create a specific Gradio update to close the accordion UI element.
    accordion_update = gr.update(open=False)
    
    # 3. Combine the accordion update with the description updates and return as a single tuple.
    return (accordion_update,) + desc_outputs

def update_button_state(smiles_text):
    """
    Checks the validity of a SMILES string in real-time using RDKit
    to update the interactive state of a button.

    Args:
        smiles_text (str): The input SMILES string from a textbox.

    Returns:
        gr.update: A Gradio update object to set the button's 'interactive' property.
    """
    # If the input textbox is empty or contains only whitespace, disable the button.
    if not smiles_text or not smiles_text.strip():
        return gr.update(interactive=False)

    # 1. Attempt to create a molecule object from the SMILES string.
    #    Sanitization is enabled by default to check for chemical validity.
    mol = Chem.MolFromSmiles(smiles_text, sanitize=True)

    # 2. The button's interactivity is determined by whether the molecule object was successfully created.
    #    If 'mol' is not None, the SMILES is valid, and the button becomes interactive. Otherwise, it's disabled.
    return gr.update(interactive=(mol is not None))


def update_cell_line_umap_plot(color_by):
    """Handles updating the 3D UMAP plot for cell lines based on user selection."""
    # This function is called whenever the radio button for color encoding changes.
    return create_3d_cell_line_umap_plot(
        umap_df=omics_umap_df,
        color_by=color_by,  # The selected feature to color the points by (e.g., 'tissue')
        hover_name='model_id',
    )

def update_drug_umap_plot(color_by):
    """Handles updating the 3D UMAP plot for drugs based on user selection."""
    # This function is called whenever the radio button for color encoding changes.
    return create_3d_drug_umap_plot(
        umap_df=drug_umap_df, 
        color_by=color_by  # The selected feature to color the points by (e.g., 'pathway_name')
    )

def predict_wrapper(smiles):
    """
    A wrapper function for the prediction model, to be called by Gradio.
    Includes error handling for model loading and the prediction process.

    Args:
        smiles (str): The input SMILES string for prediction.

    Raises:
        gr.Error: If the model is not loaded or an error occurs during prediction.
        gr.Warning: If the input SMILES is empty.

    Returns:
        tuple: A tuple containing prediction and similarity DataFrames.
    """
    # First, check if the main predictor object was loaded successfully at startup.
    if not PREDICTOR_LOADED:
        raise gr.Error("Failed to load the model. Please check the server logs.")
    # Check if the user provided an input SMILES string.
    if not smiles:
        gr.Warning("Please enter a SMILES string!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        pred_df, internal_sim_df, external_sim_df = predictor.predict(smiles)
        gr.Info("Prediction complete!")
        return pred_df, internal_sim_df, external_sim_df
        
    except Exception as e:
        # Catch any exceptions during the prediction and raise a Gradio error to the user.
        raise gr.Error(f"An error occurred during prediction: {e}")

def predict_and_generate_plots(smiles, ic50_threshold, ic50_group_by, sim_score_threshold, sim_group_by):
    """
    Performs prediction, processes the results, and generates all initial plots and UI updates.
    This is the main function triggered after a successful prediction.

    Args:
        smiles (str): The input SMILES string.
        ic50_threshold (float): Threshold for IC50 plots.
        ic50_group_by (str): Column to group by for IC50 plots.
        sim_score_threshold (float): Threshold for similarity score plot.
        sim_group_by (str): Column to group by for similarity plot.

    Returns:
        tuple: A large tuple containing numerous Gradio update objects for dataframes,
               plots, buttons, visibility states, and other UI components.
    """
    # 1. Execute prediction and perform basic data cleaning.
    pred_df, sim_df, external_sim_df = predictor.predict(smiles)
    pred_df = pred_df.drop_duplicates()
    sim_df = sim_df.drop_duplicates(subset=['drug'], keep='first')
    external_sim_df = external_sim_df.drop_duplicates(subset=['drug'], keep='first')
    
    # Pre-define a complete set of return values for cases of prediction failure or empty results.
    placeholder_fig = create_placeholder_fig()
    empty_button_updates = [gr.update(visible=False)] * 5
    fail_return = (
        [], [], [],
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        placeholder_fig, "", gr.update(visible=False),
        placeholder_fig, "", gr.update(visible=False),
        placeholder_fig, "", gr.update(visible=False),
        placeholder_fig, *empty_button_updates, gr.update(visible=False), 1, "Page 1/1", False, False, None, "...",
        placeholder_fig, *empty_button_updates, gr.update(visible=False), 1, "Page 1/1", False, False, None, "..."
    )

    # If any of the essential dataframes are empty, return the pre-defined failure state.
    if pred_df.empty or sim_df.empty or external_sim_df.empty:
        return fail_return

    # --- 2. Format data for UI display ---
    # Create a copy for display to avoid altering the original data used for plotting.
    pred_df_display = pred_df.copy()
    pred_df_display['pred_lnIC50'] = pred_df_display['pred_lnIC50'].apply(lambda x: f'{x:.3f}') # Format numbers
    pred_df_display.columns = ["Cell line", "Tissue", "Cancer Type", "Pred. lnIC50"] # Rename columns
    pred_df_display.fillna('-', inplace=True) # Replace NaN with a dash
    
    # Format the internal similarity dataframe for display.
    sim_df_display = sim_df.copy()
    sim_df_display['Score'] = sim_df_display['Score'].apply(lambda x: f'{x:.3f}')
    sim_df_display = sim_df_display.rename(columns={'drug': 'Drug', 'targets': 'Target', 'pathway_name': 'Pathway', 'Score': 'Score'})
    sim_df_display = sim_df_display[['Drug', 'Target', 'Pathway', 'Score']]
    sim_df_display.fillna('-', inplace=True)

    # Format the external similarity dataframe for display.
    external_sim_df_display = external_sim_df.copy()
    external_sim_df_display['Score'] = external_sim_df_display['Score'].apply(lambda x: f'{x:.3f}')
    external_sim_df_display = external_sim_df_display.rename(columns={'drug': 'Drug', 'targets': 'Target', 'moa': 'MOA', 'Score': 'Score'})
    external_sim_df_display = external_sim_df_display[['Drug', 'Target', 'MOA', 'Score']]
    external_sim_df_display.fillna('-', inplace=True)

    # --- 3. Generate analysis plots for different tabs ---
    fig1, info1 = create_ic50_comp_plot(pred_df, ic50_threshold, ic50_group_by)
    fig2, info2 = create_ic50_dist_plot(pred_df, ic50_threshold, ic50_group_by)
    fig3, info3 = create_drug_similarity_dist_plot(sim_df, sim_score_threshold, sim_group_by)
    
    # --- 4. Initialize the Internal Similar Drugs scatter plot and buttons ---
    # Select the most similar drug to display initially.
    top_1_internal_drug = sim_df.iloc[0]['drug']
    fig4 = create_ic50_scatter_plot(pred_df, ic50_df, top_1_internal_drug)
    
    # Prepare the first page of drug buttons (top 5).
    internal_top_5_drugs = sim_df.head(5)['drug'].tolist()
    internal_button_updates = []
    for i in range(5):
        if i < len(internal_top_5_drugs):
            drug_name = internal_top_5_drugs[i]
            # Highlight the first button as 'primary'.
            variant = 'primary' if i == 0 else 'secondary'
            internal_button_updates.append(gr.update(value=drug_name, visible=True, variant=variant))
        else:
            # Hide buttons if there are fewer than 5 drugs.
            internal_button_updates.append(gr.update(visible=False))

    # Set up pagination controls for internal drugs.
    internal_total_pages = math.ceil(len(sim_df) / 5)
    internal_page_info_text = f"Page 1 / {internal_total_pages}"
    internal_next_interactive = internal_total_pages > 1
    
    # Create the initial information text for the top drug.
    top_1_drug_info = sim_df.iloc[0]
    internal_initial_info_text = (
        f"**Drug:** {top_1_drug_info['drug']} | "
        f"**Target:** {top_1_drug_info['targets']} | "
        f"**Pathway:** {top_1_drug_info['pathway_name']} | "
        f"**Score:** {top_1_drug_info['Score']:.3f}"
    )

    # --- 5. Initialize the External Similar Drugs scatter plot and buttons ---
    top_1_external_drug = external_sim_df.iloc[0]['drug']
    fig5 = create_external_ic50_scatter_plot(pred_df, drh_ic50_df, top_1_external_drug)
    
    # Prepare the first page of external drug buttons.
    external_top_5_drugs = external_sim_df.head(5)['drug'].tolist()
    external_button_updates = []
    for i in range(5):
        if i < len(external_top_5_drugs):
            drug_name = external_top_5_drugs[i]
            variant = 'primary' if i == 0 else 'secondary'
            external_button_updates.append(gr.update(value=drug_name, visible=True, variant=variant))
        else:
            external_button_updates.append(gr.update(visible=False))

    # Set up pagination controls for external drugs.
    external_total_pages = math.ceil(len(external_sim_df) / 5)
    external_page_info_text = f"Page 1 / {external_total_pages}"
    external_next_interactive = external_total_pages > 1
    
    # Create initial info text for the top external drug.
    top_1_external_drug_info = external_sim_df.iloc[0]
    external_initial_info_text = (
        f"**Drug:** {top_1_external_drug_info['drug']} | "
        f"**Target:** {top_1_external_drug_info['targets']} | "
        f"**MOA:** {top_1_external_drug_info['moa']} | "
        f"**Score:** {top_1_external_drug_info['Score']:.3f}"
    )

    # --- 6. Return all updates as a single large tuple for Gradio ---
    return (
        # DataFrames for tables
        pred_df_display.values.tolist(), sim_df_display.values.tolist(), external_sim_df_display.values.tolist(),
        # Raw DataFrames to be stored in gr.State for later use
        pred_df, sim_df, external_sim_df,
        # Analysis plots and their visibility
        fig1, info1, gr.update(visible=True),
        fig2, info2, gr.update(visible=True),
        fig3, info3, gr.update(visible=True),
        # Internal scatter plot and its controls
        fig4, *internal_button_updates, gr.update(visible=True), 1, internal_page_info_text, 
        gr.update(value="◀ Previous", interactive=False), gr.update(value="▶ Next", interactive=internal_next_interactive), top_1_internal_drug, internal_initial_info_text,
        # External scatter plot and its controls
        fig5, *external_button_updates, gr.update(visible=True), 1, external_page_info_text, 
        gr.update(value="◀ Previous", interactive=False), gr.update(value="▶ Next", interactive=external_next_interactive), top_1_external_drug, external_initial_info_text
    )


def update_scatter_and_buttons(selected_drug, pred_df, sim_df, *all_button_values):
    """
    Updates the IC50 scatter plot and button styles when a new drug is selected from the internal list.

    Args:
        selected_drug (str): The name of the drug selected by the user.
        pred_df (pd.DataFrame): The dataframe of IC50 predictions.
        sim_df (pd.DataFrame): The dataframe of similar drugs.
        *all_button_values: The current value (name) of each drug button on the page.

    Returns:
        tuple: A tuple containing the updated plot, button style updates, info text, and the selected drug name.
    """
    # 1. Ensure data is present before creating the plot.
    if not selected_drug or pred_df is None:
        raise gr.Error("Please run a prediction and select a drug first.")
    # Create the scatter plot for the newly selected drug.
    fig = create_ic50_scatter_plot(pred_df, ic50_df, selected_drug)

    # 2. Generate a list of Gradio updates for the button styles.
    # The selected button becomes 'primary', all others become 'secondary'.
    button_updates = [gr.update(variant='primary' if btn_val == selected_drug else 'secondary') for btn_val in all_button_values]

    # 3. Update the information text with the selected drug's details from the similarity dataframe.
    drug_info = sim_df[sim_df['drug'] == selected_drug].iloc[0]
    info_text = (
        f"**Drug:** {drug_info['drug']} | "
        f"**Target:** {drug_info['targets']} | "
        f"**Pathway:** {drug_info['pathway_name']} | "
        f"**Score:** {drug_info['Score']:.3f}"
    )
    
    # 4. Return the new plot and the list of button updates.
    return fig, *button_updates, info_text, selected_drug

    
def update_external_scatter_and_buttons(selected_drug, pred_df, external_sim_df, *all_button_values):
    """
    Updates the scatter plot, buttons, and info text for the EXTERNAL similar drugs tab.
    Uses the external_sim_df stored in a gr.State component.
    """
    # 1. Ensure data is present before proceeding.
    if not selected_drug or pred_df is None or external_sim_df is None:
        raise gr.Error("Please run a prediction and select a drug first.")
        
    # 2. Create the scatter plot using the external reference IC50 data.
    fig = create_external_ic50_scatter_plot(pred_df, drh_ic50_df, selected_drug)

    # 3. Update button styles, highlighting the selected drug.
    button_updates = [gr.update(variant='primary' if btn_val == selected_drug else 'secondary') for btn_val in all_button_values]
    
    # 4. Look up all information for the selected drug from the external_sim_df.
    drug_info_row = external_sim_df[external_sim_df['drug'] == selected_drug]

    if not drug_info_row.empty:
        info = drug_info_row.iloc[0]
        # Retrieve Score, Target, and MOA directly from the external_sim_df.
        score = info.get('Score', 0.0) # Using .get for safety, though 'Score' should exist
        target = info.get('targets', 'N/A')
        moa = info.get('moa', 'N/A')
        
        info_text = (
            f"**Drug:** {selected_drug} | "
            f"**Target:** {target} | "
            f"**MOA:** {moa} | "
            f"**Score:** {score:.3f}"
        )
    else:
        # Fallback text if the drug info is somehow not found.
        info_text = (
            f"**Selected Drug:** {selected_drug} | "
            "(No metadata found)"
        )

    # 5. Return all updates.
    return fig, *button_updates, info_text, selected_drug


def change_drug_page(action, current_page, sim_df, selected_drug):
    """
    Handles pagination for the list of internal similar drugs.

    Args:
        action (str): The action to perform ("next" or "prev").
        current_page (int): The current page number.
        sim_df (pd.DataFrame): The dataframe of similar drugs.
        selected_drug (str): The currently selected drug to maintain its 'primary' style.

    Returns:
        tuple: Updates for the page number, buttons, page info text, and pagination button interactivity.
    """
    # Handle cases where the similarity dataframe is not available (e.g., before first prediction).
    if sim_df is None or sim_df.empty:
        return current_page, *([gr.update(visible=False)] * 5), "Page 1 / 1", gr.update(interactive=False), gr.update(interactive=False)
    
    total_drugs = len(sim_df)
    total_pages = math.ceil(total_drugs / 5)
    
    # 1. Calculate the new page number based on the action ('prev' or 'next').
    new_page = current_page
    if action == "next" and current_page < total_pages:
        new_page += 1
    elif action == "prev" and current_page > 1:
        new_page -= 1
        
    # 2. Slice the dataframe to get the list of drugs for the new page.
    start_index = (new_page - 1) * 5
    end_index = start_index + 5
    page_drugs = sim_df.iloc[start_index:end_index]['drug'].tolist()
    
    # 3. Create a list of updates for the 5 drug buttons on the page.
    button_updates = []
    for i in range(5):
        if i < len(page_drugs):
            drug_name = page_drugs[i]
            # Maintain the 'primary' style if a button corresponds to the currently selected drug.
            variant_style = 'primary' if drug_name == selected_drug else 'secondary'
            update = gr.update(value=drug_name, visible=True, variant=variant_style)
        else:
            # Hide any buttons that are not needed for this page.
            update = gr.update(value="", visible=False)
        button_updates.append(update)
        
    # 4. Update the page information text (e.g., "Page 2 / 5").
    page_info_text = f"Page {new_page} / {total_pages}"
    
    # 5. Update the interactivity of the 'Previous' and 'Next' buttons.
    prev_interactive = new_page > 1
    next_interactive = new_page < total_pages
    
    return new_page, *button_updates, page_info_text, gr.update(interactive=prev_interactive), gr.update(interactive=next_interactive)


def change_external_drug_page(action, current_page, external_sim_df, selected_drug):
    """
    Handles pagination for the EXTERNAL list of similar drugs.
    """
    # Handle cases where the dataframe is not available.
    if external_sim_df is None or external_sim_df.empty:
        return current_page, *([gr.update(visible=False)] * 5), "Page 1 / 1", gr.update(interactive=False), gr.update(interactive=False)
    
    # Calculate total pages based on the length of the external similarity dataframe.
    total_drugs = len(external_sim_df)
    total_pages = math.ceil(total_drugs / 5)
    
    # Calculate the new page number.
    new_page = current_page
    if action == "next" and current_page < total_pages: new_page += 1
    elif action == "prev" and current_page > 1: new_page -= 1
        
    # Get the list of drugs for the new page from the external similarity dataframe.
    start_index = (new_page - 1) * 5
    end_index = start_index + 5
    page_drugs = external_sim_df.iloc[start_index:end_index]['drug'].tolist()
    
    # Create updates for the drug buttons.
    button_updates = []
    for i in range(5):
        if i < len(page_drugs):
            drug_name = page_drugs[i]
            variant_style = 'primary' if drug_name == selected_drug else 'secondary'
            update = gr.update(value=drug_name, visible=True, variant=variant_style)
        else:
            update = gr.update(value="", visible=False)
        button_updates.append(update)
        
    # Update page info text and pagination button states.
    page_info_text = f"Page {new_page} / {total_pages}"
    prev_interactive = new_page > 1
    next_interactive = new_page < total_pages
    
    return new_page, *button_updates, page_info_text, gr.update(interactive=prev_interactive), gr.update(interactive=next_interactive)

# --- UI Configuration Constants ---
# Defines constants for configuring the Gradio user interface.
INITIAL_TOP_N = 20 # The initial number of items to display in the drug composition chart.
LOAD_MORE_COUNT = 10 # The number of additional items to load when the "Load More" button is clicked.

def create_ui():
    """
    Constructs and returns the complete Gradio user interface for the application.
    This function defines the layout, components, and all event handling logic.
    """
    # Load custom CSS from an external file to apply custom styles to the UI.
    with open("./style.css", "r", encoding="utf-8") as f:
        custom_css = f.read()
            
    # --- UI Layout Definition ---
    # Define the main UI structure using Gradio Blocks for a flexible and custom layout.
    with gr.Blocks(theme=gr.themes.Default(), title="Drug Response Predictor", css=custom_css) as demo:
        # --- Persistent State Management ---
        # Use gr.State to store data that needs to persist across user interactions.
        # This avoids re-running expensive predictions when only UI elements are updated.
        pred_data_state = gr.State() # Stores the main prediction DataFrame.
        sim_data_state = gr.State() # Stores the internal similarity DataFrame.
        external_sim_data_state = gr.State() # Stores the external similarity DataFrame.
        
        # --- Application Header ---
        # Display the main title and a brief description of the application using Markdown.
        gr.Markdown(
            """
            # AI-based Anticancer Drug Response and Similar Drug Prediction
            Enter a molecular structure (SMILES), and the AI model will predict the drug response (lnIC50) across various cancer cell lines, 
            and analyze the response pattern similarity to known anticancer drugs.
            """
        )

        # --- Section 1: Training Data Overview ---
        # An expandable accordion to show visualizations of the underlying training data.
        with gr.Accordion("Train Data Overview", open=True) as data_overview_accordion:
            with gr.Tabs():
                # Tab 1.1: Cell Line Data Composition
                with gr.TabItem("Cell Line Composition"):
                    with gr.Row():
                        # Display an interactive treemap visualizing the hierarchy of cell lines.
                        gr.Plot(value=create_interactive_cell_line_treemap(cell_line_df), show_label=False, elem_id='cell_line_treemap_container')
                    
                    with gr.Row():
                        # Display key summary statistics about the cell line data.
                        with gr.Column():
                            gr.Button(f"Total Sample: {total_samples}", elem_classes="info-button")
                        with gr.Column():
                            gr.Button(f"Total Tissue Types: {total_tissues}", elem_classes="info-button")
                        with gr.Column():
                            gr.Button(f"Total Disease Types: {total_cancer_types}", elem_classes="info-button")
                
                # Tab 1.2: Drug Data Composition
                with gr.TabItem("Drug Composition"):
                    with gr.Row():
                        # A state to keep track of the number of items shown in the pathway chart.
                        pathway_top_n_state = gr.State(value=INITIAL_TOP_N)
                    
                        # Display a stacked bar chart of drug pathways and their targets.
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
                        # Display key summary statistics about the drug data.
                        gr.Button(f"Total Unique Drugs: {total_drugs}", elem_classes="info-button")
                        gr.Button(f"Total Unique Targets: {total_targets}", elem_classes="info-button")
                        gr.Button(f"Total Unique Pathways: {total_pathways}", elem_classes="info-button")
                        
                        # A button to load more items into the chart above.
                        load_more_pathways_btn = gr.Button("Load More", variant="secondary", elem_id="load-more-button")

                # Tab 1.3: Cell Line Embedding Visualization
                with gr.TabItem("Cell Line Embedding Analysis"):
                    gr.Markdown("#### 3D UMAP of Omics Embeddings (for Erlotinib)")
                    gr.Markdown("This plot shows how the model groups cell lines based on their omics data when interacting with Erlotinib. You can rotate the plot by clicking and dragging.")
                    with gr.Row():
                        # Generate and display the initial 3D UMAP plot for cell lines.
                        initial_cell_line_umap_plot = create_3d_cell_line_umap_plot(
                            umap_df=omics_umap_df,
                            color_by='tissue',
                            hover_name='model_id',
                        )
                        cell_line_umap_plot = gr.Plot(value=initial_cell_line_umap_plot, show_label=False, elem_id='cell_line_umap_container')
                    
                    with gr.Row():
                        # Radio buttons to dynamically change the color encoding of the plot.
                        cell_line_umap_color_radio = gr.Radio(
                            choices=['tissue', 'cancer_type'], 
                            value='tissue', 
                            label="Color points by:"
                        )

                # Tab 1.4: Drug Embedding Visualization
                with gr.TabItem("Drug Embedding Analysis"):
                    gr.Markdown("#### 3D UMAP of Drug Embeddings")
                    gr.Markdown("This plot shows how the model groups different drugs based on their learned features. You can rotate the plot by clicking and dragging.")
                    with gr.Row():
                        # Generate and display the initial 3D UMAP plot for drugs.
                        initial_drug_umap_plot = create_3d_drug_umap_plot(
                            umap_df=drug_umap_df, 
                            color_by='pathway_name'
                        )
                        drug_umap_plot = gr.Plot(value=initial_drug_umap_plot, show_label=False, elem_id='drug_umap_container')
                    
                    with gr.Row():
                        # Radio buttons to change the color encoding of the drug plot.
                        drug_umap_color_radio = gr.Radio(
                            choices=['pathway_name', 'targets'],
                            value='pathway_name',
                            label="Color by"
                        )

                        
        # --- Section 2: Main Prediction Interface ---
        # A container for the primary input and output sections, arranged side-by-side.
        with gr.Row(elem_id="main_container"):
            # --- Left Column: Input and Tabulated Results ---
            with gr.Column(scale=2, elem_classes=["left-column", "stretch-height"]):
                # Textbox for the user to enter a drug's SMILES string.
                smiles_input = gr.Textbox(
                    label="Enter SMILES of the drug to analyze",
                    placeholder="e.g., COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
                    info="Paste the SMILES string of the molecule here."
                )
                
                # The main button to trigger the prediction, initially disabled.
                predict_button = gr.Button("🚀 Run Prediction", variant="primary", interactive=False)
                
                # Provide clickable example SMILES strings for user convenience.
                gr.Examples(
                    examples=[
                        ["COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"], # Erlotinib
                    ],
                    inputs=smiles_input,
                    label="Example SMILES"
                )

                # A tabbed interface for displaying molecular info and prediction data tables.
                with gr.Tabs():
                    # Tab 2.1: Detailed Molecular Properties
                    with gr.TabItem("ℹ️ Molecular Desc."):
                        with gr.Column(visible=False) as mol_desc_panel:
                            with gr.Row():
                                # Display the 2D image of the input molecule.
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
                                # Display physicochemical properties (related to Lipinski's rules).
                                with gr.Column():
                                    gr.Markdown("### Physicochemical Properties")
                                    logp_output = gr.Textbox(label="LogP (Lipophilicity)", interactive=False)
                                    tpsa_output = gr.Textbox(label="Topological Polar Surface Area (TPSA)", interactive=False)
                                    hdonors_output = gr.Textbox(label="Hydrogen Bond Donors", interactive=False)
                                    hacceptors_output = gr.Textbox(label="Hydrogen Bond Acceptors", interactive=False)
                    
                    # Tab 2.2: IC50 Prediction Results Table
                    with gr.TabItem("📈 IC50 Pred."):
                        gr.Markdown("#### Predicted drug sensitivity (lnIC50) for each cancer cell line.")
                        ic50_output = gr.DataFrame(
                            headers=["Cell line", "Tissue", "Cancer Type", "Pred. lnIC50"],
                            datatype=["str", "str", "str", "str"],
                            wrap=True,
                        )

                    # Tab 2.3: Similar Drug (Internal) Results Table
                    with gr.TabItem("📊 Similar Drug Pred."):
                        gr.Markdown("#### List of existing anticancer drugs with similar response patterns.")
                        similarity_output = gr.DataFrame(
                            headers=["Drug", "Target", "Pathway", "Score"],
                            datatype=["str", "str", "str", "str"],
                            wrap=True,
                        )

                    # Tab 2.4: Similar Drug (External/Unseen) Results Table
                    with gr.TabItem("📊 [Unseen] Similar Drug Pred."):
                        gr.Markdown("#### Similar drugs from an external dataset not seen during training.")
                        external_similarity_output = gr.DataFrame(
                            headers=["Drug", "Target", "MOA", "Score"],
                            datatype=["str", "str", "str", "str"],
                            wrap=True,
                        )

            # --- Right Column: Detailed Analysis and Visualizations ---
            with gr.Column(scale=3, elem_classes=["right-column", "stretch-height"]):
                with gr.Tabs() as analysis_tabs_group:
                    # Tab 3.1: IC50 Correlation Scatter Plot (Internal)
                    with gr.TabItem("Target/Drug IC50 Correlation", elem_id="ic50_correlation_tab"):
                        gr.Markdown("#### Correlation analysis with similar drugs")
                        with gr.Row():
                            ic50_scatter_plot = gr.Plot(show_label=False, elem_id='drug_ic50_scatter_analysis_container')

                        # A container for plot controls, initially hidden and made visible after prediction.
                        with gr.Column(visible=False) as ic50_corrlation_scatter_controls:
                            with gr.Row():
                                selected_drug_info_md = gr.Markdown("Click on a drug to display detailed information here.")
                            # Buttons to select one of the top similar drugs for comparison.
                            top_drug_buttons = []
                            with gr.Row(elem_id="ic50_drug_select_button"):
                                for _ in range(5):
                                    btn = gr.Button(visible=False, scale=1, min_width=100)
                                    top_drug_buttons.append(btn)
                            # Pagination controls to navigate through the full list of similar drugs.
                            with gr.Row(elem_classes="pagination-row"):
                                drug_page_state = gr.State(value=1)
                                selected_drug_state = gr.State(value=None)
                                
                                prev_drug_button = gr.Button("◀ Previous", interactive=False, elem_id="ic50_correlation_drug_prev_button", elem_classes="pagination-button")
                                page_info_markdown = gr.Markdown("Page 1 / 1", elem_id="drug_page_info")
                                next_drug_button = gr.Button("▶ Next", interactive=False, elem_id="ic50_correlation_drug_next_button", elem_classes="pagination-button")

                    # Tab 3.2: IC50 Correlation Scatter Plot (External)
                    with gr.TabItem("[Unseen] Target/Drug IC50 Correlation", elem_id="external_ic50_correlation_tab"):
                        gr.Markdown("#### Correlation analysis with similar drugs from an external dataset")
                        with gr.Row():
                            external_ic50_scatter_plot = gr.Plot(show_label=False, elem_id='external_drug_ic50_scatter_analysis_container')

                        # Controls for the external drug correlation plot.
                        with gr.Column(visible=False) as external_ic50_corrlation_scatter_controls:
                            with gr.Row():
                                external_selected_drug_info_md = gr.Markdown("Click on a drug to display detailed information here.")
                            external_top_drug_buttons = []
                            with gr.Row(elem_id="external_ic50_drug_select_button"):
                                for _ in range(5):
                                    external_btn = gr.Button(visible=False, scale=1, min_width=100)
                                    external_top_drug_buttons.append(external_btn)
                            with gr.Row(elem_classes="pagination-row"):
                                external_drug_page_state = gr.State(value=1)
                                external_selected_drug_state = gr.State(value=None)
                                
                                external_prev_drug_button = gr.Button("◀ Previous", interactive=False, elem_id="external_ic50_correlation_drug_prev_button", elem_classes="pagination-button")
                                external_page_info_markdown = gr.Markdown("Page 1 / 1", elem_id="external_drug_page_info")
                                external_next_drug_button = gr.Button("▶ Next", interactive=False, elem_id="external_ic50_correlation_drug_next_button", elem_classes="pagination-button")
                    
                    # Tab 3.3: Distribution of Similar Drugs
                    with gr.TabItem("Similar Drug Distribution") as drug_tab:
                        gr.Markdown("#### Distribution of Similar Drugs (Score ≥ threshold)")
                        with gr.Row():
                            drug_sim_dist_plot = gr.Plot(show_label=False, elem_id='drug_sim_dist_analysis_container')
                        # Interactive controls for filtering and grouping the plot.
                        with gr.Column(visible=False) as drug_sim_dist_controls:
                            with gr.Row():
                                drug_sim_dist_info_output = gr.Markdown()
                            with gr.Row():
                                drug_sim_dist_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.8, label="Score Threshold")
                                drug_sim_group_by_radio = gr.Radio(choices=['Target', 'Pathway'], value='Target', label="Group By")
                                
                    # Tab 3.4: Composition of Sensitive Cell Lines
                    with gr.TabItem("Cell Line Composition"):
                        gr.Markdown("#### Composition of Sensitive Cell Lines (lnIC50 <= threshold)")
                        with gr.Row():
                            ic50_comp_plot = gr.Plot(show_label=False, elem_id='cell_line_ic50_comp_analysis_container')
                        # Interactive controls for filtering and grouping the plot.
                        with gr.Column(visible=False) as ic50_comp_controls:
                            with gr.Row():
                                ic50_comp_info_output = gr.Markdown()
                            with gr.Row():
                                ic50_comp_slider = gr.Slider(minimum=1e-2, maximum=6.0, step=0.1, value=1.0, label="lnIC50 Threshold")
                                ic50_comp_group_by_radio = gr.Radio(choices=['Tissue', 'Cancer type'], value='Tissue', label="Group By")
                                
                    # Tab 3.5: Distribution of Sensitive Cell Lines
                    with gr.TabItem("Cell Line Distribution"):
                        gr.Markdown("#### Distribution of Sensitive Cell Lines (lnIC50 <= threshold)")
                        with gr.Row():
                            ic50_dist_plot = gr.Plot(show_label=False, elem_id='cell_line_ic50_dist_analysis_container')
                        # Interactive controls for filtering and grouping the plot.
                        with gr.Column(visible=False) as ic50_dist_controls:
                            with gr.Row():
                                ic50_dist_info_output = gr.Markdown()
                            with gr.Row():
                                ic50_dist_slider = gr.Slider(minimum=1e-2, maximum=6.0, step=0.1, value=1.0, label="lnIC50 Threshold")
                                ic50_dist_group_by_radio = gr.Radio(choices=['Tissue', 'Cancer type'], value='Tissue', label="Group By")
        
        ## ------------------ Event Listeners and Handlers ------------------ ##
        # This section connects UI components (inputs) to functions (outputs).
        
        # --- Event Handlers for "Train Data Overview" Section ---
        # Event for the "Load More" button in the drug composition tab.
        load_more_pathways_btn.click(
            fn=load_more_drug_targets,
            inputs=[pathway_top_n_state],
            outputs=[pathway_plot, pathway_top_n_state]
        )

        # Update the cell line UMAP plot when the color radio button is changed.
        cell_line_umap_color_radio.change(
            fn=update_cell_line_umap_plot,
            inputs=cell_line_umap_color_radio,
            outputs=cell_line_umap_plot
        )

        # Update the drug UMAP plot when the color radio button is changed.
        drug_umap_color_radio.change(
            fn=update_drug_umap_plot,
            inputs=drug_umap_color_radio,
            outputs=drug_umap_plot
        )

        # --- Event Handlers for Main Prediction Flow ---
        # Validate the SMILES input in real-time to enable/disable the predict button.
        # This provides immediate feedback to the user about the validity of their input.
        smiles_input.change(
            fn=update_button_state,
            inputs=smiles_input,
            outputs=predict_button,
            trigger_mode="always_last" # Ensures the event fires only after the user stops typing.
        )

        # A chain of events triggered by the main "Run Prediction" button.
        predict_button.click(
            fn=close_accordion_and_generate_desc, # First, close the overview and show molecular properties.
            inputs=smiles_input,
            outputs=[
                data_overview_accordion,
                mol_desc_panel, mol_image_output, 
                formula_output, weight_output, can_smiles_output, inchikey_output,
                logp_output, tpsa_output, hdonors_output, hacceptors_output
            ]
        ).then(
            fn=predict_and_generate_plots, # Then, run the prediction and update all result components.
            # Inputs include the SMILES string and the initial values from the analysis controls.
            inputs=[smiles_input, 
                    ic50_comp_slider, ic50_comp_group_by_radio,
                    drug_sim_dist_slider, drug_sim_group_by_radio], 
            # Outputs update all data tables, plots, controls, states, and pagination.
            outputs=[
                # Data tables
                ic50_output, similarity_output, external_similarity_output,
                # Persistent states
                pred_data_state, sim_data_state, external_sim_data_state,
                # Analysis plots and their controls
                ic50_comp_plot, ic50_comp_info_output, ic50_comp_controls,
                ic50_dist_plot, ic50_dist_info_output, ic50_dist_controls,
                drug_sim_dist_plot, drug_sim_dist_info_output, drug_sim_dist_controls,
                
                # Internal IC50 scatter plot and its controls
                ic50_scatter_plot, *top_drug_buttons, ic50_corrlation_scatter_controls,
                drug_page_state, page_info_markdown, 
                prev_drug_button, next_drug_button,
                selected_drug_state, selected_drug_info_md,
                
                # External IC50 scatter plot and its controls
                external_ic50_scatter_plot, *external_top_drug_buttons, external_ic50_corrlation_scatter_controls,
                external_drug_page_state, external_page_info_markdown,
                external_prev_drug_button, external_next_drug_button,
                external_selected_drug_state, external_selected_drug_info_md
            ]
        )
        
        # --- Event Handlers for Interactive Analysis Plots ---
        # A loop to link controls (slider, radio) to the IC50 composition plot.
        ic50_comp_controls_list = [ic50_comp_slider, ic50_comp_group_by_radio]
        for control in ic50_comp_controls_list:
            # Use '.release' for sliders (fires when user lets go) and '.change' for radios.
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

        # Link controls to the drug similarity distribution plot.
        drug_sim_dist_controls_list = [drug_sim_dist_slider, drug_sim_group_by_radio]
        for control in drug_sim_dist_controls_list:
            event_method = control.release if isinstance(control, gr.Slider) else control.change
            event_method(
                fn=create_drug_similarity_dist_plot,
                inputs=[sim_data_state, drug_sim_dist_slider, drug_sim_group_by_radio],
                outputs=[drug_sim_dist_plot, drug_sim_dist_info_output]
            )

        # --- Event Handlers for Internal Drug Scatter Plot and Pagination ---
        # Link each of the top 5 drug selection buttons to update the scatter plot.
        for btn in top_drug_buttons:
            btn.click(
                fn=update_scatter_and_buttons,
                inputs=[btn, pred_data_state, sim_data_state] + top_drug_buttons,
                outputs=[ic50_scatter_plot] + top_drug_buttons + [selected_drug_info_md, selected_drug_state]
            )

        # Event for the "Previous" pagination button.
        prev_drug_button.click(
            fn=partial(change_drug_page, "prev"), # Use functools.partial to pass the action "prev".
            inputs=[drug_page_state, sim_data_state, selected_drug_state],
            outputs=[drug_page_state, *top_drug_buttons, page_info_markdown, prev_drug_button, next_drug_button]
        )
        # Event for the "Next" pagination button.
        next_drug_button.click(
            fn=partial(change_drug_page, "next"), # Use functools.partial to pass the action "next".
            inputs=[drug_page_state, sim_data_state, selected_drug_state],
            outputs=[drug_page_state, *top_drug_buttons, page_info_markdown, prev_drug_button, next_drug_button]
        )

        # --- Event Handlers for External Drug Scatter Plot and Pagination ---
        # Link each of the top 5 external drug selection buttons to update the scatter plot.
        for btn in external_top_drug_buttons:
            btn.click(
                fn=update_external_scatter_and_buttons,
                inputs=[btn, pred_data_state, external_sim_data_state] + external_top_drug_buttons,
                outputs=[external_ic50_scatter_plot] + external_top_drug_buttons + [external_selected_drug_info_md, external_selected_drug_state]
            )

        # Event for the "Previous" external pagination button.
        external_prev_drug_button.click(
            fn=partial(change_external_drug_page, "prev"),
            inputs=[external_drug_page_state, external_sim_data_state, external_selected_drug_state],
            outputs=[external_drug_page_state, *external_top_drug_buttons, external_page_info_markdown, external_prev_drug_button, external_next_drug_button]
        )
        # Event for the "Next" external pagination button.
        external_next_drug_button.click(
            fn=partial(change_external_drug_page, "next"),
            inputs=[external_drug_page_state, external_sim_data_state, external_selected_drug_state],
            outputs=[external_drug_page_state, *external_top_drug_buttons, external_page_info_markdown, external_prev_drug_button, external_next_drug_button]
        )

        # --- UI Initialization on Page Load ---
        # When the UI first loads, initialize all plot areas with placeholder figures.
        # This provides a clean initial state before any predictions are run.
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
            outputs=[ic50_scatter_plot]
        ).then(
            fn=lambda: create_placeholder_fig(),
            inputs=None,
            outputs=[external_ic50_scatter_plot]
        )
        
    return demo


if __name__ == '__main__':
    app = create_ui()
    app.launch(share=True, server_name="0.0.0.0")