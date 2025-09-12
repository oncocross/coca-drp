# app_setup.py

# --- Core and Data Handling Libraries ---
import pandas as pd
import joblib

# --- Custom Application Modules ---
# Import the main predictor class.
from predictor import DrugResponsePredictor

def load_app_data() -> dict:
    """
    Handles all startup data loading and predictor initialization.
    Returns a single dictionary containing all necessary data.
    """
    # Initialize a dictionary to hold all application data.
    app_data = {
        "predictor": None,
        "is_predictor_loaded": False,
        "dataframes": {},
        "umap_data": {},
        "stats": {}
    }

    # 1. Initialize the main predictor which contains the ML models.
    try:
        app_data["predictor"] = DrugResponsePredictor()
        app_data["is_predictor_loaded"] = True
    except Exception as e:
        print(f"FATAL: Failed to load predictor. Error: {e}")
        return app_data

    # 2. Extract primary dataframes from the predictor instance.
    predictor = app_data["predictor"]
    app_data["dataframes"] = {
        "cell_lines": predictor.cell_all,
        "drugs_meta": predictor.drug_meta,
        "ic50_reference": predictor.ic50_ref,
        "drh_ic50": predictor.drh_ic50_df,
        "drh_meta": predictor.drh_meta_df
    }

    # 3. Pre-calculate summary statistics for the UI.
    cell_line_df = app_data["dataframes"]["cell_lines"]
    drug_df = app_data["dataframes"]["drugs_meta"]
    app_data["stats"] = {
        "total_samples": len(cell_line_df),
        "total_tissues": cell_line_df['tissue'].nunique(),
        "total_cancer_types": cell_line_df['cancer_type'].nunique(),
        "total_targets": drug_df['targets'].nunique(),
        "total_pathways": drug_df['pathway_name'].nunique(),
        "total_drugs": drug_df['drug_name'].nunique()
    }

    # 4. Load pre-computed UMAP data.
    try:
        app_data["umap_data"]["omics"] = joblib.load('./data/erlotinib_omics_interact_3d_umap.joblib')
        app_data["umap_data"]["drugs"] = joblib.load('./data/drug_3d_umap_with_meta.joblib')
        print("UMAP data loaded successfully.")
    except FileNotFoundError:
        print("Warning: UMAP data file not found.")
        app_data["umap_data"]["omics"] = None
        app_data["umap_data"]["drugs"] = None

    return app_data


# --- Global App State ---
# The load_app_data function is executed once when this module is first imported.
APP_DATA = load_app_data()

# Unpack the dictionary into module-level constants for easy import by other modules.
PREDICTOR = APP_DATA["predictor"]
IS_PREDICTOR_LOADED = APP_DATA["is_predictor_loaded"]
STATS = APP_DATA.get("stats", {})

DF_CELL_LINES = APP_DATA["dataframes"].get("cell_lines", pd.DataFrame())
DF_DRUGS = APP_DATA["dataframes"].get("drugs_meta", pd.DataFrame())
DF_IC50 = APP_DATA["dataframes"].get("ic50_reference", pd.DataFrame())
DF_DRH_IC50 = APP_DATA["dataframes"].get("drh_ic50", pd.DataFrame())

DF_OMICS_UMAP = APP_DATA["umap_data"].get("omics")
DF_DRUG_UMAP = APP_DATA["umap_data"].get("drugs")