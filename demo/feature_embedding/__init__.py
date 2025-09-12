# feature_embedding/__init__.py
# This file makes the 'feature_embedding' directory a Python package and
# exposes its main public functions and classes for easy importing.

# Expose the main feature generation function from the integrator module.
from .integrator import generate_drug_features

# Expose the DTI predictor class from the dti module.
from .dti import DtiPredictor