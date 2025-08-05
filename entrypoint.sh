#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Initialize Conda shell functions for the current shell
eval "$(/opt/conda/bin/conda shell.bash hook)"

conda activate ${CONDA_ENV_NAME}

echo "EntryPoint: Conda environment '${CONDA_ENV_NAME}' activated."
echo "EntryPoint: Executing CMD: $@"

# Execute the command passed as CMD from the Dockerfile
exec "$@"