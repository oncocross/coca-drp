#!/bin/bash
set -e
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

echo "Starting Jupyter Lab on port 8888..."
exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.root_dir=/workspace \
  --ServerApp.certfile=/opt/certs/mycert.pem \
  --ServerApp.keyfile=/opt/certs/mykey.key