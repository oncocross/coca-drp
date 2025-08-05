#!/bin/bash
set -e
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

echo "Starting Gradio App on port 7860..."

exec python /workspace/app.py \
  --server_name 0.0.0.0 \
  --server_port 7860 \
  --ssl-keyfile /opt/certs/mykey.key \
  --ssl-certfile /opt/certs/mycert.pem