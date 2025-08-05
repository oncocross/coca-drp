#!/bin/bash

# --- User-defined Variables ---
# IMPORTANT: Please modify the variables below with your actual values!

# Absolute path to your project code directory on the host machine
# Example: HOST_PROJECT_CODE_PATH="/home/user/my_project_code"
HOST_PROJECT_CODE_PATH=""

# Absolute path to your project data directory on the host machine (can be the same as code path if data is within)
# Example: HOST_PROJECT_DATA_PATH="/home/user/my_datasets"
HOST_PROJECT_DATA_PATH="" # Leave empty if no separate data volume is needed

# Name and tag of the Docker image you built
# Example: DOCKER_IMAGE_NAME="my_docker_img:latest"
DOCKER_IMAGE_NAME=""

# Name for the running Docker container (you can change this)
# If a container with this name already exists, remove it first (docker rm -f <name>) or use a different name.
# Example: CONTAINER_NAME="my_docker_container"
CONTAINER_NAME=""

# Host port to map to Jupyter's port 8888 inside the container
HOST_JUPYTER_PORT="" 
HOST_GRADIO_PORT=""

# --- Basic Sanity Checks (Optional but recommended) ---
if [ -z "${HOST_PROJECT_CODE_PATH}" ] || \
   [ -z "${DOCKER_IMAGE_NAME}" ] || \
   [ -z "${CONTAINER_NAME}" ] || \
   [ -z "${HOST_JUPYTER_PORT}" ]; then
    echo "Error: Please ensure HOST_PROJECT_CODE_PATH, DOCKER_IMAGE_NAME, CONTAINER_NAME, and HOST_JUPYTER_PORT are set in the script."
    exit 1
fi

# --- Build docker run command options ---
DOCKER_RUN_ARGS=(
    "-d"
    "--gpus" "all"
    "--name" "${CONTAINER_NAME}"
    "-v" "${HOST_PROJECT_CODE_PATH}:/workspace"
)

# Add data volume mount option only if the HOST_PROJECT_DATA_PATH variable is set
if [ -n "${HOST_PROJECT_DATA_PATH}" ]; then
  DOCKER_RUN_ARGS+=("-v" "${HOST_PROJECT_DATA_PATH}:/data")
fi

# Add Jupyter port mapping option only if the HOST_JUPYTER_PORT variable is set
if [ -n "${HOST_JUPYTER_PORT}" ]; then
  DOCKER_RUN_ARGS+=("-p" "${HOST_JUPYTER_PORT}:8888")
fi

# Add Jupyter port mapping option only if the HOST_JUPYTER_PORT variable is set
if [ -n "${HOST_GRADIO_PORT}" ]; then
  DOCKER_RUN_ARGS+=("-p" "${HOST_GRADIO_PORT}:7860")
fi


# --- Run Docker Container ---
echo "--- Starting Docker Container for Jupyter & Gradio ---"
echo "Image Name: ${DOCKER_IMAGE_NAME}"
echo "Container Name: ${CONTAINER_NAME}"
echo "Host Code Path: ${HOST_PROJECT_CODE_PATH}   ---> /workspace (Container)"
echo "Jupyter Port (Host): ${HOST_JUPYTER_PORT} ---> 8888 (Container)"
echo "Gradio Port (Host): ${HOST_GRADIO_PORT} ---> 7860 (Container)"
echo "---------------------------------"

# Execute the command
echo "Executing: docker run ${DOCKER_RUN_ARGS[*]} ${DOCKER_IMAGE_NAME}"
docker run "${DOCKER_RUN_ARGS[@]}" "${DOCKER_IMAGE_NAME}"

echo "-----------------------------------------------------"
echo "Docker container ${CONTAINER_NAME} is starting..."
echo "Access Jupyter at: https://<your_server_ip_or_localhost>:${HOST_JUPYTER_PORT}"
echo "Access Gradio at:  https://<your_server_ip_or_localhost>:${HOST_GRADIO_PORT}"
echo "To see logs for both services: docker logs -f ${CONTAINER_NAME}"
echo "To get a shell inside: docker exec -it ${CONTAINER_NAME} bash"
echo "To stop: docker stop ${CONTAINER_NAME}"