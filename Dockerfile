
# Dockerfile for ONCO-CDRP

# 1. Select base image
# Using NVIDIA CUDA 11.8 + cuDNN 8 + Ubuntu 20.04 development image.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 2. Set environment variables (UTF-8, disable Python buffering, etc.)
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ARG CONDA_ENV_NAME=cdpr_demo_env
ENV CONDA_ENV_NAME=${CONDA_ENV_NAME}

# 3. Update system and install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    bzip2 \
    openssl \
    supervisor && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Install Miniconda
ENV CONDA_DIR /opt/conda
ENV MINICONDA_INSTALLER_SCRIPT_NAME "Miniconda3-latest-Linux-x86_64.sh"

RUN echo "Starting Miniconda installation..." && \
    echo "Downloading ${MINICONDA_INSTALLER_SCRIPT_NAME} from anaconda.com..." && \
    wget --no-verbose "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER_SCRIPT_NAME}" -O ~/miniconda.sh && \
    echo "Download complete. Verifying downloaded file (optional step)..." && \
    echo "Installing Miniconda to ${CONDA_DIR}..." && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    echo "Miniconda installation finished. Removing installer script..." && \
    rm ~/miniconda.sh && \
    echo "Cleaning up Conda installation..." && \
    ${CONDA_DIR}/bin/conda clean -a -y && \
    echo "Conda cleanup complete. Creating symlink for conda shell functions..." && \
    ln -s ${CONDA_DIR}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "Miniconda setup complete."

# 5. Add Conda bin directory to PATH
ENV PATH $CONDA_DIR/bin:$PATH

# 6. Copy Conda environment definition file
COPY environment.yml .

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 7. Create Conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# For interactive shells (docker exec) auto-activation
RUN { \
    echo '. /opt/conda/etc/profile.d/conda.sh'; \
    echo "conda activate ${CONDA_ENV_NAME}"; \
    } >> ~/.bashrc

# 8. SSL Certificate Generation for HTTPS
RUN mkdir -p /opt/certs && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /opt/certs/mykey.key \
    -out /opt/certs/mycert.pem \
    -subj "/C=KR/ST=Seoul/L=Seoul/O=MyPersonalServer/CN=localhost"


# 9. Setup custom entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# 10. Set working directory (inside the container)
WORKDIR /workspace

# 9. Copy Supervisor config and scripts
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY scripts/ /scripts/
RUN chmod +x /scripts/*.sh

# 10. Expose both Jupyter and Gradio ports
EXPOSE 8888
EXPOSE 7860

# 11. Default command to run Supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]