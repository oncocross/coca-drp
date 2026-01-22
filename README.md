# 🧬 COCA-DRP: AI-based Anticancer Drug Response Predictor

> **⚡Try the Live Demo:** [Click Here to Access the Web Service](https://ed9db735935b1fe8bb.gradio.live)

An interactive web application that predicts cancer drug response (lnIC50) across various cell lines based on a drug's molecular structure (SMILES). It also identifies known anticancer drugs with similar response patterns from both internal (GDSC) and external (DRH) datasets.

<br>

## ✨ Features

* **Drug Response Prediction**: Input a SMILES string to get predicted lnIC50 values across a wide range of cancer cell lines.
* **Similar Drug Identification**: Discovers drugs with similar cell line response patterns from both internal (GDSC) and external (DRH) databases.
* **Interactive Data Visualization**:
    * **IC50 Correlation Analysis**: Scatter plots comparing the predicted lnIC50 profile of your compound against known drugs.
    * **Drug Similarity Distribution**: Bar charts showing the distribution of similar drugs, grouped by Target, Pathway, or MOA.
    * **Cell Line Analysis**: Donut and violin plots to explore the composition and distribution of sensitive cell lines.
* **Molecular Property Analysis**: Automatically calculates and displays key physicochemical properties and generates a 2D image of the input molecule.
* **Training Data Overview**: Provides visualizations of the underlying training data, including treemaps of cell line composition and 3D UMAP plots of drug/cell line embeddings.

<br>

## 📂 Data Preparation (Required)
Before building the container, please download the necessary datasets and model weights.

- **Download Link:** [Google Drive Link Here](https://drive.google.com/drive/folders/1FOCi12zA83n_CRGv9reS5FX96U1nw9Er?usp=sharing)
- **Instructions:**
  1. Download the files from the link above.
  2. Unzip/Place the files into the `demo/data/` directory within this repository. <br>
     *(Note: Ensure the file structure matches the Project Structure diagram above.)*
<br>

## 🛠️ Project Structure

The project is organized into a modular architecture to ensure clarity, maintainability, and scalability.

```
.demo
├── main.py                     # Main application entry point
├── app_setup.py                # Handles data loading and model initialization
├── predictor.py                # Core prediction pipeline class
├── layout/                     # Gradio UI layout modules
│   ├── __init__.py
│   ├── analysis_panel.py
│   ├── data_overview_panel.py
│   └── io_panel.py
|
├── callbacks/                  # Gradio event handler modules
├── feature_embedding/          # Feature engineering modules (Morgan, GIN, DTI)
├── model/                      # PyTorch model definitions (CDRModel, layers)
├── visualizations/             # Plotting utility functions
├── static/                     # CSS stylesheets
├── dataset/                    # PyTorch dataset classes
├── weights/                    # Pre-trained model weights
└── data/                       # Required datasets (Place downloaded files here)
```

<br>

## 🚀 Getting Started

This application is designed to be run in a Docker container to ensure a consistent and isolated environment.

### Prerequisites

* **Docker**: Ensure Docker is installed on your system.
* **(Optional) NVIDIA GPU**: For GPU acceleration, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) must be installed.

### Installation & Running

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/oncocross/coca-drp.git
    cd coca-drp
    ```

2.  **Build the Docker Image:** From the root of the project directory, run the following command to build the Docker image. This may take some time.
    ```bash
    docker build -t <your_image_name> .
    ```

3.  **Configure and Run the Application:** template script is provided to easily run the container while mounting the necessary data and weights directories.

    a. **Copy the template script**
    ```bash
    cp run_docker_template.sh run_docker.sh
    ```

    b. **Edit the script** <br>
    ```bash
    # Modify these variables in run_docker.sh
    HOST_PROJECT_CODE_PATH="/home/user/my_project_code"
    HOST_PROJECT_DATA_PATH="/home/user/my_datasets"
    DOCKER_IMAGE_NAME="your_image_name:latest"
    CONTAINER_NAME="your_container_name"
    HOST_JUPYTER_PORT="your_port_num" 
    HOST_GRADIO_PORT="your_port_num"
    ```

    c. **Build the Docker Container** <br>
    ```bash
    chmod +x run_docker.sh
    ./run_docker.sh
    ```

    d. **Connect to the container's shell**
    ```bash
    docker exec -it <your_container_name> bash
    ```
    
    e. **Run the Application (Inside the Container):**
    ```bash
    cd demo
    python main.py
    ```

<br>

## 🔧 How It Works

1.  **Input**: The user provides a drug's molecular structure as a SMILES string.
2.  **Feature Engineering (`feature_embedding/`)**: The SMILES string is converted into multiple feature vectors (Morgan Fingerprint, GIN embedding, DTI profile).
3.  **Prediction (`predictor.py`)**: The `CDRModel` (`model/`) takes the drug features and pre-loaded cell line omics data to predict lnIC50 values for each cell line.
4.  **Similarity Analysis**: The predicted lnIC50 profile is compared against reference profiles from GDSC and DRH datasets to find drugs with similar response patterns.
5.  **Visualization (`visualizations/`)**: The results are rendered as interactive plots and tables in the Gradio UI (`layout/` and `callbacks/`).
