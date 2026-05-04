# Robust Indoor Localization via Wi-Fi CSI with a Custom Multilayer Perceptron

## Abstract
This repository presents an implementation of an indoor localization system leveraging Wi-Fi Channel State Information (CSI) and a custom-built Multilayer Perceptron (MLP) for precise position estimation. The methodology involves efficient data loading, spatial fingerprinting, and rigorous evaluation against traditional machine learning baselines such as k-Nearest Neighbors (k-NN), Random Forest, Ridge Regression, Support Vector Regressor (SVR), and Decision Trees. The code, primarily developed in a Google Colab environment, demonstrates a from-scratch NumPy implementation of the MLP and robust data handling techniques for large CSI datasets. Performance analysis includes error vector mapping, and RMSE comparisons across varying numbers of Access Points (APs) and spatial grid resolutions.

## 1. Introduction
Accurate indoor localization remains a critical challenge with diverse applications ranging from IoT devices to location-based services. This project explores the use of Wi-Fi CSI, which provides richer environmental information than traditional Received Signal Strength (RSS), as input features for machine learning models. A core contribution is the detailed, from-scratch implementation of an MLP using pure NumPy, allowing for deep insight into the model's mechanics and direct comparison with well-established supervised-learning based baseline models.

## 2. Methodology

### 2.1 Data Acquisition and Preprocessing
Our system utilizes the **WiLoc dataset** (specifically, the `Building2` subset), a massive measured dataset of Wi-Fi CSI. Due to the large file sizes, a memory-efficient loader is implemented using `h5py` for MATLAB v7.3 files and `scipy.io` as a fallback for older formats. Raw CSI data (real and imaginary parts) is converted into magnitude features.

### 2.2 Spatial Fingerprinting
Measurements are aggregated into spatial `GRID_SIZE` blocks (e.g., 0.5m) to form unique location fingerprints. The system identifies and focuses on the `MAX_APS` (e.g., 60) most frequently heard Access Points (APs) to construct a consistent feature vector representing the CSI profile at each grid location. This process involves estimating AP locations for visualization purposes (centroid method, not used in ML training).

### 2.3 Data Splitting and Normalization
The aggregated dataset is split into training (70%), validation (15%), and test (15%) sets. Crucially, input features are normalized (mean-zero, unit-variance) using statistics derived *only* from the training set to prevent data leakage.

### 2.4 Proposed Multilayer Perceptron (MLP)
An MLP model is implemented from scratch using pure NumPy. It consists of multiple `Linear`, `ReLU`, `Dropout`, and `BatchNorm1d` layers. An `AdamOptimizer` is also implemented manually for parameter updates. The model predicts 2D coordinates (X, Y) based on the CSI fingerprints.

### 2.5 Baseline Models
For comparative analysis, several traditional machine learning regressors are employed:
*   **k-Nearest Neighbors (k-NN) Regressor**
*   **Random Forest Regressor**
*   **Ridge Regression**
*   **Support Vector Regressor (SVR)** with RBF kernel
*   **Decision Tree Regressor**

### 2.6 Evaluation Metrics
Model performance is primarily assessed using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) in meters.

## 3. Results
The custom NumPy MLP consistently outperforms traditional machine learning baselines across various configurations. Detailed plots demonstrate:
*   **Learning Curves:** Training and validation MSE/RMSE over epochs.
*   **Error Vector Map:** Visual representation of prediction errors on sampled test data.
*   **Performance vs. AP Density:** RMSE comparison of models as the number of tracked APs varies (e.g., 10 to 60 APs, with fixed 0.5m grid size).
*   **Performance vs. Grid Resolution:** RMSE comparison as the spatial grid size varies (e.g., 0.1m, 0.3m, 0.5m, with fixed 30 APs).

## 4. Getting Started

### 4.1 Prerequisites
*   Python 3.x
*   Google Colab environment (recommended for GPU access and easy Drive mounting)
*   Required Python libraries:
    *   `numpy`
    *   `pandas`
    *   `scipy`
    *   `h5py`
    *   `torch`
    *   `scikit-learn`
    *   `matplotlib`
    *   `matplot2tikz` (for LaTeX/TikZ figure generation)

### 4.2 Installation
All necessary Python libraries can be installed via pip. If running in Google Colab, some may already be present, but it's good practice to ensure all are installed:
```bash
pip install numpy pandas scipy h5py torch scikit-learn matplotlib matplot2tikz
```

### 4.3 Data Download
The project relies on the **WiLoc dataset**. Please refer to the official paper for download instructions:

*   **Paper:** "WiLoc: Massive Measured Dataset of Wi-Fi Channel State Information with Application to Machine-Learning Based Localization" (Accepted by IEEE INFOCOM 2026)
*   **Paper Link:** [Arxiv Version Link](https://arxiv.org/abs/2602.09115)
*   **Dataset Link:** [Link for Downloading Dataset](https://forms.gle/8Z1zUVeF9ssfKR4r9)

After downloading, extract the `Building2_CSI.mat` and `Building2_Metadata.mat` files into a directory named `WiLoc_Data` within your Google Drive's root, or adjust the `BASE_PATH` variable in the notebook accordingly.

### 4.4 Running the Code (in Google Colab)

1.  **Open the Jupyter Notebook:** Upload `EE559_Project_Codes.ipynb` to Google Colab.
2.  **Download and Upload Dataset:** Before proceeding, download the WiLoc dataset (`Building2_CSI.mat` and `Building2_Metadata.mat`) using the link provided in Section 4.3. Then, upload these files to a directory named `WiLoc_Data` in your Google Drive.
3.  **Mount Google Drive:** Execute the cell under the `# Load Data` section to mount your Google Drive. This is crucial for accessing the `WiLoc_Data` folder.
4.  **Install `matplot2tikz`:** Run the `pip install matplot2tikz` cell to ensure all plotting utilities are available.
5.  **Load and Preprocess Data:** Execute the cells in the `# Load Data` section, including the `Smart & Memory-Efficient Loader` and the `Spatial Fingerprinting` sections. Pay attention to `GRID_SIZE` and `MAX_APS` in the `Configuration` cell, as these directly influence the dataset construction.
6.  **Prepare Data for ML:** Run the data splitting and normalization cells to get `X_train_scaled`, `y_train`, etc., and the PyTorch DataLoaders.
7.  **Visualize Dataset:** Execute the cells under `# Dataset Visualization` to see the measurement trajectory, grid centers, and estimated AP locations.
8.  **Train Proposed MLP:** Run all cells under `# Proposed MLP: From-Scratch Implementation with Pure NumPy`. This will train the implemented custom MLP, evaluate it on the test set, and generate performance plots (MSE, RMSE, Error Map).
9.  **Train Baselines:** Execute the cells under `# Baselines` to train and evaluate the traditional ML models for comparison.
10. **Generate Results Plots:** Run the cells under `# Results vs number of APs` and `# Results vs Grid Size`. **Note:** The data (`N_AP`, `MLP_RMSE_test`, etc.) in these sections is pre-collected. To reproduce these results, you would need to manually adjust `MAX_APS` or `GRID_SIZE` in the `Configuration` cell (and potentially in the `Spatial Fingerprinting` cell) and re-run the entire MLP and Baselines sections for each configuration, then collect the `final_test_rmse` values.


## 5. File Structure
```
.  
├── EE559_Project_Codes.ipynb  # The main Jupyter Notebook
├── README.md                 # This file
└── WiLoc_Data/               # Directory for dataset files (should be uploaded in Google Drive)
    ├── Building2_CSI.mat
    └── Building2_Metadata.mat

```

## 6. Contact
For any questions or suggestions, please open an issue in this repository or contact [fadakarg@usc.edu].
