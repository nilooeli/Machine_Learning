# Unsupervised Learning Algorithms

This folder holds hands-on notebooks demonstrating common
unsupervised-learning techniques in Python:

- **K-Means Clustering** (`KMeans_clustering.ipynb`)  
- **Hierarchical (Agglomerative) Clustering** (`Hierarchical_clustering.ipynb`)  
- **DBSCAN** (`DBSCAN_demo.ipynb`)  
- **Principal Component Analysis (PCA)** (`PCA_dimensionality_reduction.ipynb`)  
- **t-SNE Visualization** (`tSNE_visualization.ipynb`)  

---

## How to Run

1. Clone the **root** repository and change into this folder:  
   ```bash
   
   git clone https://github.com/nilooeli/Machine_Learning.git
   cd Machine_Learning/Unsupervised_Learning_algorithms

2. Create a virtual environment and install requirements:
   ```bash
   
   python -m venv venv
   source venv/bin/activate    # or `venv\Scripts\activate` on Windows
   pip install -r ../requirements.txt

3. Lauch Jupyter and open any notebook:
   ```bash

   jupyter notebook
   

## Requirements

See [../requirements.txt](../requirements.txt) for all packages.

## 1. Breast Cancer Scaling Effect

*Notebook:* [breast_cancer_scaling_effect.ipynb](unsupervised_learning_algorithms/breast_cancer_scaling_effect.ipynb)

*Description:* This notebook benchmarks four supervised-learning models on the Wisconsin breast-cancer dataset and demostrates how applying StandardScaler impacts each model's accuracy and confusion matrix.
