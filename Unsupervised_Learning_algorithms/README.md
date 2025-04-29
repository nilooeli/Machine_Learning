## Table of Contents

-[Unsupervised Learning Projects](#Unsupervised-Learning-algorithms)
   - [1. PCA Digits - Handwritten Digits](#1-pca-digits)
     

## Unsupervised Learning Algorithms

This folder holds hands-on notebooks demonstrating common
unsupervised-learning techniques in Python:

- **PCA (Principle Component Analysis)** (`pca_digits.ipynb`)  
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

## 1. PCA Digits - Handwriten Digits

*Notebook:* [knn_digits.ipynb](Unsupervised_Learning_algorithms/knn_digits.ipynb)

*Description:* This notebook applies two of PCA: one for 95% variance (for clean image reconstruction) and one for 10 components (to show model performance). It compress a high-dimesional dataset while preserving as much information as possible.
It visualzess 2-D comples data and improves classification by reducing noise and reducdancy before applying  machine learning.




