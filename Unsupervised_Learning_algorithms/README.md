# Unsupervised Learning Algorithms Projects

This folder contains multiple unsupervised learning projects implemented in Python, including a full PCA (Principal Component Analysis) implementation from scratch and other examples such as dimensionality reduction and exception handling.

## Table of Contents
1. [PCA on Handwritten Digits](#pca-on-handwritten-digits)
2. [Exception Handling Practice](#exception-handling-practice)
3. [PCA-2D-To-1D](#pca-2d-to-1d)
4. [PCA From Scratch (3D to 2D Projection)](#1-pca-from-scratch-3d-to-2d-projection)
5. [Cancer Subtype Discovery using NMF](#cancer-subtype-discovery-using-nmf)
   

## 1. PCA on Handwritten Digits
- Input: High-dimensional dataset from `sklearn.datasets.load_digits`
- Output: Lower-dimensional visualization using the top 2 principal components
- Tools used: scikit-learn, matplotlib

This project demonstrates how PCA can help in understanding and visualizing complex datasets like handwritten digits. The digit images are compressed while preserving their structural integrity, making the results interpretable and useful for downstream tasks.

---

## 2. Exception Handling Practice
- Input: Custom Python code snippets that raise or catch exceptions
- Output: Printed results and explanations of each control flow scenario
- Tools used: Basic Python

This notebook introduces how to use Python’s exception handling blocks effectively (`try`, `except`, `else`, `finally`) through examples. It's ideal for understanding robust programming techniques.

---

## 3. PCA-2D-To-1D
- Input: A 2D synthetic dataset
- Output: A 1D projection using the first principal component
- Tools used: NumPy, Matplotlib

Demonstrates PCA on a simple 2D dataset to reduce dimensionality to 1D. The notebook includes projection lines, eigenvector visualization, and scatter plots before and after dimensionality reduction.

---

## 4. PCA From Scratch (3D to 2D Projection)
- Input: A 3D dataset with 10 data points
- Output: A 2D projection of the data using the top 2 principal components
- Tools used: NumPy, Matplotlib

---

## 5. Cancer Subtype Discovery using NMF

- **Inputs**: Breast cancer dataset from 'sklearn.dataset.load_breast_cancer'
- **Goal**: Use Non-negative Matrix Factorization (NMF) to reduce the 30-dimensional gene expression data into a smaller number of latent components
- **Steps**:
   - Normalize the data using 'MinMaxScaler'
   - Apply 'NMF' to extract 5 latent features'
   - Cluster samples using 'KMeans'
   - Visualize the results using a 2D scatter plot based on the first two components
- **Tools used**: 'scikit-learn', 'matplotlib', 'Numpy'

The project shows how NMF can uncover hidden structure in gene expression data, potentionally revealing biological subtypes of breast cancer. Clustering and visualization provide additional insight into how samples relate to each other in the reduced feature spcae.

## Key Concepts

### What is PCA?
Principal Component Analysis is a linear dimensionality reduction technique that:
- Identifies directions (principal components) of maximum variance in the data
- Projects data onto a new coordinate system aligned with these directions
- Enables compression of data with minimal information loss

---

## Step-by-Step Process

### 1. Create the Dataset
```python
X = np.array([[2.5, 2.4, 1.2],
              [0.5, 0.7, 0.3],
              ...])  # 10x3 matrix
```
Each row is an observation vector in \( \mathbb{R}^3 \).

### 2. Mean-Center the Data
```python
X_meaned = X - np.mean(X, axis=0)
```
This centers the dataset at the origin to eliminate bias due to translation.

### 3. Compute the Covariance Matrix
```python
cov_mat = np.cov(X_meaned, rowvar=False)
```
This 3x3 matrix captures how features vary with each other.

### 4. Eigenvalue Decomposition
```python
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
```
We use `eigh()` because the covariance matrix is symmetric.

### 5. Sort Eigenvectors by Eigenvalues
```python
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sorted_indices]
eigenvalues = eigenvalues[sorted_indices]
```
This step orders the principal components by the amount of variance they explain.

### 6. Project the Data
```python
W = eigenvectors[:, :2]          # Top 2 PCs
X_reduced = X_meaned @ W         # 3D -> 2D
```
The result is the 2D representation of the original data.

### 7. Visualize the Result
```python
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], ...)
```
This plots the data in the new PC1-PC2 coordinate system.

---

## Extras

### Visualizing PC1 and PC2 in 3D
```python
ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], label=f'PC{i+1}')
```
This adds vector arrows in 3D to represent the directions of maximum variance.

---

## Learning Outcomes
- Reinforced understanding of covariance, eigenvalues, and eigenvectors
- Practical experience with projecting high-dimensional data
- Demonstrated how PCA compresses data with minimal information loss

---

## Files Included
- `pca_digits.ipynb` – Applies PCA to the sklearn digits dataset to reduce dimensionality and visualize the digits in 2D or 3D space. A useful example for visualizing high-dimensional data and using PCA for preprocessing.
- `exception_practice.ipynb` – Demonstrates Python’s exception handling with `try`, `except`, `else`, and `finally` blocks. Useful for practicing how to write robust code that manages errors gracefully.
- `PCA-2D-To-1D.ipynb` – PCA example that reduces 2D data to 1D using NumPy. Demonstrates how data points are projected onto the direction of maximum variance (PC1), with visualizations showing the projection lines and resulting 1D embedding.
- `PCA-3D-To-2D.ipynb` – PCA example that reduces 3D data to 2D using NumPy. Demonstrates how data points are projected onto the two directions of maximum variance (PC1 and PC2), with visualizations in both 3D and 2D showing the projections and principal component vectors.
