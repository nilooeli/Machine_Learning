## Table of Contents

- [Supervised Learning Projects](#supervised-learning-projects)
  - [1. K-Nearest Neighbors Classification - Handwritten Digits](#1-k-nearest-neighbors-classification---handwritten-digits)
  - [2. K-Nearest Neighbors Regression - California Housing](#2-k-nearest-neighbors-regression---california-housing)
  - [3. Regression_models_comparison](#3-linear-vs-ridge-vs-lasso-elasticnet-regression)
  - [4. Logistic Regression Classification](#4-logistic-regression-classification)

## Supervised Learning Projects

### 1. K-Nearest Neighbors Classification - Handwritten Digits

*Notebook:* [knn_digits.ipynb](supervised_algorithms/knn_digits.ipynb)

*Description:* This project utilizes the K-Nearest Neighbors (KNN) algorithm to classify handwritten digits using the MNIST dataset. The goal is to accurately predict digit labels based on pixel data.

### 2. K-Nearest Neighbors Regression - California Housing

*Notebook:* [knn_regression.ipynb](supervised_algorithms/knn_regression.ipynb)

*Description:* This project applies the KNN regression technique to predict California housing prices based on various features such as location, size, and demographics.

### 3. Regression Model Comparison

*Notebook:* [regression_models_comparison.ipynb](supervised_algorithms/regression_models_comparison.ipynb)

*Description:* This project compares linear regression, ridge, loss, and elasticnet regression models to understand their performance differences. It provides insights into regularization techniques and their impact on model accuracy.

### 4. Logistic Regression Classification

*Notebook:* [logistic_regression_classification.ipynb](supervised_algorithms/logistic_regression_classification.ipynb)

*Description:* This project employs logistic regression to perform binary classification tasks. It demonstrates the application of logistic regression on a chosen dataset to predict categorical outcomes.


## 1. K-Nearest Neighbors Classification - Handwritten Digits
   - Location: supervised_algorithm/knn_digits.ipynb
   - Goal: Use KNN to classify images of handwritten digits from the scikit digits dataset.
   - Key Features:
      - Loads digit images using load_digits()
      - Splits data into training and test sets
      - Trains a KNN classifier using KNeighborsClassifier
      - Evaluate with:
          - Accuracy
          - Confusion matrix
   - Performs PCA (2D) to reduce dimensionality for visualization
  
## 2. K-Nearest Neighbors Regression - California Housing
   - Location: supervised_algorithm/knn_regression.ipynb
   - Goal: Predict median house values using KNN regression on the California Housing dataset.
   - Key Features:
      - Loads data using fetch_california_housing()
      - Split data into training and testing sets
      - Trains a KNN Regressor using KNeighborsRegressor
      - Evaluate with:
          - Mean Squared Error (MSE)
          - R2 Score
## 3. Linear vs Ridge vs Lasso vs ElasticNet Regression
   - Location: supervised_algorithms/linear_ridgge_regression.ipynb
   - Goal: Compare multiple linear model on the California Housing dataset to evaluate their performance in predicting housing prices.
   - Key Features:
      - Loads and preprocesses the California Housing
      - Trains and evaluates:
         - Linear Regression
         - Ridge Regression
         - Lasso Regression
         - ElasticNet Regression
   - Model Compared:
      - Linear Regression (no Regularization)
      - Ridge Regression (L2 Regularization)
      - Lasso Regression (L1 Regularization)
      - ElasticNet Regression (L1 + L2 Combined)
   - Uses Mean Squared Error (MSE) and R2 Score for evaluation
   - Includes side-by-side resutls for easy comparison

## 4. Linear Classification with Logistic Regression
   - Location: supervised_algorithm/logistic_regression_classification.ipynb
   - Goal: Classify tumors as either Malignant (0) or Benign (1) based on various medical attributes (like mean radius, texture, etc.)
   - Key Features:
      - Uses the Breast Cancer Wisconsin datset to calssify tumors as Malignant or Benign
      - Applies Logistic Regression to classify model
      - Evaluation:
         - Accuracy score
         - Classification report (precision, recall, f1-score)
         - Confusion matrix visualization using seaborn
   
      
      



















     
# Author:
# Niloo Eli
