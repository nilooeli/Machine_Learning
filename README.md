# Supervised Learning Projects 
This repository contains hands-on rojects focused on supervised learning techniques using Python and scikit-learn.
Each project demonstrates a different algorithem or application, complete with model training, evaluation, and visualization

## 1. K-Nearest Neighbors Classification - Handwritten Digits
   Location: supervised_algorithm/knn_digits.ipynb
   Goal: Use KNN to classify images of handwritten digits from the scikit digits dataset.
   Key Features:
   - Loads digit images using load_digits()
   - Splits data into training and test sets
   - Trains a KNN classifier using KNeighborsClassifier
   - Evaluate with:
       - Accuracy
       - Confusion matrix
   - Performs PCA (2D) to reduce dimensionality for visualization
  
## 2. K-Nearest Neighbors Regression - California Housing
   Location: supervised_algorithm/knn_regression.ipynb
   Goal: Predict median house values using KNN regression on the California Housing dataset.
   Key Features:
   - Loads data using fetch_california_housing()
   - Split data into training and testing sets
   - Trains a KNN Regressor using KNeighborsRegressor
   - Evaluate with:
       - Mean Squared Error (MSE)
       - R2 Score
## 3. Linear vs Ridge vs Lasso vs ElasticNet Regression
   Location: supervised_algorithms/linear_ridgge_regression.ipynb
   Goal: Compare multiple linear model on the California Housing dataset to evaluate their performance in predicting housing prices.
   ### Model Compared:
   - Linear Regression (no Regularization)
   - Ridge Regression (L2 Regularization)
   - Lasso Regression (L1 Regularization)
   - ElasticNet Regression (L1 + L2 Combined)
   Key Features:
   - Loads and preprocesses the California Housing
   - Trains and evaluates:
      - Linear Regression
      - Ridge Regression
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
