# 🎬 Classic Recommender Systems (Bias & SVD Models)

**Python Project - Implementation for E-Commerce Recommender Systems**

## 💡 Overview
This project implements two core collaborative filtering algorithms in Python to predict movie ratings on a scale of 1 to 5. The primary goal was to calculate the Mean Squared Error (MSE) on the training set and generate predictions for the test set.

## ⚙️ Algorithms and Performance

| File | Method | Description | Train MSE (from mse.txt) |
| :--- | :--- | :--- | :--- |
| **`task1.py`** | **Bias Model** | Implemented using Least Squares to find the global, user, and item biases ($\hat{r}_{u,i} = \mu + b_u + b_i$). | **0.8796** |
| **`task2.py`** | **SVD** | Implemented using Singular Value Decomposition (Matrix Factorization) with $k=10$ latent factors. | **6.0266** |

## 📂 Project Files

| File Name | Content |
| :--- | :--- |
| **`task1.py`** | Python code for the **Bias Model**. |
| **`task2.py`** | Python code for the **SVD Model**. |
| **`mse.txt`** | Generated text file containing the final MSE results for both models. |
| **`pred1.csv`** | Final rating predictions for the test set (from Task 1). |
| **`pred2.csv`** | Final rating predictions for the test set (from Task 2). |
