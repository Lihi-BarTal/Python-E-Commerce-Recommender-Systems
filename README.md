# 🎬 Classic Recommender Systems (Bias & SVD)

**University Project - E-Commerce Recommender Systems (HW 2 Part 1)**

## 💡 Overview
This project implements and tests two fundamental collaborative filtering algorithms for predicting movie ratings on a scale of 1 to 5. The primary goal was to report the Mean Squared Error (MSE) on the training set and generate predictions for the test set.

## ⚙️ Implemented Algorithms

1.  **`task1.py` - Bias Model (Least Squares):**
    * **Method:** Solves the linear system to find the global mean and individual user/item biases, $\hat{r}_{u,i} = \mu + b_u + b_i$.
    * **Result (Train MSE):** **0.8796** (as reported in `mse.txt`)

2.  **`task2.py` - SVD / Matrix Factorization:**
    * **Method:** Applies Singular Value Decomposition (SVD) to the user-item rating matrix with a rank $k=10$ to approximate the latent factors.
    * **Result (Train MSE):** **6.0266** (as reported in `mse.txt`)

## 📂 Project Files

| File Name | Description |
| :--- | :--- |
| **`task1.py`** | Python implementation of the **Bias Model** (Least Squares). Calculates MSE and generates predictions. |
| **`task2.py`** | Python implementation of the **SVD / Matrix Factorization** model ($k=10$). Calculates MSE and generates predictions. |
| **`mse.txt`** | Plain text file containing the calculated **MSE values** for the two models. |
| **`pred1.csv`** | Generated predictions for the test set using the **Bias Model** (Task 1). |
| **`pred2.csv`** | Generated predictions for the test set using the **SVD Model** (Task 2). |

## 🚀 How to Run

1.  **Dependencies:** Requires `numpy`, `pandas`, and `scipy`.
2.  **Input Data:** Requires the original `train.csv` and `test.csv` files (not included here).
3.  **Execution:** Run the scripts sequentially:
    ```bash
    python task1.py
    python task2.py
    ```
    (This regenerates `mse.txt`, `pred1.csv`, and `pred2.csv`).
