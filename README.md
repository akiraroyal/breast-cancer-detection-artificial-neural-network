# Breast Cancer Detection – Artificial Neural Network

This project builds an artificial neural network (ANN) to predict whether a breast tumor is **malignant** or **benign** using diagnostic features from the classic breast cancer dataset.  
It includes:

- A Jupyter notebook for **data exploration, feature scaling, and model training**
- A saved **TensorFlow/Keras model** and **StandardScaler**
- A **Streamlit web app** for interactive predictions

---

## 1. Project Overview

Goal: use a simple fully connected neural network to perform **binary classification** on breast cancer diagnostic data.

Key steps:

1. Load and inspect the dataset
2. Clean and scale the features
3. Train and evaluate an ANN classifier
4. Save the trained model and scaler
5. Expose the model with a Streamlit app so users can enter feature values and get a prediction

---

## 2. Dataset

- Source: `sklearn.datasets.load_breast_cancer`
- Samples: 569
- Features: 30 numeric features per tumor (mean radius, mean texture, mean smoothness, etc.)
- Target:
  - `0` → malignant  
  - `1` → benign

In the notebook, the data and target are combined into a single pandas DataFrame for exploration and preprocessing.

---

## 3. Model Architecture

The ANN is built with **TensorFlow/Keras** as a simple feed-forward network:

- Input: 30 scaled features
- Hidden Layer 1: Dense(6), activation = ReLU
- Hidden Layer 2: Dense(6), activation = ReLU
- Output Layer: Dense(1), activation = Sigmoid (for binary classification)

**Loss & optimizer**

- Loss: `binary_crossentropy`
- Optimizer: e.g. `adam`
- Metrics: `accuracy`

The trained model is saved as:

- `breast_cancer_ann.keras` – trained Keras model
- `scaler.pkl` – fitted `StandardScaler` for preprocessing

---

## 4. Model Performance

Evaluated on a **held-out test set**:

- **Accuracy:** ~0.95 (94–95%)
- **Confusion Matrix:** shows low misclassification rates
- **Malignant class metrics (example):**
  - Precision ≈ 0.97
  - Recall ≈ 0.88
  - F1-score ≈ 0.96

Interpretation:

- **Precision (malignant):** when the model predicts *malignant*, it is correct most of the time.
- **Recall (malignant):** out of all *actual* malignant cases, the model correctly identifies the majority.
- Overall, the ANN performs well as a prototype, but it is not intended for real-world clinical use.

---

## 5. Project Structure

```text
.
├── app.py                                   # Streamlit app for interactive predictions
├── breast_cancer__artificial_neural_network.ipynb  # Notebook: EDA, training, evaluation
├── breast_cancer_ann.keras                  # Saved Keras model
├── scaler.pkl                               # Saved StandardScaler
├── requirements.txt                         # Python dependencies
└── .gitignore
