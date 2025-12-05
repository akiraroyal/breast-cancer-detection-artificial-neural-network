import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("breast_cancer_ann.keras", compile=False)
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("Breast Cancer Malignancy Prediction (ANN)")
st.write("Enter tumor measurements to predict whether the tumor is malignant or benign.")


feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
    "Compactness Error", "Concavity Error", "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

inputs = []

cols = st.columns(3)
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        value = st.number_input(feature, value=0.0)
        inputs.append(value)


if st.button("Predict Diagnosis"):
    input_array = np.array(inputs).reshape(1, -1)


    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0][0]

    if prediction > 0.5:
        st.success("✅ Prediction: Benign (Not Cancer)")
    else:
        st.error("⚠️ Prediction: Malignant (Cancer)")

    st.write(f"Model Confidence Score: {prediction:.4f}")
