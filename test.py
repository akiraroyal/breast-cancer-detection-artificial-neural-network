import tensorflow as tf
from pathlib import Path

model_path = Path(__file__).parent / "breast_cancer_ann.keras"
print("Using model:", model_path)
print("Exists:", model_path.exists(), "Size:", model_path.stat().st_size, "bytes")

model = tf.keras.models.load_model(model_path, compile=False)
print(" Loaded model OK from check_model.py")
