import streamlit as st

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="Emergency Vehicle Detector", layout="centered")

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os

# Title
st.title("Emergency Vehicle Detection using CNN")

# Load the model
@st.cache_resource
def load_cnn_model():
    model_path = "emergency_vehicle_cnn.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error("Model file not found. Please make sure 'emergency_vehicle_cnn.h5' exists.")
        return None

model = load_cnn_model()

# Upload an image
uploaded_file = st.file_uploader("Upload an image of a vehicle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)

    # Predict
    prediction = model.predict(img_array)[0][0]
    predicted_class = "Emergency Vehicle" if prediction >= 0.5 else "Non-Emergency Vehicle"

    # Display result
    st.markdown(f"### Prediction: **{predicted_class}**")
    st.progress(float(prediction))
