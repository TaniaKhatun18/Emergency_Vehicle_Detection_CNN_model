import streamlit as st
import numpy as np
import cv2
import gdown
import os
from PIL import Image
from tensorflow.keras.models import load_model

# Set page config at the top (MUST be first Streamlit command)
st.set_page_config(page_title="Emergency Vehicle Detector", layout="centered")

# Model file details
MODEL_PATH = "emergency_vehicle_cnn.h5"
GDRIVE_MODEL_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"  # <-- Replace with your actual Google Drive ID

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading the model..."):
        url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)

# Streamlit app UI
st.title("Emergency Vehicle Detector")
st.write("Upload an image of a vehicle to detect if it's an emergency vehicle.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    image_array = np.array(image)
    image_resized = cv2.resize(image_array, (64, 64))  # Adjust size if needed
    image_normalized = image_resized / 255.0
    input_image = np.expand_dims(image_normalized, axis=0)

    # Prediction
    prediction = model.predict(input_image)
    class_label = "Emergency Vehicle" if prediction[0][0] > 0.5 else "Non-Emergency Vehicle"

    # Result
    st.subheader("Prediction:")
    st.success(f"The image is classified as: **{class_label}**")
