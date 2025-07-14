import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

st.set_page_config(page_title="Emergency Vehicle Detector", layout="centered")
st.title("Emergency Vehicle Detection")

MODEL_PATH = "emergency_vehicle_cnn.h5"
GDRIVE_MODEL_ID = "1sgHAva3pdl5kpo4sJ_Oly9J9H-kRQSQP"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading the model..."):
        url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload an image of a vehicle", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    resized = image.resize((128, 128))
    image_array = np.array(resized) / 255.0  # normalize
    input_image = np.expand_dims(image_array, axis=0)  # Shape: (1, 128, 128, 3)

    prediction = model.predict(input_image)
    st.write("🔍 Raw prediction output:", prediction)

    # If prediction is a probability score (sigmoid)
    if prediction.shape[-1] == 1:
        predicted_class = int(prediction[0][0] > 0.5)
        class_names = ["Non-Emergency Vehicle", "Emergency Vehicle"]
    else:
        # If prediction is softmax (2 class)
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_names = ["Non-Emergency Vehicle", "Emergency Vehicle"]

    st.subheader("Prediction:")
    st.success(f"🚗 The vehicle is: **{class_names[predicted_class]}**")
