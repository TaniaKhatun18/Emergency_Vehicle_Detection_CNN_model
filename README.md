# 🚨 Emergency Vehicle Detection Using CNN

This project detects whether a vehicle image is an **Emergency** or **Non-Emergency Vehicle** using a trained Convolutional Neural Network (CNN) model. It is deployed via a **Streamlit web interface** and uses **Google Drive** to download the model if it's not locally available.

---

## 🔍 Project Objective

To help traffic control systems and smart city infrastructure identify emergency vehicles in real-time and take appropriate actions (like giving signal priority to ambulances or fire trucks).

---

## 📸 Demo Screenshots

### ✅ Emergency Vehicle Detected
![Emergency Vehicle](emergency_prediction.jpg)

### 🚗 Non-Emergency Vehicle Detected
![Non-Emergency Vehicle](images/non_emergency_prediction.jpg)

---

## 🧠 Model Architecture

The CNN architecture used for this binary classification task is:

```text
Input Layer: (128, 128, 3)
↓
Conv2D (32 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D (64 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Flatten
↓
Dense (128) + ReLU
↓
Dense (2 or 1) + Softmax/Sigmoid
Optimizer: Adam

Loss Function: categorical_crossentropy or binary_crossentropy

Accuracy Achieved: ~94%

🚀 Project Features
Upload any vehicle image (JPG/PNG)

Streamlit web interface for real-time prediction

Auto-download model from Google Drive

Displays class prediction with success message

Clean, responsive UI with icons

🧰 Tech Stack
Component	Technology
Programming Lang	Python
Model Framework	TensorFlow / Keras
UI Framework	Streamlit
Image Handling	Pillow (PIL), NumPy
Model Storage	Google Drive + gdown

📁 Folder Structure

Emergency_Vehicles/
│
├── app.py                         # Main Streamlit app
├── emergency_vehicle_cnn.h5      # Trained CNN model (auto-downloaded)
├── images/
│   ├── emergency_prediction.jpg   # Screenshot
│   └── non_emergency_prediction.jpg
├── requirements.txt              # Python dependencies
├── README.md                     # Project Documentation
📦 Setup Instructions
Install dependencies


pip install -r requirements.txt
Run the app

streamlit run app.py
Upload image and get prediction!

The model will auto-download from Google Drive if it's not present in the folder.

🌐 Deployment
This project can be deployed on:

Streamlit Cloud

Localhost

Heroku (with minor changes)

🔗 Model Access
Model stored on Google Drive:

https://drive.google.com/uc?id=1sgHAva3pdl5kpo4sJ_Oly9J9H-kRQSQP

You don’t need to manually download it — the app does it automatically using gdown.

🧪 Future Enhancements
Real-time video stream detection

Add bounding boxes using YOLO for object localization

Mobile/web deployment

Multi-class vehicle classification

🙏 Acknowledgements
Guidance: Ms. Arpita Roy

Course: Artificial Intelligence Programming Assistance (NSTIW Kolkata)

Tools: Keras, Streamlit, Google Drive API

📌 Author
👩‍💻 Tania Khatun
AI Developer | NSTIW Kolkata

This project is part of my final submission for the AIPA Certificate Course 2024–2025.

📎 License
This project is licensed under the MIT License.

