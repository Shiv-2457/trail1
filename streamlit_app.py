import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load MobileNetV2 Model (Pre-trained)
@st.cache_resource()
def load_pretrained_model():
    model = MobileNetV2(weights="imagenet", include_top=True)
    return model

# Load model
model = load_pretrained_model()
st.write("✅ Pre-trained MobileNetV2 Model Loaded!")

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("♻️ Waste Classification App")
st.write("Upload an image to classify the type of waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess & Predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=1)

    st.write(f"### 🏷️ Predicted Class: **{predicted_class[0][0][1]}**")
    st.write(f"📝 Confidence Score: **{predicted_class[0][0][2]:.2f}**")
