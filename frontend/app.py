import json
import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🚧",
    layout="centered"
)

st.title("🚧 AI Road Damage Detection & Severity Assessment")
st.write("Upload an image of the road to detect damages and evaluate severity.")

# System paths map (Points to root or backend depending on structure)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust paths to match your folder structure (Checking if backend folder wrapper exists)
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "backend"))

if os.path.exists(BACKEND_DIR):
    MODEL_PATH = os.path.join(BACKEND_DIR, "road_model.keras")
    CLASS_LABELS_PATH = os.path.join(BACKEND_DIR, "class_labels.json")
    SEVERITY_MAPPING_PATH = os.path.join(BACKEND_DIR, "severity_mapping.json")
else:
    MODEL_PATH = os.path.join(BASE_DIR, "road_model.keras")
    CLASS_LABELS_PATH = os.path.join(BASE_DIR, "class_labels.json")
    SEVERITY_MAPPING_PATH = os.path.join(BASE_DIR, "severity_mapping.json")

@st.cache_resource
def load_ai_model():
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model file missing at: {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_configs():
    if not os.path.isfile(CLASS_LABELS_PATH) or not os.path.isfile(SEVERITY_MAPPING_PATH):
        st.error("Configuration JSON files are missing!")
        return {}, {}
        
    with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, list):
        labels = {i: name for i, name in enumerate(labels)}
    elif isinstance(labels, dict):
        labels = {int(key): value for key, value in labels.items()}

    with open(SEVERITY_MAPPING_PATH, "r", encoding="utf-8") as f:
        severity = json.load(f)
        
    return labels, severity

model = load_ai_model()
class_labels, severity_mapping = load_configs()

IMG_SIZE = 224
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader("Choose a road image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Road Image", use_container_width=True)
    
    if st.button("Run Detection"):
        with st.spinner("Analyzing road conditions..."):
            try:
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image, verbose=0)
                
                predicted_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                
                damage_type = class_labels.get(predicted_index, "Unknown")
                severity = severity_mapping.get(damage_type, "Low/Unknown")
                
                st.success("Analysis Complete!")
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Damage Type", value=damage_type)
                col2.metric(label="Severity Level", value=severity)
                col3.metric(label="Confidence", value=f"{round(confidence * 100, 2)}%")
                        
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
