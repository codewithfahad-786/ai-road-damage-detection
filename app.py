import streamlit as st
import numpy as np
import json

from PIL import Image
from tensorflow.keras.models import load_model


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="centered"
)

st.title("🛣️ AI-Based Smart Road Damage Detection")
st.write("Upload a road damage image for prediction.")


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_ai_model():

    model = load_model("road_damage_model.keras")

    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)

    with open("severity_mapping.json", "r") as f:
        severity_mapping = json.load(f)

    return model, class_labels, severity_mapping


model, class_labels, severity_mapping = load_ai_model()


# -----------------------------
# Image Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    confidence = float(
        prediction[0][predicted_index]
    ) * 100

    damage = class_labels[predicted_index]

    severity = severity_mapping[damage]["severity"]

    recommendation = severity_mapping[damage]["recommendation"]

    st.success("Prediction Completed!")

    st.subheader("Prediction Result")

    st.write(f"**Damage Type:** {damage}")

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.write(f"**Severity:** {severity}")

    st.write(f"**Recommendation:** {recommendation}")