import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="RoadSight AI | Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #f4f8fc 0%, #edf4fa 55%, #f8fafc 100%); color: #102033; }
.block-container { max-width: 1240px; padding-top: 1.5rem; padding-bottom: 3rem; }
.hero { padding: 2.1rem 2.2rem; border-radius: 24px; background: linear-gradient(120deg,#0b1f36 0%,#123d62 58%,#087e8b 100%); color: white; box-shadow: 0 18px 45px rgba(15,43,72,.17); margin-bottom: 1.35rem; }
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2.55rem; line-height: 1.1; margin: .45rem 0 .7rem 0; color: white; }
.hero p { color: #d8e8f5; font-size: 1rem; max-width: 780px; margin-bottom: 0; }
.eyebrow { color: #72e4d1; text-transform: uppercase; letter-spacing: .16em; font-weight: 700; font-size: .76rem; }
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; color: #102033; margin: .4rem 0 .2rem 0; }
.muted { color: #64748b; font-size: .93rem; }
.metric-card { background: white; border: 1px solid #e1eaf2; border-radius: 16px; padding: 1rem 1.1rem; min-height: 105px; box-shadow: 0 5px 18px rgba(20,48,76,.04); }
.metric-label { color: #64748b; font-size: .82rem; font-weight: 600; margin-bottom: .45rem; }
.metric-value { color: #102033; font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; overflow-wrap: anywhere; }
.info-card { background: white; border: 1px solid #e1eaf2; border-radius: 16px; padding: 1.15rem 1.2rem; }
.stButton > button { border-radius: 11px; min-height: 2.8rem; font-weight: 700; }
[data-testid="stSidebar"] { background: #eef4f9; border-right: 1px solid #dce7f0; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Paths: support common repo layouts
# Put road_model.keras and JSON files beside this app.py,
# or in backend/ if you prefer to preserve your folder setup.
# ---------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    APP_DIR / "road_model.keras",
    APP_DIR / "backend" / "road_model.keras",
    APP_DIR / "models" / "road_model.keras",
    APP_DIR / "road_damage_model.keras",
    APP_DIR / "backend" / "road_damage_model.keras",
]
LABEL_CANDIDATES = [
    APP_DIR / "class_labels.json",
    APP_DIR / "backend" / "class_labels.json",
    APP_DIR / "models" / "class_labels.json",
]
SEVERITY_CANDIDATES = [
    APP_DIR / "severity_mapping.json",
    APP_DIR / "backend" / "severity_mapping.json",
    APP_DIR / "models" / "severity_mapping.json",
]

IMG_SIZE = 224


def first_existing(paths):
    return next((path for path in paths if path.is_file()), None)


def read_json(path, default=None):
    if path is None:
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


@st.cache_resource(show_spinner="Loading the trained road-damage model...")
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)


def normalize_labels(raw):
    if isinstance(raw, list):
        return {i: str(label) for i, label in enumerate(raw)}
    if isinstance(raw, dict):
        try:
            return {int(k): str(v) for k, v in raw.items()}
        except (ValueError, TypeError):
            return {i: str(v) for i, v in enumerate(raw.values())}
    return {
        0: "Alligator_Crack",
        1: "Longitudinal_Crack",
        2: "Other_Damage",
        3: "Pothole",
        4: "Transverse_Crack",
    }


def preprocess_image(image):
    # Match the preprocessing used in the provided FastAPI main.py.
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def pretty_label(label):
    return str(label).replace("_", " ")


def severity_details(label, mapping):
    details = mapping.get(label, {}) if isinstance(mapping, dict) else {}
    if isinstance(details, dict):
        severity = details.get("severity", "Unknown")
        recommendation = details.get("recommendation", "Please arrange a road inspection.")
    else:
        severity = str(details) if details else "Unknown"
        recommendation = "Please arrange a road inspection."
    return str(severity), str(recommendation)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛣️ RoadSight AI")
    st.caption("Road Damage Intelligence")
    st.divider()
    page = st.radio("NAVIGATION", ["Detection workspace", "About the project"])
    st.divider()
    st.markdown("**AI model classes**")
    st.caption("Alligator crack · Longitudinal crack · Other damage · Pothole · Transverse crack")
    st.divider()
    st.caption("Streamlit · TensorFlow · EfficientNet")
    st.caption("Use predictions as decision support; confirm important findings on site.")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Computer vision • Road safety • Smart infrastructure</div>
  <h1>AI Road Damage Detection</h1>
  <p>Upload a road photo or capture one with your camera. Run your trained model locally in this Streamlit app without relying on the expired Railway API.</p>
</div>
""", unsafe_allow_html=True)

model_path = first_existing(MODEL_CANDIDATES)
labels_path = first_existing(LABEL_CANDIDATES)
severity_path = first_existing(SEVERITY_CANDIDATES)
class_labels = normalize_labels(read_json(labels_path))
severity_mapping = read_json(severity_path, {})

if page == "About the project":
    st.markdown('<div class="section-title">About this project</div>', unsafe_allow_html=True)
    st.markdown('<p class="muted">A computer-vision interface for classifying common road surface damage.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="info-card"><h3>📤 Image upload</h3><p>Upload JPG, JPEG, or PNG images from your device.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-card"><h3>📸 Camera capture</h3><p>Capture a new photo using a supported webcam or mobile camera.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="info-card"><h3>🧠 Local inference</h3><p>Load the existing Keras model directly in Streamlit, with no Railway API call.</p></div>', unsafe_allow_html=True)
    st.info("The trained road_model.keras file is required for real AI predictions. The class labels and severity mapping JSON files must match the model.")
else:
    st.markdown('<div class="section-title">Detection workspace</div>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Choose an input method, preview the road image, then run detection.</p>', unsafe_allow_html=True)

    upload_tab, camera_tab = st.tabs(["📤 Upload image", "📸 Capture image"])
    selected_image = None
    image_source = ""

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose a road image",
            type=["jpg", "jpeg", "png"],
            help="Use a clear, well-lit photo of the road surface.",
            key="road_image_upload",
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file)
                image_source = uploaded_file.name
            except Exception:
                st.error("This image could not be opened. Please choose another JPG or PNG file.")

    with camera_tab:
        camera_file = st.camera_input("Take a photo of the road", key="road_camera_capture")
        if camera_file is not None:
            try:
                selected_image = Image.open(camera_file)
                image_source = "Camera capture"
            except Exception:
                st.error("Could not read the camera image. Please try again.")

    if selected_image is not None:
        selected_image = ImageOps.exif_transpose(selected_image).convert("RGB")
        st.markdown("#### Image preview")
        preview_col, details_col = st.columns([1.15, 0.85], gap="large")
        with preview_col:
            st.image(selected_image, caption=image_source, use_container_width=True)
        with details_col:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("**Image ready for analysis**")
            st.write(f"**Source:** {image_source}")
            st.write(f"**Resolution:** {selected_image.width} × {selected_image.height} pixels")
            st.write("**Color mode:** RGB")
            st.markdown('</div>', unsafe_allow_html=True)
            analyze = st.button("🔍 Detect road damage", type="primary", use_container_width=True)

        if analyze:
            if model_path is None:
                st.error(
                    "Trained model file not found. Upload and camera preview work, but AI prediction needs "
                    "`road_model.keras`. Put it beside `app.py` or in `backend/`, then redeploy."
                )
            elif labels_path is None or severity_path is None:
                st.error(
                    "Could not find `class_labels.json` or `severity_mapping.json`. "
                    "Put both beside `app.py` or in the `backend/` folder."
                )
            else:
                try:
                    with st.spinner("Analyzing road surface with the trained model..."):
                        model = load_model(str(model_path))
                        prediction = model.predict(preprocess_image(selected_image), verbose=0)
                        scores = np.asarray(prediction).reshape(-1)

                        if scores.size == 0:
                            raise ValueError("The model returned an empty prediction.")

                        # This matches the supplied API, which treats model output as class probabilities.
                        predicted_index = int(np.argmax(scores))
                        confidence = float(scores[predicted_index]) * 100
                        damage_type = class_labels.get(predicted_index, f"Class {predicted_index}")
                        severity, recommendation = severity_details(damage_type, severity_mapping)

                    st.divider()
                    st.markdown('<div class="section-title">Detection results</div>', unsafe_allow_html=True)
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">Predicted damage</div>'
                            f'<div class="metric-value">{pretty_label(damage_type)}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with r2:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">Model confidence</div>'
                            f'<div class="metric-value">{confidence:.2f}%</div></div>',
                            unsafe_allow_html=True,
                        )
                    with r3:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">Severity</div>'
                            f'<div class="metric-value">{severity}</div></div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("#### Recommended action")
                    if severity.lower() == "high":
                        st.error(recommendation)
                    elif severity.lower() == "medium":
                        st.warning(recommendation)
                    else:
                        st.info(recommendation)

                    st.markdown("#### Class probabilities")
                    probability_rows = {}
                    for index, raw_score in enumerate(scores):
                        label = class_labels.get(index, f"Class {index}")
                        probability_rows[pretty_label(label)] = float(raw_score) * 100
                    st.bar_chart(probability_rows)
                    for label, value in probability_rows.items():
                        st.caption(f"{label}: {value:.2f}%")
                        st.progress(max(0, min(int(round(value)), 100)))

                    st.caption("AI results may be wrong on unclear or unusual images. Verify important findings with a qualified road inspector.")
                except Exception as exc:
                    st.error("Prediction failed. Check that the model file and TensorFlow version match the original project.")
                    st.exception(exc)
    else:
        st.markdown("""
        <div class="info-card">
          <h3>Ready when you are</h3>
          <p>Upload an existing road image or use the camera tab to capture a photo. The app will show a preview before analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    if model_path:
        st.success(f"Model file found: {model_path.relative_to(APP_DIR)}")
    else:
        st.warning("Model file not found yet. Image upload and camera preview are available, but real predictions require road_model.keras.")
