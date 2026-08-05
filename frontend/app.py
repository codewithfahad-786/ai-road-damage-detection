```python
import streamlit as st
import requests
from PIL import Image
import io

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://ai-road-damage-detection-production.up.railway.app"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛣️ AI Road Damage Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a road image to detect road damage and assess its severity.</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("About the Project")

    st.write(
        """
        This application uses a deep learning model to detect
        road damage from uploaded images.

        **Model:** EfficientNetB0

        **Damage Classes:**
        - Alligator Crack
        - Longitudinal Crack
        - Other Damage
        - Pothole
        - Transverse Crack
        """
    )

    st.divider()

    st.write("### Backend")
    st.code(API_URL)

    st.write("Backend: Railway")
    st.write("Frontend: Streamlit")

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# IMAGE PREVIEW
# ============================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(
                image,
                caption="Road Image",
                use_container_width=True
            )

        with col2:
            st.subheader("Image Information")

            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Size:** {image.size}")

        st.divider()

        # ====================================================
        # PREDICTION BUTTON
        # ====================================================

        if st.button(
            "🔍 Detect Road Damage",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Analyzing image..."):

                try:
                    # Reset file pointer
                    uploaded_file.seek(0)

                    # Read image bytes
                    image_bytes = uploaded_file.read()

                    # Send image to FastAPI backend
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={
                            "file": (
                                uploaded_file.name,
                                image_bytes,
                                uploaded_file.type
                            )
                        },
                        timeout=120
                    )

                    # =================================================
                    # RESPONSE HANDLING
                    # =================================================

                    if response.status_code == 200:

                        result = response.json()

                        st.success("✅ Analysis completed successfully!")

                        st.subheader("📊 Detection Result")

                        # -------------------------------------------------
                        # Display common possible response fields
                        # -------------------------------------------------

                        prediction = (
                            result.get("prediction")
                            or result.get("class")
                            or result.get("predicted_class")
                            or result.get("damage_type")
                            or result.get("label")
                        )

                        confidence = (
                            result.get("confidence")
                            or result.get("prediction_confidence")
                        )

                        severity = (
                            result.get("severity")
                            or result.get("severity_level")
                        )

                        # -------------------------------------------------
                        # Result columns
                        # -------------------------------------------------

                        c1, c2, c3 = st.columns(3)

                        with c1:
                            st.metric(
                                "Damage Type",
                                str(prediction)
                                if prediction is not None
                                else "N/A"
                            )

                        with c2:
                            if confidence is not None:

                                try:
                                    confidence_value = float(confidence)

                                    if confidence_value <= 1:
                                        confidence_value *= 100

                                    st.metric(
                                        "Confidence",
                                        f"{confidence_value:.2f}%"
                                    )

                                except:
                                    st.metric(
                                        "Confidence",
                                        str(confidence)
                                    )
                            else:
                                st.metric(
                                    "Confidence",
                                    "N/A"
                                )

                        with c3:
                            st.metric(
                                "Severity",
                                str(severity)
                                if severity is not None
                                else "N/A"
                            )

                        # -------------------------------------------------
                        # Full API Response
                        # -------------------------------------------------

                        st.markdown("### 🔎 Complete API Response")

                        st.json(result)

                    else:

                        st.error(
                            f"❌ Backend returned HTTP {response.status_code}"
                        )

                        st.code(response.text)

                except requests.exceptions.Timeout:

                    st.error(
                        "⏱️ Backend request timed out. "
                        "Railway may be waking up or processing the model."
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "❌ Could not connect to the Railway backend."
                    )

                    st.info(
                        "Check that the Railway backend is running and "
                        "the public URL is correct."
                    )

                except Exception as e:

                    st.error(
                        f"❌ An unexpected error occurred: {str(e)}"
                    )

else:

    st.info(
        "👆 Please upload a road image to start detection."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <p>AI-Based Smart Road Damage Detection & Severity Assessment System</p>
        <p>Powered by TensorFlow • EfficientNetB0 • FastAPI • Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
```
