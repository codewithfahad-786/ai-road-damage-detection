```python
import streamlit as st
import requests
from PIL import Image

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
# HEADER
# ============================================================

st.title("🛣️ AI Road Damage Detection")
st.write(
    "Upload a road image to detect road damage and assess its severity."
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("About Project")

    st.write(
        """
        This application uses a deep learning model to detect
        road damage from images.

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

    st.write("**Backend:** Railway")
    st.write("**Frontend:** Streamlit")

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

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
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {image.size}")
        st.write(f"**Format:** {image.format}")

    st.divider()

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔍 Detect Road Damage",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing road image..."):

            try:

                uploaded_file.seek(0)

                image_bytes = uploaded_file.read()

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
                # SUCCESS
                # =================================================

                if response.status_code == 200:

                    result = response.json()

                    st.success("✅ Road damage detection completed!")

                    # =================================================
                    # GET RESPONSE DATA
                    # =================================================

                    damage_type = result.get(
                        "damage_type",
                        "Unknown"
                    )

                    confidence = result.get(
                        "confidence",
                        0
                    )

                    severity_data = result.get(
                        "severity",
                        {}
                    )

                    severity = severity_data.get(
                        "severity",
                        "Unknown"
                    )

                    recommendation = severity_data.get(
                        "recommendation",
                        "No recommendation available"
                    )

                    probabilities = result.get(
                        "probabilities",
                        {}
                    )

                    # =================================================
                    # RESULT
                    # =================================================

                    st.subheader("📊 Detection Result")

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Damage Type",
                            damage_type.replace("_", " ")
                        )

                    with c2:
                        st.metric(
                            "Confidence",
                            f"{float(confidence):.2f}%"
                        )

                    with c3:
                        st.metric(
                            "Severity",
                            severity
                        )

                    # =================================================
                    # RECOMMENDATION
                    # =================================================

                    st.subheader("🛠️ Recommendation")

                    if severity.lower() == "high":
                        st.error(
                            f"⚠️ {recommendation}"
                        )

                    elif severity.lower() == "medium":
                        st.warning(
                            f"⚠️ {recommendation}"
                        )

                    else:
                        st.info(
                            f"ℹ️ {recommendation}"
                        )

                    # =================================================
                    # PROBABILITIES
                    # =================================================

                    st.subheader("📈 Class Probabilities")

                    for class_name, probability in probabilities.items():

                        st.write(
                            f"**{class_name.replace('_', ' ')}:** "
                            f"{float(probability):.2f}%"
                        )

                        st.progress(
                            min(float(probability) / 100, 1.0)
                        )

                    # =================================================
                    # COMPLETE RESPONSE
                    # =================================================

                    with st.expander("🔎 View Complete API Response"):
                        st.json(result)

                # =================================================
                # API ERROR
                # =================================================

                else:

                    st.error(
                        f"❌ Backend Error: HTTP {response.status_code}"
                    )

                    st.code(response.text)

            # =====================================================
            # CONNECTION ERROR
            # =====================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the Railway backend."
                )

                st.info(
                    "Please check that the Railway service is running."
                )

            # =====================================================
            # TIMEOUT ERROR
            # =====================================================

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ Request timed out. "
                    "The Railway backend may be waking up."
                )

            # =====================================================
            # OTHER ERROR
            # =====================================================

            except Exception as e:

                st.error(
                    f"❌ Unexpected error: {str(e)}"
                )

else:

    st.info(
        "👆 Upload a road image above to start detection."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Smart Road Damage Detection & Severity Assessment System"
)

st.caption(
    "EfficientNetB0 • TensorFlow • FastAPI • Railway • Streamlit"
)
```
