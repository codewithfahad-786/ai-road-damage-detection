import streamlit as st
import requests
from PIL import Image

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="centered"
)

# ==============================
# Railway Backend URL
# ==============================
BACKEND_URL = "https://ai-road-damage-detection-production.up.railway.app"

# ==============================
# Title & Description
# ==============================
st.title("🛣️ AI Road Damage Detection")
st.write("Upload a road image to detect road damage and its severity.")
st.divider()

# ==============================
# Image Upload
# ==============================
uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# Prediction Logic
# ==============================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Road Damage", type="primary"):

        with st.spinner("Analyzing image... Please wait"):

            try:
                # Reset file position
                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                # Send image to Railway FastAPI backend
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    files=files,
                    timeout=180
                )

                # Check response
                if response.status_code == 200:
                    result = response.json()

                    damage_type = result.get("damage_type", "Unknown")
                    confidence = result.get("confidence", 0)
                    severity_data = result.get("severity", {})
                    severity = severity_data.get("severity", "Unknown")
                    recommendation = severity_data.get("recommendation", "No recommendation available")
                    probabilities = result.get("probabilities", {})

                    # ==============================
                    # Results Display
                    # ==============================
                    st.success("✅ Prediction completed!")

                    st.subheader("📊 Detection Result")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Damage Type", damage_type)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}%")

                    st.subheader("⚠️ Severity Analysis")
                    st.info(f"**Severity Level:** {severity}")
                    st.warning(f"**Recommendation:** {recommendation}")

                    # ==============================
                    # Class Probabilities
                    # ==============================
                    if probabilities:
                        st.subheader("📈 Class Probabilities")
                        for class_name, probability in probabilities.items():
                            st.write(f"**{class_name}:** {probability:.2f}%")
                            st.progress(min(int(round(float(probability))), 100))

                else:
                    st.error(f"❌ Backend Error: {response.status_code}")
                    st.code(response.text)

            except requests.exceptions.Timeout:
                st.error("⏱️ Backend request timed out. Please try again.")

            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to Railway backend. Please check if the service is running.")

            except Exception as e:
                st.error("❌ Something went wrong.")
                st.exception(e)
