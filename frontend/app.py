import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Railway FastAPI URL
API_URL = "https://agile-energy-production-e5d4.up.railway.app/predict"


# Page configuration
st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🚧",
    layout="centered"
)


# Title
st.title("🚧 AI Road Damage Detection")
st.write(
    "Upload a road image to detect road damage, "
    "confidence, severity, and recommended action."
)


# Image uploader
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    # Prediction button
    if st.button("🔍 Detect Road Damage", use_container_width=True):

        with st.spinner("Analyzing road image..."):

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

                # Send image to FastAPI
                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=120
                )

                # Check response
                if response.status_code == 200:

                    result = response.json()

                    damage_type = result["damage_type"]
                    confidence = result["confidence"]

                    severity_data = result["severity"]

                    severity = severity_data["severity"]
                    recommendation = severity_data["recommendation"]

                    # Results
                    st.success("Prediction completed successfully!")

                    st.subheader("📊 Detection Result")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Damage Type",
                            damage_type
                        )

                    with col2:
                        st.metric(
                            "Confidence",
                            f"{confidence:.2f}%"
                        )

                    st.subheader("⚠️ Severity")

                    st.info(severity)

                    st.subheader("💡 Recommendation")

                    st.write(recommendation)

                    # Probabilities
                    st.subheader("📈 Class Probabilities")

                    probabilities = result["probabilities"]

                    for class_name, probability in probabilities.items():

                        st.write(
                            f"**{class_name.replace('_', ' ')}:** "
                            f"{probability:.2f}%"
                        )

                        st.progress(
                            min(int(probability), 100)
                        )

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                    st.write(response.text)

            except requests.exceptions.Timeout:

                st.error(
                    "The API took too long to respond. "
                    "Please try again."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the Railway API."
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )

