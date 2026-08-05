import streamlit as st
import requests

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="AI Road Damage Detection", page_icon="🚧", layout="centered"
)

st.title("🚧 AI Road Damage Detection & Severity Assessment")
st.write("Upload an image of the road to detect damages via FastAPI Backend.")

# 🔴 APNE LIVE DEPLOYED BACKEND KA URL YAHAN DALEIN 🔴
# Localhost internet par kaam nahi karega, live https link hona zaroori hai.
BACKEND_URL = "https://your-deployed-fastapi-url.com"

# =====================================================
# FRONTEND INTERFACE & API CALL
# =====================================================
uploaded_file = st.file_uploader(
    "Choose a road image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image directly from memory
    st.image(uploaded_file, caption="Uploaded Road Image", use_container_width=True)

    if st.button("Run Damage Analysis", type="primary"):
        with st.spinner("Sending image to AI Backend server..."):
            try:
                # Prepare the payload for FastAPI UploadFile
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                # Make the POST request to your FastAPI server
                response = requests.post(BACKEND_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    # Extract data from backend JSON response
                    damage_type = result.get("damage_type", "Unknown")
                    severity = result.get("severity", "Unknown")
                    confidence = result.get("confidence", 0.0)
                    probabilities = result.get("probabilities", {})

                    st.success("Analysis Complete successfully!")

                    # Metrics view layout split
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label="Damage Type", value=damage_type)
                    col2.metric(label="Severity Level", value=severity)
                    col3.metric(label="Confidence", value=f"{confidence}%")

                    # Display Probabilities charts if returned by backend
                    if probabilities:
                        st.write("### 📊 Distribution Probabilities")
                        for name, percentage in probabilities.items():
                            st.write(f"**{name}**")
                            # Convert back to 0-1 scale for streamlit progress bar
                            st.progress(percentage / 100)
                            st.caption(f"{percentage}%")
                else:
                    st.error(
                        f"Backend Error (Status {response.status_code}): {response.text}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the backend server. Please verify your BACKEND_URL."
                )
            except Exception as e:
                st.error(
                    f"Something went wrong during the API pipeline: {str(e)}"
                )
