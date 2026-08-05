import streamlit as st
import requests

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="AI Road Damage Detection", 
    page_icon="🚧", 
    layout="centered"
)

st.title("🚧 AI Road Damage Detection & Severity Assessment")
st.write("Upload an image of the road to detect damages via AI Backend.")

# =====================================================
# LIVE RAILWAY BACKEND CONFIGURATION
# =====================================================
BACKEND_URL = "https://railway.app"

# =====================================================
# FRONTEND INTERFACE & API CALL
# =====================================================
uploaded_file = st.file_uploader(
    "Choose a road image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Road Image", use_container_width=True)

    if st.button("Run Damage Analysis", type="primary"):
        with st.spinner("Sending image to AI Backend server..."):
            try:
                # Prepare payload
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                # POST request to Railway
                response = requests.post(BACKEND_URL, files=files)

                if response.status_code == 200:
                    try:
                        result = response.json()
                        
                        damage_type = result.get("damage_type", "Unknown")
                        severity = result.get("severity", "Unknown")
                        confidence = result.get("confidence", 0.0)
                        probabilities = result.get("probabilities", {})

                        st.success("Analysis Complete successfully!")

                        # Metrics Display
                        col1, col2, col3 = st.columns(3)
                        col1.metric(label="Damage Type", value=damage_type)
                        col2.metric(label="Severity Level", value=severity)
                        col3.metric(label="Confidence", value=f"{confidence}%")

                        if probabilities:
                            st.write("### 📊 Distribution Probabilities")
                            for name, percentage in probabilities.items():
                                st.write(f"**{name}**")
                                st.progress(percentage / 100)
                                st.caption(f"{percentage}%")
                                
                    except ValueError:
                        st.error("🔴 Backend returned a 200 OK success status, but the content wasn't valid JSON text.")
                        st.text_area("Raw Server Output Data:", value=response.text, height=250)
                else:
                    st.error(f"🔴 Backend Error (Status Code: {response.status_code})")
                    st.text_area("Server Raw Response Logs:", value=response.text, height=250)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Railway server. Please wait a moment.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
