import streamlit as st
import requests

st.title("🚧 Road Damage Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# APKA BACKEND URL (Change this to your actual deployed URL if not local)
BACKEND_URL = "http://localhost:8080/predict" 

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            # Prepare file payload
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            # Send request to FastAPI
            response = requests.post(BACKEND_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Damage Type: {result['damage_type']}")
                st.info(f"Severity: {result['severity']}")
                st.metric(label="Confidence", value=f"{result['confidence']}%")
            else:
                st.error("Error communicating with backend server.")
