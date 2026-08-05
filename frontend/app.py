import streamlit as st
import requests

# Aapka asli live Railway API link yahan bilkul sahi set hai
API_URL = "https://railway.app"

st.set_page_config(page_title="AI Road Damage Detection", layout="centered")

st.title("🛣 AI Road Damage Detection System")
st.write("Upload a road image to detect structural damage and access severity.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # FIX: use_column_width ko use_container_width se replace kar diya hai
    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
    
    if st.button("Analyze Road Damage"):
        with st.spinner("Analyzing image through CNN model... Please wait."):
            # Image ko binary bytes mein convert karke FastAPI ko bhejna
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                # Railway API ke /predict endpoint par request bhejein
                response = requests.post(f"{API_URL}predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Analysis Complete!")
                    
                    # Metrics alignment
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Detected Damage Type", value=result['damage_type'].replace('_', ' '))
                    with col2:
                        st.metric(label="Severity Level", value=result['severity'])
                        
                    st.metric(label="Model Confidence", value=f"{result['confidence']}%")
                    
                    # Probabilities bar chart dikhane ke liye
                    st.write("### 📊 Prediction Probabilities across all classes:")
                    st.bar_chart(result['probabilities'])
                else:
                    st.error(f"Backend Server Error ({response.status_code}): {response.text}")
                    
            except Exception as e:
                st.error(f"Could not connect to FastAPI server at Railway. Error details: {str(e)}")
