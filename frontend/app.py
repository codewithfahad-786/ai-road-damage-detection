import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Road Damage Detection", layout="centered")

st.title("🔴 AI Road Damage Detection & Severity Assessment")
st.write("Upload a road image to detect damages and assess severity.")

# IMPORTANT NOTE: 
# "http://127.0.0" sirf aapke computer (localhost) par kaam karega.
# Jab aap frontend Streamlit Cloud par deploy karte hain, to backend (FastAPI) ko 
# Render.com ya Hugging Face par deploy karke uska public URL yahan lagana lazmi hai.
BACKEND_URL = "http://127.0.0" 

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Uploaded image ko read aur display karna
    image = Image.open(uploaded_file)
    
    # FIXED: use_column_width ko use_container_width se replace kiya hai
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Road Damage"):
        with st.spinner("Analyzing image... Please wait..."):
            try:
                # PIL Image ko bytes mein convert karna
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else "JPEG")
                img_byte_arr = img_byte_arr.getvalue()
                
                # Payload taiyar karna
                files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                
                # API Call
                response = requests.post(BACKEND_URL, files=files, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("Analysis Complete!")
                    
                    # Dashboard Layout
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Damage Type", value=data.get("damage_type", "Unknown"))
                    with col2:
                        st.metric(label="Severity Level", value=data.get("severity", "Unknown"))
                        
                    st.metric(label="Confidence Score", value=f"{data.get('confidence', 0)}%")
                    
                    # Graph Chart
                    st.write("### Class Probabilities")
                    st.bar_chart(data.get("probabilities", {}))
                else:
                    st.error(f"Backend error code: {response.status_code}")
                    st.json(response.json())
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Error: Streamlit Cloud aapke local computer (127.0.0.1) se connect nahi ho sakta. Aapko apna FastAPI backend kisi hosting platform (Render, Hugging Face) par deploy karna hoga aur uska link yahan dalna hoga.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
