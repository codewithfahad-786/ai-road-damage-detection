import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Road Damage Detection", layout="centered")

st.title("🔴 AI Road Damage Detection & Severity Assessment")
st.write("Upload a road image to detect damages and assess severity.")

# URL ko bilkul clear aur lock kar diya hai
BACKEND_URL = "https://railway.app"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Road Damage"):
        with st.spinner("Analyzing image... Please wait..."):
            try:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="JPEG")
                img_byte_arr = img_byte_arr.getvalue()
                
                files = {"file": (uploaded_file.name, img_byte_arr, "image/jpeg")}
                
                # Headers lagaye hain taake server clear HTML accept na kare
                headers = {"Accept": "application/json"}
                
                response = requests.post(BACKEND_URL, files=files, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        st.success("Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Damage Type", value=data.get("damage_type", "Unknown"))
                        with col2:
                            st.metric(label="Severity Level", value=data.get("severity", "Unknown"))
                            
                        st.metric(label="Confidence Score", value=f"{data.get('confidence', 0)}%")
                        
                        st.write("### Class Probabilities")
                        st.bar_chart(data.get("probabilities", {}))
                        
                    except Exception as json_err:
                        st.error("🔴 Connection is hitting the wrong server. Details below:")
                        st.write(f"**Target URL used:** `{BACKEND_URL}`")
                        st.write("### Raw Server Response:")
                        st.code(response.text[:500])
                else:
                    st.error(f"🔴 Server error code: {response.status_code}")
                    st.code(response.text[:500])
                    
            except Exception as e:
                st.error(f"System error: {str(e)}")
