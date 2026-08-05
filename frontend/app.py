import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Road Damage Detection", layout="centered")

st.title("🔴 AI Road Damage Detection & Severity Assessment")
st.write("Upload a road image to detect damages and assess severity.")

# WARNING: Jab aapka FastAPI backend kisi live hosting (jaise Render ya HuggingFace) 
# par deploy ho jaye, to is URL ko apne live API link se zaroor badal dein.
BACKEND_URL = "http://127.0.0" 

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Uploaded image ko screen par dikhane ke liye
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Road Damage"):
        with st.spinner("Analyzing image... Please wait..."):
            try:
                # PIL Image ko bytes mein convert karna taake network par bheja ja sake
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else "JPEG")
                img_byte_arr = img_byte_arr.getvalue()
                
                # FastAPI ke mutabiq payload file taiyar karna
                files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                
                # Backend API ko request bhejna
                response = requests.post(BACKEND_URL, files=files)
                
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
                    
                    # Graph Chart dikhane ke liye
                    st.write("### Class Probabilities")
                    st.bar_chart(data.get("probabilities", {}))
                else:
                    st.error(f"Backend error code: {response.status_code}")
                    st.json(response.json())
                    
            except Exception as e:
                st.error(f"Could not connect to backend server: {str(e)}")
