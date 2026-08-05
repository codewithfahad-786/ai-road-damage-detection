import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Road Damage Detection", layout="centered")

st.title("🔴 AI Road Damage Detection & Severity Assessment")
st.write("Upload a road image to detect damages and assess severity.")

# Aapka bilkul verified live URL
BACKEND_URL = "https://railway.app"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image ko read aur store karna safely
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Road Damage"):
        with st.spinner("Analyzing image... Please wait..."):
            try:
                # 1. PIL Image ko standard RGB mein convert karna (Tafseeli verification)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # 2. Image ko bytes mein clean save karna
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="JPEG")
                img_byte_arr = img_byte_arr.getvalue()
                
                # 3. Payload files dict ready karna (MIME type specify karna zaroori hai)
                files = {"file": (uploaded_file.name, img_byte_arr, "image/jpeg")}
                
                # 4. Post request send karna safely
                response = requests.post(BACKEND_URL, files=files, timeout=60)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        st.success("Analysis Complete!")
                        
                        # Dashboard Layout
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Damage Type", value=data.get("damage_type", "Unknown"))
                        with col2:
                            st.metric(label="Severity Level", value=data.get("severity", "Unknown"))
                            
                        st.metric(label="Confidence Score", value=f"{data.get('confidence', 0)}%")
                        
                        # Graph Chart showing probabilities
                        st.write("### Class Probabilities")
                        st.bar_chart(data.get("probabilities", {}))
                        
                    except Exception as json_err:
                        st.error("🔴 Backend code executed but sent unexpected text format.")
                        st.write("### Raw Server Response:")
                        st.code(response.text)
                else:
                    st.error(f"🔴 Server returned an error code: {response.status_code}")
                    st.write("### Error Details:")
                    st.code(response.text)
                    
            except requests.exceptions.Timeout:
                st.error("❌ Timeout Error: AI Model processing took more than 60 seconds.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Error: Streamlit Cloud cannot reach your Railway API right now.")
            except Exception as e:
                st.error(f"An unexpected system error occurred: {str(e)}")
