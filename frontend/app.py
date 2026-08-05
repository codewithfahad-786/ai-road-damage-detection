import streamlit as st
import requests
from PIL import Image

# =========================

# Configuration

# =========================

API_URL = "https://ai-road-damage-detection-production.up.railway.app"

# =========================

# Page Configuration

# =========================

st.set_page_config(
page_title="AI Road Damage Detection",
page_icon="🛣️",
layout="centered"
)

# =========================

# Title

# =========================

st.title("🛣️ AI Road Damage Detection")
st.write(
"Upload a road image and the AI model will detect the "
"type and severity of road damage."
)

st.divider()

# =========================

# Image Upload

# =========================

uploaded_file = st.file_uploader(
"Upload a road image",
type=["jpg", "jpeg", "png"]
)

# =========================

# Prediction

# =========================

if uploaded_file is not None:

```
image = Image.open(uploaded_file)

st.image(
    image,
    caption="Uploaded Road Image",
    use_container_width=True
)

if st.button("🔍 Detect Road Damage"):

    with st.spinner("Analyzing image..."):

        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=120
            )

            if response.status_code == 200:

                result = response.json()

                damage_type = result.get(
                    "damage_type",
                    "Unknown"
                )

                confidence = result.get(
                    "confidence",
                    0
                )

                severity_data = result.get(
                    "severity",
                    {}
                )

                severity = severity_data.get(
                    "severity",
                    "Unknown"
                )

                recommendation = severity_data.get(
                    "recommendation",
                    "No recommendation available"
                )

                probabilities = result.get(
                    "probabilities",
                    {}
                )

                st.success("✅ Prediction completed!")

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

                st.write(
                    f"**Severity:** {severity}"
                )

                st.write(
                    f"**Recommendation:** {recommendation}"
                )

                st.subheader("📈 Class Probabilities")

                for class_name, probability in probabilities.items():

                    st.write(
                        f"**{class_name}:** "
                        f"{probability:.2f}%"
                    )

                    st.progress(
                        min(
                            max(
                                float(probability) / 100,
                                0.0
                            ),
                            1.0
                        )
                    )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.code(
                    response.text
                )

        except requests.exceptions.Timeout:

            st.error(
                "⏱️ Request timed out. "
                "The Railway backend is taking too long to respond."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to the Railway backend."
            )

        except Exception as e:

            st.error(
                f"❌ Error: {str(e)}"
            )
```

# =========================

# Footer

# =========================

st.divider()

st.caption(
"AI-Based Smart Road Damage Detection & Severity Assessment System"
)
