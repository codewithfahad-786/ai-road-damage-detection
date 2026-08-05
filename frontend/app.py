import streamlit as st
import requests
from PIL import Image

API_URL = "https://ai-road-damage-detection-production.up.railway.app"

st.set_page_config(
page_title="AI Road Damage Detection",
page_icon="🛣️",
layout="centered"
)

st.title("🛣️ AI Road Damage Detection")
st.write("Upload a road image to detect road damage type and severity.")

st.divider()

uploaded_file = st.file_uploader(
"Upload Road Image",
type=["jpg", "jpeg", "png"]
)

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
                API_URL + "/predict",
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

                severity_info = result.get(
                    "severity",
                    {}
                )

                severity = severity_info.get(
                    "severity",
                    "Unknown"
                )

                recommendation = severity_info.get(
                    "recommendation",
                    "No recommendation available"
                )

                probabilities = result.get(
                    "probabilities",
                    {}
                )

                st.success("Prediction completed successfully!")

                st.subheader("Detection Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Damage Type",
                        damage_type
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{float(confidence):.2f}%"
                    )

                st.subheader("Severity")

                st.write(
                    "Severity: " + str(severity)
                )

                st.write(
                    "Recommendation: " + str(recommendation)
                )

                st.subheader("Class Probabilities")

                for class_name, probability in probabilities.items():

                    probability = float(probability)

                    st.write(
                        f"{class_name}: {probability:.2f}%"
                    )

                    st.progress(
                        min(max(probability / 100, 0.0), 1.0)
                    )

            else:

                st.error(
                    f"Backend returned error: {response.status_code}"
                )

                st.code(response.text)

        except requests.exceptions.Timeout:

            st.error(
                "Backend request timed out. Please try again."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to Railway backend."
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )
```

st.divider()

st.caption(
"AI-Based Smart Road Damage Detection & Severity Assessment System"
)
