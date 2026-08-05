import streamlit as st
import requests
from PIL import Image

st.set_page_config(
page_title="AI Road Damage Detection",
page_icon="🛣️",
layout="centered"
)

st.title("🛣️ AI Road Damage Detection")
st.write("Upload a road image to detect damage type and severity.")

# Railway Backend URL

API_URL = "https://ai-road-damage-detection-production.up.railway.app/predict"

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
                API_URL,
                files=files,
                timeout=120
            )

            if response.status_code == 200:

                result = response.json()

                damage_type = result.get(
                    "damage_type",
                    "Unknown"
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

                confidence = result.get(
                    "confidence",
                    0
                )

                probabilities = result.get(
                    "probabilities",
                    {}
                )

                st.success("Detection completed successfully!")

                st.subheader("📋 Detection Result")

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

                if probabilities:

                    st.subheader(
                        "📊 Class Probabilities"
                    )

                    for class_name, probability in probabilities.items():

                        st.write(
                            f"**{class_name}:** "
                            f"{probability:.2f}%"
                        )

                        st.progress(
                            min(
                                max(
                                    int(probability),
                                    0
                                ),
                                100
                            )
                        )

            else:

                st.error(
                    f"Backend Error: "
                    f"{response.status_code}"
                )

                st.code(response.text)

        except requests.exceptions.Timeout:

            st.error(
                "The backend took too long to respond. "
                "Please try again."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the Railway backend. "
                "Please check that the Railway service is running."
            )

        except Exception as e:

            st.error(
                f"An unexpected error occurred: {str(e)}"
            )
```
