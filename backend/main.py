```python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io
import os


# =====================================================
# APP INITIALIZATION
# =====================================================

app = FastAPI(
    title="AI Road Damage Detection API",
    description="CNN Based Road Damage Detection and Severity Assessment",
    version="1.0"
)


# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# BASE DIRECTORY
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "road_damage_model.keras"
)

print("Loading model...")
print("Model path:", MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")
print("Model:", model.name)


# =====================================================
# LOAD CLASS LABELS
# =====================================================

CLASS_LABELS_PATH = os.path.join(
    BASE_DIR,
    "class_labels.json"
)

with open(
    CLASS_LABELS_PATH,
    "r"
) as f:
    class_labels = json.load(f)


# Convert list format to dictionary
if isinstance(class_labels, list):

    class_labels = {
        i: name
        for i, name in enumerate(class_labels)
    }


# Convert dictionary keys to integers
if isinstance(class_labels, dict):

    class_labels = {
        int(k): v
        for k, v in class_labels.items()
    }


print("Classes:")
print(class_labels)


# =====================================================
# LOAD SEVERITY MAPPING
# =====================================================

SEVERITY_PATH = os.path.join(
    BASE_DIR,
    "severity_mapping.json"
)

with open(
    SEVERITY_PATH,
    "r"
) as f:
    severity_mapping = json.load(f)


print("Severity mapping loaded successfully!")


# =====================================================
# IMAGE SETTINGS
# =====================================================

IMG_SIZE = 224


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # Convert to NumPy array
    image = np.array(
        image,
        dtype=np.float32
    )

    # EfficientNet preprocessing
    image = tf.keras.applications.efficientnet.preprocess_input(
        image
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# =====================================================
# ROOT API
# =====================================================

@app.get("/")
def home():

    return {
        "status": "success",
        "message": "AI Road Damage Detection API Running",
        "endpoint": "/predict"
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =====================================================
# PREDICTION API
# =====================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )


        # -------------------------------------------------
        # PREPROCESS IMAGE
        # -------------------------------------------------

        processed_image = preprocess_image(
            image
        )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            processed_image,
            verbose=0
        )


        # -------------------------------------------------
        # GET PREDICTED CLASS
        # -------------------------------------------------

        predicted_index = int(
            np.argmax(prediction[0])
        )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = float(
            np.max(prediction[0])
        )


        # -------------------------------------------------
        # DAMAGE TYPE
        # -------------------------------------------------

        damage_type = class_labels.get(
            predicted_index,
            "Unknown"
        )


        # -------------------------------------------------
        # SEVERITY
        # -------------------------------------------------

        severity = severity_mapping.get(
            damage_type,
            "Unknown"
        )


        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        probabilities = {}

        for i, name in class_labels.items():

            probabilities[name] = round(
                float(prediction[0][i]) * 100,
                2
            )


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {

            "status": "success",

            "damage_type":
                damage_type,

            "severity":
                severity,

            "confidence":
                round(
                    confidence * 100,
                    2
                ),

            "probabilities":
                probabilities
        }


    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }
```
