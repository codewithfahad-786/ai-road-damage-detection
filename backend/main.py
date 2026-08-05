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


# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = "road_damage_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Model:", model.name)


# =====================================================
# LOAD CLASS LABELS
# =====================================================

with open("class_labels.json", "r") as f:
    class_labels = json.load(f)


# If JSON is list format
if isinstance(class_labels, list):

    class_labels = {
        i: name 
        for i, name in enumerate(class_labels)
    }


print("Classes:")
print(class_labels)


print("Classes:")
print(class_labels)


# =====================================================
# LOAD SEVERITY MAPPING
# =====================================================

with open("severity_mapping.json", "r") as f:
    severity_mapping = json.load(f)


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

IMG_SIZE = 224


def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image = np.array(
        image,
        dtype=np.float32
    )


    # EfficientNet preprocessing
    image = tf.keras.applications.efficientnet.preprocess_input(
        image
    )


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
        "message":
        "AI Road Damage Detection API Running"
    }



# =====================================================
# PREDICTION API
# =====================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read image

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    )


    # preprocess

    processed_image = preprocess_image(
        image
    )


    # prediction

    prediction = model.predict(
        processed_image,
        verbose=0
    )


    predicted_index = int(
        np.argmax(prediction)
    )


    confidence = float(
        np.max(prediction)
    )


    damage_type = class_labels[
        predicted_index
    ]


    # severity

    severity = severity_mapping.get(
        damage_type,
        "Unknown"
    )


    probabilities = {}

    for i, name in class_labels.items():

        probabilities[name] = round(
            float(prediction[0][i]) * 100,
            2
        )


    return {

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
