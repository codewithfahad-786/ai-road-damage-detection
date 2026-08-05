
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io
import os

app = FastAPI(
    title="AI Road Damage Detection API",
    description="CNN Based Road Damage Detection and Severity Assessment",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "road_damage_model.keras"
)

CLASS_LABELS_PATH = os.path.join(
    BASE_DIR,
    "class_labels.json"
)

SEVERITY_PATH = os.path.join(
    BASE_DIR,
    "severity_mapping.json"
)

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

with open(CLASS_LABELS_PATH, "r") as f:
    class_labels = json.load(f)

if isinstance(class_labels, list):
    class_labels = {
        i: name
        for i, name in enumerate(class_labels)
    }

class_labels = {
    int(key): value
    for key, value in class_labels.items()
}

print("Class labels:")
print(class_labels)

with open(SEVERITY_PATH, "r") as f:
    severity_mapping = json.load(f)

print("Severity mapping loaded successfully!")

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

    image = tf.keras.applications.efficientnet.preprocess_input(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


@app.get("/")
def home():

    return {
        "message": "AI Road Damage Detection API Running",
        "status": "success"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    )

    processed_image = preprocess_image(
        image
    )

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = int(
        np.argmax(prediction[0])
    )

    confidence = float(
        np.max(prediction[0])
    )

    damage_type = class_labels[
        predicted_index
    ]

    severity = severity_mapping.get(
        damage_type,
        "Unknown"
    )

    probabilities = {}

    for index, name in class_labels.items():

        probabilities[name] = round(
            float(prediction[0][index]) * 100,
            2
        )

    return {
        "damage_type": damage_type,
        "severity": severity,
        "confidence": round(
            confidence * 100,
            2
        ),
        "probabilities": probabilities
    }
:::

### Ab **sab se pehle** local syntax check

PowerShell mein:

```powershell
cd D:\Road_Damage_AI_App
python -m py_compile backend\main.py
