import io
import json
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI(
    title="AI Road Damage Detection API",
    description="CNN Based Road Damage Detection and Severity Assessment",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================== #
# FILE PATHS                                            #
# ===================================================== #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# NAAM CHANGE: road_damage_model.keras ko road_model.keras kar diya gaya hai
MODEL_PATH = os.path.join(BASE_DIR, "road_model.keras")
CLASS_LABELS_PATH = os.path.join(BASE_DIR, "class_labels.json")
SEVERITY_MAPPING_PATH = os.path.join(BASE_DIR, "severity_mapping.json")

print("========================================")
print("AI Road Damage Detection API")
print("========================================")
print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)

# ===================================================== #
# CHECK FILES                                           #
# ===================================================== #
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError("Model file not found: " + MODEL_PATH)

if not os.path.isfile(CLASS_LABELS_PATH):
    raise FileNotFoundError("Class labels file not found: " + CLASS_LABELS_PATH)

if not os.path.isfile(SEVERITY_MAPPING_PATH):
    raise FileNotFoundError(
        "Severity mapping file not found: " + SEVERITY_MAPPING_PATH
    )

# ===================================================== #
# LOAD MODEL                                            #
# ===================================================== #
print("Loading road damage model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# ===================================================== #
# LOAD CLASS LABELS                                     #
# ===================================================== #
with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels = json.load(f)

if isinstance(class_labels, list):
    class_labels = {i: name for i, name in enumerate(class_labels)}
elif isinstance(class_labels, dict):
    class_labels = {int(key): value for key, value in class_labels.items()}

print("Class labels:", class_labels)

# ===================================================== #
# LOAD SEVERITY MAPPING                                 #
# ===================================================== #
with open(SEVERITY_MAPPING_PATH, "r", encoding="utf-8") as f:
    severity_mapping = json.load(f)
print("Severity mapping loaded successfully!")

# ===================================================== #
# IMAGE PREPROCESSING                                   #
# ===================================================== #
IMG_SIZE = 224


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


# ===================================================== #
# ROOT ENDPOINT                                         #
# ===================================================== #
@app.get("/")
def home():
    return {
        "message": "AI Road Damage Detection API Running",
        "status": "success",
        "model_loaded": True,
    }


# ===================================================== #
# HEALTH ENDPOINT                                       #
# ===================================================== #
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


# ===================================================== #
# PREDICTION ENDPOINT                                   #
# ===================================================== #
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="File type is missing.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Please upload a valid image."
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image, verbose=0)
        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        damage_type = class_labels.get(predicted_index, "Unknown")
        severity = severity_mapping.get(damage_type, "Unknown")

        probabilities = {}
        for index, name in class_labels.items():
            if index < len(prediction[0]):
                probabilities[name] = round(
                    float(prediction[0][index]) * 100, 2
                )

        return {
            "damage_type": damage_type,
            "severity": severity,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Prediction failed: " + str(e)
        )
