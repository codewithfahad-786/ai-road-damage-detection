import os
# Must be set BEFORE importing tensorflow to successfully block logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import io
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Suppress additional Keras output flags
tf.get_logger().setLevel('ERROR')

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

# ===================================================== #
# FILE PATHS                                            #
# ===================================================== #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "road_damage_model.keras")
CLASS_LABELS_PATH = os.path.join(BASE_DIR, "class_labels.json")
SEVERITY_MAPPING_PATH = os.path.join(BASE_DIR, "severity_mapping.json")

# ===================================================== #
# CHECK FILES                                           #
# ===================================================== #
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError("Model file not found: " + MODEL_PATH)

if not os.path.isfile(CLASS_LABELS_PATH):
    raise FileNotFoundError("Class labels file not found: " + CLASS_LABELS_PATH)

if not os.path.isfile(SEVERITY_MAPPING_PATH):
    raise FileNotFoundError("Severity mapping file not found: " + SEVERITY_MAPPING_PATH)

# ===================================================== #
# LOAD MODEL & CONFIGS                                  #
# ===================================================== #
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels = json.load(f)

if isinstance(class_labels, list):
    class_labels = {i: name for i, name in enumerate(class_labels)}
elif isinstance(class_labels, dict):
    class_labels = {int(key): value for key, value in class_labels.items()}

with open(SEVERITY_MAPPING_PATH, "r", encoding="utf-8") as f:
    severity_mapping = json.load(f)

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
# ENDPOINTS                                             #
# ===================================================== #
@app.get("/")
def home():
    return {
        "message": "AI Road Damage Detection API Running",
        "status": "success",
        "model_loaded": True
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="File type is missing.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Get raw prediction and flatten to a reliable 1D array
        raw_prediction = model.predict(processed_image, verbose=0)
        prediction_vector = raw_prediction[0].tolist()  # Converts directly to standard Python floats
        
        predicted_index = int(np.argmax(prediction_vector))
        confidence = float(prediction_vector[predicted_index])
        
        damage_type = class_labels.get(predicted_index, "Unknown")
        severity = severity_mapping.get(damage_type, "Unknown")
        
        probabilities = {}
        for index, name in class_labels.items():
            if index < len(prediction_vector):
                probabilities[name] = round(float(prediction_vector[index]) * 100, 2)
                
        return {
            "damage_type": damage_type,
            "severity": severity,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(e))
