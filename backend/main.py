import os
import json
import numpy as np
from PIL import Image
from io import BytesIO

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Road Damage Detection API",
    description="CNN based road damage detection and severity assessment API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "road_damage_model.keras")
CLASS_LABELS_PATH = os.path.join(BASE_DIR, "class_labels.json")
SEVERITY_MAPPING_PATH = os.path.join(BASE_DIR, "severity_mapping.json")


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_LABELS_PATH):
    raise FileNotFoundError(
        f"class_labels.json not found: {CLASS_LABELS_PATH}"
    )

if not os.path.exists(SEVERITY_MAPPING_PATH):
    raise FileNotFoundError(
        f"severity_mapping.json not found: {SEVERITY_MAPPING_PATH}"
    )


# ============================================================
# LOAD CLASS LABELS
# ============================================================

with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels = json.load(f)


# ============================================================
# LOAD SEVERITY MAPPING
# ============================================================

with open(SEVERITY_MAPPING_PATH, "r", encoding="utf-8") as f:
    severity_mapping = json.load(f)


# ============================================================
# NORMALIZE CLASS LABELS
# ============================================================

if isinstance(class_labels, dict):
    # Example:
    # {"0": "Longitudinal_Crack", "1": "Transverse_Crack"}
    CLASS_NAMES = {
        int(k): str(v)
        for k, v in class_labels.items()
    }

elif isinstance(class_labels, list):
    # Example:
    # ["Longitudinal_Crack", "Transverse_Crack", ...]
    CLASS_NAMES = {
        i: str(name)
        for i, name in enumerate(class_labels)
    }

else:
    raise ValueError(
        "class_labels.json must contain either a list or dictionary."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading road damage model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Road damage model loaded successfully.")


# ============================================================
# MODEL INPUT SIZE
# ============================================================

try:
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) >= 3:
        MODEL_HEIGHT = int(input_shape[1])
        MODEL_WIDTH = int(input_shape[2])
    else:
        MODEL_HEIGHT = 224
        MODEL_WIDTH = 224

except Exception:
    MODEL_HEIGHT = 224
    MODEL_WIDTH = 224


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes: bytes):
    try:
        image = Image.open(BytesIO(image_bytes))

        # Convert image to RGB
        image = image.convert("RGB")

        # Resize according to model input
        image = image.resize(
            (MODEL_WIDTH, MODEL_HEIGHT)
        )

        # Convert to NumPy
        image_array = np.array(image, dtype=np.float32)

        # Normalize pixel values
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {str(e)}"
        )


# ============================================================
# GET CLASS NAME
# ============================================================

def get_class_name(class_index: int):
    return CLASS_NAMES.get(
        class_index,
        f"Class_{class_index}"
    )


# ============================================================
# GET SEVERITY
# ============================================================

def get_severity(class_name: str, class_index: int):
    """
    Supports different possible severity_mapping.json formats.
    """

    # Direct class-name lookup
    if isinstance(severity_mapping, dict):

        if class_name in severity_mapping:
            value = severity_mapping[class_name]

            if isinstance(value, dict):
                return (
                    value.get("severity")
                    or value.get("level")
                    or value.get("name")
                    or str(value)
                )

            return str(value)

        # Class index lookup
        if str(class_index) in severity_mapping:
            value = severity_mapping[str(class_index)]

            if isinstance(value, dict):
                return (
                    value.get("severity")
                    or value.get("level")
                    or value.get("name")
                    or str(value)
                )

            return str(value)

    return "Unknown"


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AI Road Damage Detection API is running",
        "status": "success",
        "model": "EfficientNetB0",
        "classes": list(CLASS_NAMES.values()),
        "input_size": f"{MODEL_WIDTH}x{MODEL_HEIGHT}"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# CLASS INFORMATION
# ============================================================

@app.get("/classes")
def get_classes():
    return {
        "classes": CLASS_NAMES
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check file type
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File type could not be detected."
        )

    allowed_types = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG, PNG, or WEBP image."
        )

    # Read image
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    # Preprocess
    processed_image = preprocess_image(
        image_bytes
    )

    # Prediction
    try:
        predictions = model.predict(
            processed_image,
            verbose=0
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(e)}"
        )

    # Convert prediction to NumPy
    predictions = np.asarray(predictions)

    # Handle normal classification output
    if predictions.ndim == 2:
        probabilities = predictions[0]

    elif predictions.ndim == 1:
        probabilities = predictions

    else:
        probabilities = predictions.reshape(-1)

    # Find highest probability
    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    # Class name
    predicted_class = get_class_name(
        predicted_index
    )

    # Severity
    severity = get_severity(
        predicted_class,
        predicted_index
    )

    # All class probabilities
    class_probabilities = {}

    for index, probability in enumerate(probabilities):

        class_name = get_class_name(index)

        class_probabilities[class_name] = round(
            float(probability) * 100,
            2
        )

    # Response
    return {
        "success": True,
        "filename": file.filename,
        "predicted_class": predicted_class,
        "class_index": predicted_index,
        "confidence": round(
            confidence * 100,
            2
        ),
        "severity": severity,
        "class_probabilities": class_probabilities
    }


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
