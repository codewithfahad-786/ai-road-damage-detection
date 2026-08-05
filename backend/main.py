from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io
import os

# ============================================================

# APP

# ============================================================

app = FastAPI(
title="AI Road Damage Detection API",
description="CNN Based Road Damage Detection and Severity Assessment",
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

MODEL_PATH = os.path.join(
BASE_DIR,
"road_damage_model.keras"
)

CLASS_LABELS_PATH = os.path.join(
BASE_DIR,
"class_labels.json"
)

SEVERITY_MAPPING_PATH = os.path.join(
BASE_DIR,
"severity_mapping.json"
)

# ============================================================

# CHECK REQUIRED FILES

# ============================================================

print("========================================")
print("AI Road Damage Detection API")
print("========================================")

print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)

if not os.path.isfile(MODEL_PATH):
raise FileNotFoundError(
f"Model file not found: {MODEL_PATH}. "
"Make sure road_damage_model.keras is available in the backend folder."
)

if not os.path.isfile(CLASS_LABELS_PATH):
raise FileNotFoundError(
f"class_labels.json not found: {CLASS_LABELS_PATH}"
)

if not os.path.isfile(SEVERITY_MAPPING_PATH):
raise FileNotFoundError(
f"severity_mapping.json not found: {SEVERITY_MAPPING_PATH}"
)

# ============================================================

# LOAD MODEL

# ============================================================

print("Loading road damage model...")

model = tf.keras.models.load_model(
MODEL_PATH,
compile=False
)

print("Model loaded successfully!")
print("Model name:", model.name)

# ============================================================

# LOAD CLASS LABELS

# ============================================================

with open(
CLASS_LABELS_PATH,
"r",
encoding="utf-8"
) as f:
class_labels = json.load(f)

# Convert list format to dictionary

if isinstance(class_labels, list):
class_labels = {
str(i): name
for i, name in enumerate(class_labels)
}

# Make sure dictionary keys are strings

class_labels = {
str(key): value
for key, value in class_labels.items()
}

print("Class labels:", class_labels)

# ============================================================

# LOAD SEVERITY MAPPING

# ============================================================

with open(
SEVERITY_MAPPING_PATH,
"r",
encoding="utf-8"
) as f:
severity_mapping = json.load(f)

print("Severity mapping loaded successfully.")

# ============================================================

# IMAGE SETTINGS

# ============================================================

IMG_SIZE = 224

# ============================================================

# IMAGE PREPROCESSING

# ============================================================

def preprocess_image(image: Image.Image):

```
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
```

# ============================================================

# ROOT ENDPOINT

# ============================================================

@app.get("/")
def home():

```
return {
    "status": "success",
    "message": "AI Road Damage Detection API is running",
    "model": "EfficientNetB0",
    "version": "1.0.0"
}
```

# ============================================================

# HEALTH CHECK

# ============================================================

@app.get("/health")
def health():

```
return {
    "status": "healthy",
    "model_loaded": model is not None
}
```

# ============================================================

# PREDICTION ENDPOINT

# ============================================================

@app.post("/predict")
async def predict(
file: UploadFile = File(...)
):

```
# --------------------------------------------------------
# Check file type
# --------------------------------------------------------

if not file.content_type or not file.content_type.startswith(
    "image/"
):
    raise HTTPException(
        status_code=400,
        detail="Please upload a valid image file."
    )


# --------------------------------------------------------
# Read uploaded image
# --------------------------------------------------------

try:

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    )

except Exception:

    raise HTTPException(
        status_code=400,
        detail="Unable to read the uploaded image."
    )


# --------------------------------------------------------
# Preprocess image
# --------------------------------------------------------

try:

    processed_image = preprocess_image(
        image
    )

except Exception as e:

    raise HTTPException(
        status_code=500,
        detail=f"Image preprocessing failed: {str(e)}"
    )


# --------------------------------------------------------
# Model prediction
# --------------------------------------------------------

try:

    prediction = model.predict(
        processed_image,
        verbose=0
    )

except Exception as e:

    raise HTTPException(
        status_code=500,
        detail=f"Model prediction failed: {str(e)}"
    )


# --------------------------------------------------------
# Prediction index
# --------------------------------------------------------

predicted_index = int(
    np.argmax(prediction[0])
)


confidence = float(
    np.max(prediction[0])
)


# --------------------------------------------------------
# Damage type
# --------------------------------------------------------

damage_type = class_labels.get(
    str(predicted_index),
    f"Class {predicted_index}"
)


# --------------------------------------------------------
# Severity
# --------------------------------------------------------

severity = severity_mapping.get(
    damage_type,
    "Unknown"
)


# --------------------------------------------------------
# Probabilities
# --------------------------------------------------------

probabilities = {}

for i, probability in enumerate(prediction[0]):

    class_name = class_labels.get(
        str(i),
        f"Class {i}"
    )

    probabilities[class_name] = round(
        float(probability) * 100,
        2
    )


# --------------------------------------------------------
# Response
# --------------------------------------------------------

return {

    "success": True,

    "damage_type": damage_type,

    "severity": severity,

    "confidence": round(
        confidence * 100,
        2
    ),

    "probabilities": probabilities

}
```
