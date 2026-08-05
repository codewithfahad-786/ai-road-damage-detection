from fastapi import FastAPI, File, UploadFile, HTTPException
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

# =====================================================

# CHECK REQUIRED FILES

# =====================================================

print("========================================")
print("AI Road Damage Detection API")
print("========================================")

print("Backend directory:", BASE_DIR)
print("Model path:", MODEL_PATH)

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

# =====================================================

# LOAD MODEL

# =====================================================

print("Loading road damage model...")

model = tf.keras.models.load_model(
MODEL_PATH,
compile=False
)

print("Model loaded successfully!")
print("Model:", model.name)

# =====================================================

# LOAD CLASS LABELS

# =====================================================

with open(
CLASS_LABELS_PATH,
"r",
encoding="utf-8"
) as f:
class_labels = json.load(f)

# Convert list format to dictionary

if isinstance(class_labels, list):

```
class_labels = {
    i: name
    for i, name in enumerate(class_labels)
}
```

# Convert JSON string keys to integer keys

elif isinstance(class_labels, dict):

```
class_labels = {
    int(key): value
    for key, value in class_labels.items()
}
```

print("Classes:")
print(class_labels)

# =====================================================

# LOAD SEVERITY MAPPING

# =====================================================

with open(
SEVERITY_MAPPING_PATH,
"r",
encoding="utf-8"
) as f:
severity_mapping = json.load(f)

print("Severity mapping loaded successfully!")

# =====================================================

# IMAGE PREPROCESSING

# =====================================================

IMG_SIZE = 224

def preprocess_image(image):

```
# Convert to RGB
image = image.convert("RGB")

# Resize
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
```

# =====================================================

# ROOT API

# =====================================================

@app.get("/")
def home():

```
return {
    "message": "AI Road Damage Detection API Running",
    "status": "success",
    "model_loaded": True
}
```

# =====================================================

# HEALTH CHECK

# =====================================================

@app.get("/health")
def health():

```
return {
    "status": "healthy",
    "model_loaded": model is not None
}
```

# =====================================================

# PREDICTION API

# =====================================================

@app.post("/predict")
async def predict(
file: UploadFile = File(...)
):

```
# Check file type
if not file.content_type or not file.content_type.startswith(
    "image/"
):
    raise HTTPException(
        status_code=400,
        detail="Please upload a valid image file."
    )

try:

    # Read uploaded image
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # Open image
    image = Image.open(
        io.BytesIO(contents)
    )

    # Preprocess image
    processed_image = preprocess_image(
        image
    )

    # Model prediction
    prediction = model.predict(
        processed_image,
        verbose=0
    )

    # Get predicted class
    predicted_index = int(
        np.argmax(prediction[0])
    )

    # Confidence
    confidence = float(
        np.max(prediction[0])
    )

    # Get damage type
    damage_type = class_labels.get(
        predicted_index,
        "Unknown"
    )

    # Get severity
    severity = severity_mapping.get(
        damage_type,
        "Unknown"
    )

    # Calculate probabilities
    probabilities = {}

    for i, name in class_labels.items():

        if i < len(prediction[0]):

            probabilities[name] = round(
                float(prediction[0][i]) * 100,
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

except HTTPException:
    raise

except Exception as e:

    raise HTTPException(
        status_code=500,
        detail=f"Prediction failed: {str(e)}"
    )
```
