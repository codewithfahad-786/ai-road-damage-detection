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

BASE_DIR = os.path.dirname(os.path.abspath(**file**))

# =====================================================

# FILE PATHS

# =====================================================

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

# =====================================================

# LOAD MODEL

# =====================================================

print("Loading road damage model...")

model = tf.keras.models.load_model(
MODEL_PATH
)

print("Model loaded successfully!")

# =====================================================

# LOAD CLASS LABELS

# =====================================================

with open(
CLASS_LABELS_PATH,
"r"
) as f:

```
class_labels = json.load(f)
```

# Convert list format to dictionary

if isinstance(class_labels, list):

```
class_labels = {
    i: name
    for i, name in enumerate(class_labels)
}
```

# Convert JSON keys to integers

class_labels = {
int(key): value
for key, value in class_labels.items()
}

print("Class labels:")
print(class_labels)

# =====================================================

# LOAD SEVERITY MAPPING

# =====================================================

with open(
SEVERITY_PATH,
"r"
) as f:

```
severity_mapping = json.load(f)
```

print("Severity mapping loaded successfully!")

# =====================================================

# IMAGE SETTINGS

# =====================================================

IMG_SIZE = 224

# =====================================================

# IMAGE PREPROCESSING

# =====================================================

def preprocess_image(image):

```
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
```

# =====================================================

# ROOT ENDPOINT

# =====================================================

@app.get("/")
def home():

```
return {
    "message": "AI Road Damage Detection API Running",
    "status": "success"
}
```

# =====================================================

# HEALTH ENDPOINT

# =====================================================

@app.get("/health")
def health():

```
return {
    "status": "healthy",
    "model_loaded": True
}
```

# =====================================================

# PREDICTION ENDPOINT

# =====================================================

@app.post("/predict")
async def predict(
file: UploadFile = File(...)
):

```
# -------------------------------------------------
# Read uploaded image
# -------------------------------------------------

contents = await file.read()

image = Image.open(
    io.BytesIO(contents)
)


# -------------------------------------------------
# Preprocess image
# -------------------------------------------------

processed_image = preprocess_image(
    image
)


# -------------------------------------------------
# Model prediction
# -------------------------------------------------

prediction = model.predict(
    processed_image,
    verbose=0
)


# -------------------------------------------------
# Get predicted class
# -------------------------------------------------

predicted_index = int(
    np.argmax(prediction[0])
)


# -------------------------------------------------
# Get confidence
# -------------------------------------------------

confidence = float(
    np.max(prediction[0])
)


# -------------------------------------------------
# Get damage type
# -------------------------------------------------

damage_type = class_labels[
    predicted_index
]


# -------------------------------------------------
# Get severity
# -------------------------------------------------

severity = severity_mapping.get(
    damage_type,
    "Unknown"
)


# -------------------------------------------------
# Class probabilities
# -------------------------------------------------

probabilities = {}

for index, name in class_labels.items():

    probabilities[name] = round(
        float(prediction[0][index]) * 100,
        2
    )


# -------------------------------------------------
# Return result
# -------------------------------------------------

return {

    "damage_type": damage_type,

    "severity": severity,

    "confidence": round(
        confidence * 100,
        2
    ),

    "probabilities": probabilities

}
```
