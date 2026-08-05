import io
import json
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # Import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (Aapka baki model loading aur config code same rahega) ...


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Please upload a valid image."
        )

    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)

        # Prediction
        prediction = model.predict(processed_image, verbose=0)

        # Standard Python type conversion (Crucial for JSON)
        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        damage_type = class_labels.get(predicted_index, "Unknown")
        severity_data = severity_mapping.get(damage_type, "Low/Unknown")

        if isinstance(severity_data, dict):
            severity = severity_data.get("severity", "Low/Unknown")
        else:
            severity = severity_data

        probabilities = {}
        for index, name in class_labels.items():
            if index < len(prediction[0]):
                probabilities[name] = round(
                    float(prediction[0][index]) * 100, 2
                )

        # 🔴 EXPLICIT JSON RESPONSE FORCING (Fixes 200 OK Content issues)
        return JSONResponse(
            content={
                "damage_type": str(damage_type),
                "severity": str(severity),
                "confidence": round(confidence * 100, 2),
                "probabilities": probabilities,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Prediction failed: {str(e)}"}
        )
