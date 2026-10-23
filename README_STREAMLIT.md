# AI Road Damage Detection — Streamlit

## What changed
- Removed the dependency on the expired Railway `/predict` API.
- Added a professional Streamlit UI.
- Added image upload and camera capture.
- Loads the existing Keras model locally and uses the preprocessing from your supplied FastAPI `main.py`.
- Shows damage type, confidence, severity, recommendation, and class scores.

## Required files
Make sure these files are committed to the same GitHub repository:
- `app.py` (this updated file)
- `requirements.txt` (use the one in this package)
- `road_model.keras`
- `class_labels.json`
- `severity_mapping.json`

The app searches for the model and JSON files beside `app.py`, or inside `backend/` or `models/`.

## Deploy on Streamlit Community Cloud
1. Upload/replace `app.py` and `requirements.txt` in your existing GitHub repository.
2. Confirm the model file and both JSON files are present in one of the supported paths.
3. Visit https://share.streamlit.io and sign in with GitHub.
4. Create app, select your existing repository and branch.
5. Set **Main file path** to `app.py`.
6. Deploy.

## Important model-size note
If `road_model.keras` is too large for GitHub, do not try to paste its contents into code. Keep it in a supported model storage location and modify the app to download it securely at startup, or use Git LFS if supported by your deployment workflow. Do not commit private tokens or passwords.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important
The prediction pipeline assumes a 224x224 RGB image and `tf.keras.applications.efficientnet.preprocess_input`, matching the FastAPI code supplied with the project. If training used different preprocessing or output activations, inference must be adjusted to match the trained model.
