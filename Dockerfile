FROM python:3.10-slim

WORKDIR /app

# System dependencies install karein (Pillow/TensorFlow ke liye)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Backend requirements copy aur install karein
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pura project copy karein
COPY . .

# Railway ki port par Uvicorn run karein
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
