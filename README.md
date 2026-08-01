# 🛣️ AI-Based Smart Road Damage Detection & Severity Assessment System

## 📌 Project Overview

AI-Based Smart Road Damage Detection & Severity Assessment System is a deep learning application that automatically identifies different types of road damage from images using EfficientNetB0. The system also estimates the severity level and provides maintenance recommendations through an interactive Streamlit web application.

---

## 🚀 Features

- Detects five different types of road damage
- EfficientNetB0 Deep Learning Model
- Image Upload Interface
- Confidence Score
- Severity Assessment
- Maintenance Recommendation
- Streamlit Web Application
- Real-time Prediction

---

## 📂 Dataset

Dataset Used:

**RDD2022 (Road Damage Detection Dataset)**

The dataset contains annotated road images collected from multiple countries with YOLO formatted labels.

---

## 🏷️ Damage Classes

- Alligator Crack
- Longitudinal Crack
- Transverse Crack
- Pothole
- Other Damage

---

## 🧠 Deep Learning Model

- TensorFlow 2.21
- Keras
- EfficientNetB0
- Transfer Learning

Input Size:

224 × 224 RGB Images

---

## 📊 Model Performance

| Metric | Value |
|---------|--------|
| Test Accuracy | **80.77%** |
| Total Test Crops | 9,674 |
| Correct Predictions | 7,814 |
| Number of Classes | 5 |

---

## ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- GitHub

---

## 📁 Project Structure

```
Road_Damage_AI_App/
│
├── app.py
├── road_damage_model.keras
├── class_labels.json
├── severity_mapping.json
├── requirements.txt
├── README.md
```

---

## ▶️ Run Locally

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit App

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

Add your Streamlit App URL here.

Example:

https://your-app.streamlit.app

---

## 📷 Application Features

- Upload Image
- AI Prediction
- Confidence Score
- Severity Level
- Maintenance Recommendation

---

## 📈 Future Improvements

- YOLOv11 Road Damage Detection
- Video Processing
- Real-time Camera Detection
- GPS Integration
- Road Damage Reporting System

---

## 👨‍💻 Developed By

**Fahad Khan**

Deep Learning Final Course Project

Software House Training Project

---

## 📜 License

This project is developed for educational and research purposes.
