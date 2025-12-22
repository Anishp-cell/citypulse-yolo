# 🚨 CityPulse: AI-Powered Road Safety System
> **Accident Detection + Real-Time Medical Guidance**

I’m thrilled to share my latest end-to-end computer vision project that brings together YOLOv8 object detection and AI-driven emergency response — built to save lives and improve urban infrastructure.

## 🎯 The Problem

Road accidents claim **1.3M lives annually** (WHO), and infrastructure issues like potholes cause thousands of avoidable incidents. In emergencies, every minute of response time is critical.

## 💡 The Solution: CityPulse

A full-stack AI application that:
- **Detects accidents and potholes** in real time from images/videos
- **Provides severity-based medical guidance** for bystanders
- **Automates notifications** to relevant authorities with documentation

---

## 🔧 Technical Architecture

### 1. Custom YOLOv8 Model
Built a unified detection system by merging car accident + pothole datasets.
- **6-class classification**: `no_accident`, `minor`, `moderate`, `severe`, `totaled_vehicle`, `pothole`
- **End-to-end preprocessing pipeline**: Mixed annotation formats (XML → YOLO)
- **Optimized training**: Augmentation, mixed-precision, and class balancing
- **Real-time inference**: Runs efficiently on CPU/GPU

**📊 Best Model Results (Epoch 58):**
- **Precision**: 0.63
- **Recall**: 0.63
- **mAP@50**: 0.57
- **mAP@50-95**: 0.33

### 2. Intelligent Guidance System
Designed a structured medical knowledge base mapped to 4 severity levels:
- From **minor** (delayed symptom monitoring) → **severe** (CPR, spinal stabilization) → **critical** (multi-casualty triage)
- Clear, actionable steps simplified for non-medical bystanders

### 3. Local AI Integration (Mistral)
Integrated Mistral for personalized medical guidance.
- **Fast response**: <2s response time with short-context inference
- **Privacy-first**: Runs locally on CPU → no cloud dependency
- **Reliable**: Graceful fallback to structured guidance when needed

### 4. Production-Ready Web Application
- **Dual upload**: Support for images/videos with real-time annotated detection
- **Severity visualization**: Color-coded indicators for quick decision-making
- **Interactive UI**: Progressive video frame tracking with expandable guidance sections
- **Backend**: Handles model caching, file management, notifications, and codecs

### 5. Notification System
- **Automated alerts**: Templates for emergency services & road departments
- **Rich Documentation**: Emails include annotated images + severity-level documentation
- **Logging**: Incident logging with timestamps for future accountability

---

## 📊 Technical Highlights

- **Computer Vision**: Multi-dataset YOLOv8 pipeline, real-time inference, video buffering
- **ML Engineering**: Transfer learning, hyperparameter tuning, quantization considerations
- **Software Engineering**: Modular codebase, YAML configs, type hints, error handling
- **System Design**: Scalable for traffic camera feeds, optimized for low latency (<2s), and extensible for municipal integration

## 💻 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## 🤝 Impact Potential

CityPulse is more than detection — it’s a comprehensive emergency response ecosystem that:
1. **Reduces emergency response time** via automation
2. **Provides life-saving guidance** to non-medical bystanders
3. **Helps municipalities prioritize repairs** & allocate resources
4. **Scales to city-wide deployment** for smart infrastructure

---
*Built with ❤️ for safer cities.*
