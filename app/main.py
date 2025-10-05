from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from service.inference import Detector

# Initialize FastAPI
app = FastAPI(title="CityPulse YOLO Inference API")

# Load model once at startup
detector = Detector(weights="runs_citypulse/yolov8n_pothole_vbest2/weights/best.pt")

@app.get("/")
def home():
    return {"message": "🚀 CityPulse YOLOv8 API is running on Railway!"}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run detection
    result = detector.predict_image(temp_path)
    os.remove(temp_path)

    return JSONResponse(content=result)

