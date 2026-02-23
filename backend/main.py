from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import cv2
import uuid
from pathlib import Path
from backend.service.traffic_safety_service import TrafficSafetyService

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent  # project root

# Model weights — try relative first, then well-known paths
WEIGHTS_PATH = str(BASE_DIR / "runs_citypulse" / "yolov8n_pothole_vbest2" / "weights" / "best.pt")
if not os.path.exists(WEIGHTS_PATH):
    # Fallback for Docker where weights are copied to /app/model/
    WEIGHTS_PATH = "/app/model/best.pt"

UPLOAD_DIR = Path("static/uploads")
PROCESSED_DIR = Path("static/processed")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- APP INIT ---
app = FastAPI(title="CityPulse API", description="Road Incident Detection & Guidance API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (processed images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve frontend
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

# Initialize Service
print(f"Initializing Service with weights: {WEIGHTS_PATH}")
service = TrafficSafetyService(weights_path=WEIGHTS_PATH)

# --- ROUTES ---

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": service.model is not None}

@app.get("/")
def home():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "✅ CityPulse Backend is Running", "docs": "/docs"}

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Upload an image for analysis. Returns detections, guidance, and annotated image URL."""
    try:
        file_ext = file.filename.split('.')[-1]
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.{file_ext}"
        file_path = UPLOAD_DIR / filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = service.detect_image(str(file_path))

        if "error" in result:
            return JSONResponse(status_code=500, content={"error": result["error"]})

        processed_filename = f"pred_{filename}"
        processed_path = PROCESSED_DIR / processed_filename
        cv2.imwrite(str(processed_path), result["processed_image"])

        image_url = f"/static/processed/{processed_filename}"

        return {
            "success": True,
            "image_url": image_url,
            "detections": result["detections"],
            "highest_severity": result["highest_severity"],
            "guidance": result["guidance"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

VIDEO_DIR = Path("static/videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """Upload a video for analysis. Returns aggregated detections, timeline, keyframe, and guidance."""
    try:
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ('mp4', 'avi', 'mov', 'mkv', 'webm'):
            raise HTTPException(status_code=400, detail="Unsupported video format. Use MP4, AVI, MOV, MKV, or WebM.")

        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.{file_ext}"
        file_path = VIDEO_DIR / filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = service.detect_video(str(file_path))

        if "error" in result:
            return JSONResponse(status_code=500, content={"error": result["error"]})

        # Save the worst-severity keyframe as an image
        keyframe_filename = f"keyframe_{unique_id}.jpg"
        keyframe_path = PROCESSED_DIR / keyframe_filename
        if result["processed_image"] is not None:
            cv2.imwrite(str(keyframe_path), result["processed_image"])

        image_url = f"/static/processed/{keyframe_filename}"

        return {
            "success": True,
            "image_url": image_url,
            "detections": result["detections"],
            "highest_severity": result["highest_severity"],
            "guidance": result["guidance"],
            "timeline": result.get("timeline", []),
            "video_info": result.get("video_info", {})
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/notify")
async def send_notification(data: dict):
    """Trigger a simulated notification. Expects: {"type": "accident"|"pothole", "location": "string"}"""
    detection_type = data.get("type")
    location = data.get("location")

    if not detection_type or not location:
        raise HTTPException(status_code=400, detail="Missing type or location")

    success, message = service.send_notification(detection_type, location)
    return {"success": success, "message": message}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
