from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date
from typing import Optional
from datetime import datetime, timedelta
from backend.database import engine, get_db, Base
from backend.models import Incident, Notification, IncidentStatus, UserRole
from backend.routes import auth
from backend.dependencies import get_current_user, require_officer_or_admin, User
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

# --- FILE VALIDATION ---
MAX_IMAGE_SIZE = 50 * 1024 * 1024    # 50 MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024   # 500 MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


async def save_upload_with_size_limit(upload: UploadFile, path: Path, max_bytes: int) -> int:
    """Stream-write an uploaded file, raising HTTP 413 if it exceeds max_bytes."""
    size = 0
    try:
        with open(path, "wb") as f:
            while True:
                chunk = await upload.read(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum allowed size is {max_bytes // (1024 * 1024)} MB."
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    return size


# --- APP INIT ---
app = FastAPI(title="CityPulse API", description="Road Incident Detection & Guidance API")

# CORS — restrict to explicit origins in production via ALLOWED_ORIGINS env var
# e.g. ALLOWED_ORIGINS="https://citypulse.ai,https://app.citypulse.ai"
_origins_raw = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _origins_raw.split(",") if o.strip()] or ["http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
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

# Initialize Database
print("Initializing Database...")
Base.metadata.create_all(bind=engine)

# --- ROUTES ---
# Include auth router
app.include_router(auth.router)

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
async def analyze_image(
    file: UploadFile = File(...), 
    location: str = Form(None), 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) # Requires ANY logged in user (citizen, officer, or admin)
):
    """Upload an image for analysis. Returns detections, guidance, and annotated image URL."""
    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image type. Allowed: JPEG, PNG, WebP."
            )
        file_ext = (file.filename or "upload").rsplit(".", 1)[-1].lower()
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.{file_ext}"
        file_path = UPLOAD_DIR / filename

        await save_upload_with_size_limit(file, file_path, MAX_IMAGE_SIZE)

        result = service.detect_image(str(file_path))

        if "error" in result:
            return JSONResponse(status_code=500, content={"error": result["error"]})

        processed_filename = f"pred_{filename}"
        processed_path = PROCESSED_DIR / processed_filename
        cv2.imwrite(str(processed_path), result["processed_image"])

        image_url = f"/static/processed/{processed_filename}"

        # Save to DB
        db_incident = Incident(
            image_url=image_url,
            severity=result["highest_severity"],
            detection_results=result["detections"],
            guidance=result["guidance"],
            address_text=location,
            reported_by=current_user.id
        )
        db.add(db_incident)
        db.commit()
        db.refresh(db_incident)

        return {
            "success": True,
            "incident_id": db_incident.id,
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
async def analyze_video(
    file: UploadFile = File(...), 
    location: str = Form(None), 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) # Requires ANY logged in user
):
    """Upload a video for analysis. Returns aggregated detections, timeline, keyframe, and guidance."""
    try:
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ('mp4', 'avi', 'mov', 'mkv', 'webm'):
            raise HTTPException(status_code=400, detail="Unsupported video format. Use MP4, AVI, MOV, MKV, or WebM.")

        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.{file_ext}"
        file_path = VIDEO_DIR / filename

        await save_upload_with_size_limit(file, file_path, MAX_VIDEO_SIZE)

        result = service.detect_video(str(file_path))

        if "error" in result:
            return JSONResponse(status_code=500, content={"error": result["error"]})

        # Save the worst-severity keyframe as an image
        keyframe_filename = f"keyframe_{unique_id}.jpg"
        keyframe_path = PROCESSED_DIR / keyframe_filename
        if result["processed_image"] is not None:
            cv2.imwrite(str(keyframe_path), result["processed_image"])

        image_url = f"/static/processed/{keyframe_filename}"

        # Save to DB
        db_incident = Incident(
            video_url=f"/static/videos/{filename}",
            image_url=image_url,
            severity=result["highest_severity"],
            detection_results=result["detections"],
            guidance=result["guidance"],
            address_text=location,
            reported_by=current_user.id
        )
        db.add(db_incident)
        db.commit()
        db.refresh(db_incident)

        return {
            "success": True,
            "incident_id": db_incident.id,
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
async def send_notification_endpoint(
    data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Send a real notification to authorities.
    Expects: {"type": "accident"|"pothole", "location": "string", "incident_id": int|null}"""
    detection_type = data.get("type")
    location = data.get("location")
    incident_id = data.get("incident_id")

    if not detection_type or not location:
        raise HTTPException(status_code=400, detail="Missing type or location")
    if detection_type not in ("accident", "pothole"):
        raise HTTPException(status_code=400, detail="type must be 'accident' or 'pothole'")

    success, message = service.send_notification(detection_type, location, incident_id=incident_id)

    # Persist notification records when incident_id is provided
    if incident_id:
        incident = db.query(Incident).filter(Incident.id == incident_id).first()
        if incident:
            if os.environ.get("SENDGRID_API_KEY"):
                db.add(Notification(
                    incident_id=incident_id,
                    channel="email",
                    recipient=os.environ.get("ALERT_EMAIL", "authority@example.com"),
                    status="sent" if success else "failed",
                ))
            if os.environ.get("TWILIO_ACCOUNT_SID"):
                db.add(Notification(
                    incident_id=incident_id,
                    channel="sms",
                    recipient=os.environ.get("ALERT_PHONE", "emergency"),
                    status="sent" if success else "failed",
                ))
            if success:
                incident.status = IncidentStatus.notified
            db.commit()

    return {"success": success, "message": message}


@app.get("/api/incidents")
def list_incidents(
    skip: int = 0,
    limit: int = 20,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List incidents. Citizens see only their own; officers/admins see all."""
    query = db.query(Incident)
    if current_user.role == UserRole.citizen:
        query = query.filter(Incident.reported_by == current_user.id)
    if severity:
        query = query.filter(Incident.severity == severity)
    if status:
        query = query.filter(Incident.status == status)

    total = query.count()
    incidents = query.order_by(Incident.created_at.desc()).offset(skip).limit(min(limit, 100)).all()

    return {
        "total": total,
        "incidents": [
            {
                "id": inc.id,
                "severity": inc.severity,
                "status": inc.status.value if inc.status else None,
                "address_text": inc.address_text,
                "image_url": inc.image_url,
                "created_at": inc.created_at.isoformat() if inc.created_at else None,
                "reported_by": inc.reported_by,
            }
            for inc in incidents
        ],
    }


@app.get("/api/incidents/{incident_id}")
def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single incident by ID."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    if current_user.role == UserRole.citizen and incident.reported_by != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this incident")
    return {
        "id": incident.id,
        "severity": incident.severity,
        "status": incident.status.value if incident.status else None,
        "address_text": incident.address_text,
        "image_url": incident.image_url,
        "video_url": incident.video_url,
        "detection_results": incident.detection_results,
        "guidance": incident.guidance,
        "lat": incident.lat,
        "lng": incident.lng,
        "created_at": incident.created_at.isoformat() if incident.created_at else None,
        "reported_by": incident.reported_by,
    }


@app.patch("/api/incidents/{incident_id}/status")
def update_incident_status(
    incident_id: int,
    data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_officer_or_admin),
):
    """Update incident status. Officers/admins only."""
    new_status = data.get("status")
    if not new_status:
        raise HTTPException(status_code=400, detail="Missing 'status' field")
    try:
        status_val = IncidentStatus(new_status)
    except ValueError:
        valid = [s.value for s in IncidentStatus]
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid}")
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = status_val
    db.commit()
    return {"success": True, "incident_id": incident_id, "status": new_status}


@app.get("/api/stats")
def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Aggregate stats for the dashboard. Officers/admins see all; citizens see own."""
    base_q = db.query(Incident)
    if current_user.role == UserRole.citizen:
        base_q = base_q.filter(Incident.reported_by == current_user.id)

    total = base_q.count()

    # Breakdown by severity
    severity_rows = (
        base_q.with_entities(Incident.severity, func.count(Incident.id))
        .group_by(Incident.severity)
        .all()
    )
    by_severity = {row[0]: row[1] for row in severity_rows}

    # Breakdown by status
    status_rows = (
        base_q.with_entities(Incident.status, func.count(Incident.id))
        .group_by(Incident.status)
        .all()
    )
    by_status = {(row[0].value if row[0] else "unknown"): row[1] for row in status_rows}

    # Last 7 days daily counts
    seven_days_ago = datetime.utcnow() - timedelta(days=6)
    daily_rows = (
        base_q.filter(Incident.created_at >= seven_days_ago)
        .with_entities(
            cast(Incident.created_at, Date).label("day"),
            func.count(Incident.id).label("count"),
        )
        .group_by("day")
        .order_by("day")
        .all()
    )
    daily = [{"date": str(row.day), "count": row.count} for row in daily_rows]

    return {
        "total": total,
        "by_severity": by_severity,
        "by_status": by_status,
        "daily_last_7_days": daily,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
