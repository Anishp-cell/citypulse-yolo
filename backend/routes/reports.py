"""Reporting flow: analyze a photo (AI + location + nearby hazards), then submit it as a report."""

import threading
from datetime import UTC, datetime
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.deps import current_user
from backend.domain.geo import bbox_around, haversine_m
from backend.domain.priority import hazard_priority
from backend.domain.status import MERGEABLE_STATUSES, HazardStatus, record_creation
from backend.ml.detector import POTHOLE, suggest_severity
from backend.models import HAZARD_TYPES, SEVERITIES, Follow, Hazard, Report, Upload, User
from backend.ratelimit import DAY, HOUR, rate_limit
from backend.services.images import (
    MAX_BYTES,
    ImageTooLarge,
    UnsupportedImage,
    make_public_crop,
    make_thumb,
    process_upload,
)
from backend.services.storage import new_key

router = APIRouter(prefix="/api/reports", tags=["reports"])
# Decoding + inference is memory-heavy; never run more than two at once on a 512 MB instance.
_ANALYZE_SLOTS = threading.BoundedSemaphore(2)

AI_MIN_CONFIDENCE = 0.25
NEARBY_RADIUS_M = 30.0
MERGE_RADIUS_M = 20.0
ATTACH_MAX_M = 50.0
SEVERITY_ORDER = {s: i for i, s in enumerate(SEVERITIES)}


def _clip(value: str | None, n: int) -> str | None:
    return value[:n] if value else value


def _nearby(db: Session, lat: float, lng: float, radius_m: float, hazard_type: str | None = None):
    min_lat, max_lat, min_lng, max_lng = bbox_around(lat, lng, radius_m)
    q = db.query(Hazard).filter(Hazard.lat.between(min_lat, max_lat), Hazard.lng.between(min_lng, max_lng),
                                Hazard.status.in_(list(MERGEABLE_STATUSES)))
    if hazard_type:
        q = q.filter(Hazard.type == hazard_type)
    found = [(haversine_m(lat, lng, h.lat, h.lng), h) for h in q.all()]
    return sorted([(d, h) for d, h in found if d <= radius_m], key=lambda t: t[0])


def _thumb_url(request: Request, hazard: Hazard) -> str | None:
    cover = hazard.reports[0].thumb_key if hazard.reports else None
    return request.app.state.storage.url(cover, public=True) if cover else None


@router.post("/analyze", dependencies=[Depends(rate_limit("analyze", 20, HOUR))])
def analyze(request: Request,
            file: UploadFile = File(...),
            lat: float = Form(..., ge=-90, le=90),
            lng: float = Form(..., ge=-180, le=180),
            accuracy_m: float | None = Form(None, ge=0),
            location_source: Literal["gps", "manual"] = Form("gps"),
            user: User = Depends(current_user),
            db: Session = Depends(get_db)):
    if abs(lat) < 1e-6 and abs(lng) < 1e-6:
        raise HTTPException(status_code=422, detail="invalid_location")
    raw = file.file.read(MAX_BYTES + 1)
    with _ANALYZE_SLOTS:
        try:
            img = process_upload(raw)
        except ImageTooLarge:
            raise HTTPException(status_code=413, detail="image_too_large")
        except UnsupportedImage:
            raise HTTPException(status_code=415, detail="unsupported_image")
        del raw
        bgr = cv2.imdecode(np.frombuffer(img.full_jpeg, np.uint8), cv2.IMREAD_COLOR)
        dets = [d for d in request.app.state.detector.detect(bgr) if d.label == POTHOLE]
    boxes = [list(d.box) for d in dets]
    confidence = round(max((d.confidence for d in dets), default=0.0), 4)
    ai = {"is_pothole": confidence >= AI_MIN_CONFIDENCE, "confidence": confidence,
          "severity_suggestion": suggest_severity(dets, bgr.shape), "boxes": boxes}

    storage = request.app.state.storage
    crop = make_public_crop(img.full_jpeg, [tuple(b) for b in boxes])
    keys = {"image_key": new_key("full"), "public_image_key": new_key("public"), "thumb_key": new_key("thumbs")}
    storage.put(keys["image_key"], img.full_jpeg, "image/jpeg", public=False)
    storage.put(keys["public_image_key"], crop, "image/jpeg", public=True)
    storage.put(keys["thumb_key"], make_thumb(crop), "image/jpeg", public=True)

    place = request.app.state.geocoder.reverse(db, lat, lng)
    upload = Upload(user_id=user.id, ai=ai, lat=lat, lng=lng, accuracy_m=accuracy_m,
                    location_source=location_source, place=place.__dict__, **keys)
    db.add(upload)
    db.commit()

    nearby = [{"id": h.id, "type": h.type, "status": h.status.value, "distance_m": round(d, 1),
               "report_count": h.report_count, "thumb_url": _thumb_url(request, h)}
              for d, h in _nearby(db, lat, lng, NEARBY_RADIUS_M)]
    return {"upload_id": upload.id, "ai": ai, "place": place.__dict__, "nearby": nearby,
            "preview_url": storage.url(keys["public_image_key"], public=True)}


class SubmitIn(BaseModel):
    upload_id: str
    hazard_type: Literal[HAZARD_TYPES]  # type: ignore[valid-type]
    severity: Literal[SEVERITIES] = "medium"  # type: ignore[valid-type]
    note: str | None = Field(default=None, max_length=500)
    attach_to_hazard_id: int | None = None


def _result(report: Report, hazard: Hazard) -> dict:
    first_report_id = min(r.id for r in hazard.reports)
    return {"hazard_id": hazard.id, "report_id": report.id, "created_new": report.id == first_report_id,
            "status": hazard.status.value}


@router.post("", dependencies=[Depends(rate_limit("reports", 30, DAY))])
def submit(body: SubmitIn,
           idempotency_key: str | None = Header(default=None, max_length=80),
           user: User = Depends(current_user),
           db: Session = Depends(get_db)):
    if idempotency_key:
        existing = db.query(Report).filter_by(user_id=user.id, idempotency_key=idempotency_key).first()
        if existing:  # a retry of a request that already succeeded
            return _result(existing, existing.hazard)

    upload = db.get(Upload, body.upload_id)
    if upload is None or upload.user_id != user.id:
        raise HTTPException(status_code=404, detail="upload_not_found")
    if upload.consumed_at is not None:
        raise HTTPException(status_code=409, detail="upload_already_used")

    now = datetime.now(UTC)
    ai = upload.ai or {}
    hazard: Hazard | None = None
    if body.attach_to_hazard_id is not None:
        hazard = db.get(Hazard, body.attach_to_hazard_id)
        if hazard is None or hazard.status not in MERGEABLE_STATUSES:
            raise HTTPException(status_code=404, detail="hazard_not_found")
        if haversine_m(upload.lat, upload.lng, hazard.lat, hazard.lng) > ATTACH_MAX_M:
            raise HTTPException(status_code=422, detail="too_far")
    else:
        candidates = _nearby(db, upload.lat, upload.lng, MERGE_RADIUS_M, body.hazard_type)
        hazard = candidates[0][1] if candidates else None

    if hazard is None:
        ai_ok = body.hazard_type == POTHOLE and bool(ai.get("is_pothole"))
        place = upload.place or {}
        hazard = Hazard(type=body.hazard_type, severity=body.severity, lat=upload.lat, lng=upload.lng,
                        status=HazardStatus.open if ai_ok else HazardStatus.pending_review, ai_verified=ai_ok,
                        address=_clip(place.get("display"), 300), road=_clip(place.get("road"), 200),
                        ward=_clip(place.get("ward"), 120), city=_clip(place.get("city"), 80),
                        cover_image_key=upload.public_image_key, report_count=1,
                        first_reported_at=now, last_reported_at=now, updated_at=now)
        db.add(hazard)
        db.flush()
        record_creation(db, hazard, actor=user)
    else:
        already = db.query(Report).filter_by(hazard_id=hazard.id, user_id=user.id).first() is not None
        if not already:
            hazard.report_count += 1
        if SEVERITY_ORDER[body.severity] > SEVERITY_ORDER[hazard.severity]:
            hazard.severity = body.severity
        hazard.last_reported_at = now
        hazard.updated_at = now

    report = Report(hazard_id=hazard.id, user_id=user.id, image_key=upload.image_key,
                    public_image_key=upload.public_image_key, thumb_key=upload.thumb_key,
                    ai_label=POTHOLE if ai.get("is_pothole") else None, ai_confidence=ai.get("confidence"),
                    ai_boxes=ai.get("boxes"), location_source=upload.location_source,
                    gps_accuracy_m=upload.accuracy_m, lat=upload.lat, lng=upload.lng, note=body.note,
                    idempotency_key=idempotency_key)
    db.add(report)
    upload.consumed_at = now
    if db.get(Follow, (user.id, hazard.id)) is None:
        db.add(Follow(user_id=user.id, hazard_id=hazard.id))
    db.flush()
    db.refresh(hazard)
    hazard.priority_score = hazard_priority(hazard, now)
    db.commit()
    return _result(report, hazard)
