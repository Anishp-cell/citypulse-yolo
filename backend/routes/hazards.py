"""Public hazard reads: map listing, hazard page, city scorecard, status metadata."""

import statistics
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session, selectinload

from backend.db import get_db
from backend.deps import current_user_optional
from backend.domain.status import ACTIVE_STATUSES, PUBLIC_STATUSES, HazardStatus
from backend.models import HAZARD_TYPES, Hazard, User, aware
from backend.services.present import can_view, detail, summary

router = APIRouter(prefix="/api", tags=["hazards"])

STATUS_GROUPS = {
    "active": ACTIVE_STATUSES,
    "fixed": frozenset({HazardStatus.fixed_unverified, HazardStatus.fixed}),
    "all": PUBLIC_STATUSES,
}
ROAD_SEVERITY_WEIGHT = {"low": 5, "medium": 10, "high": 20}
MAX_RESULTS = 500


def parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    try:
        min_lng, min_lat, max_lng, max_lat = (float(v) for v in bbox.split(","))
    except ValueError:
        raise HTTPException(status_code=422, detail="bbox must be minLng,minLat,maxLng,maxLat")
    if not (-90 <= min_lat <= max_lat <= 90 and -180 <= min_lng <= max_lng <= 180):
        raise HTTPException(status_code=422, detail="bbox out of range")
    return min_lng, min_lat, max_lng, max_lat


@router.get("/hazards")
def list_hazards(request: Request, bbox: str,
                 status: Literal["active", "fixed", "all"] = "active",
                 type: Literal[HAZARD_TYPES] | None = None,  # type: ignore[valid-type]
                 db: Session = Depends(get_db)):
    min_lng, min_lat, max_lng, max_lat = parse_bbox(bbox)
    q = (db.query(Hazard).options(selectinload(Hazard.reports))
         .filter(Hazard.lat.between(min_lat, max_lat), Hazard.lng.between(min_lng, max_lng),
                 Hazard.status.in_(list(STATUS_GROUPS[status]))))
    if type:
        q = q.filter(Hazard.type == type)
    rows = q.order_by(Hazard.priority_score.desc()).limit(MAX_RESULTS + 1).all()
    storage = request.app.state.storage
    return {"hazards": [summary(h, storage) for h in rows[:MAX_RESULTS]], "truncated": len(rows) > MAX_RESULTS}


def get_visible_hazard(db: Session, hazard_id: int, user: User | None) -> Hazard:
    h = db.get(Hazard, hazard_id)
    if h is None or not can_view(h, user):
        raise HTTPException(status_code=404, detail="hazard_not_found")
    return h


@router.get("/hazards/{hazard_id}")
def get_hazard(hazard_id: int, request: Request, user: User | None = Depends(current_user_optional),
               db: Session = Depends(get_db)):
    h = get_visible_hazard(db, hazard_id, user)
    return detail(db, h, user, request.app.state.storage, request.app.state.settings.app_url)


@router.get("/scorecard")
def scorecard(city: str = Query("Pune", max_length=80), db: Session = Depends(get_db)):
    now = datetime.now(UTC)
    hazards = (db.query(Hazard).filter(Hazard.city.ilike(city), Hazard.status.in_(list(PUBLIC_STATUSES))).all())

    wards: dict[str, list[Hazard]] = defaultdict(list)
    roads: dict[tuple[str, str | None], list[Hazard]] = defaultdict(list)
    for h in hazards:
        wards[h.ward or "Unknown"].append(h)
        if h.road and h.status in ACTIVE_STATUSES:
            roads[(h.road, h.ward)].append(h)

    ward_rows = []
    for ward, hs in wards.items():
        fix_days = [(aware(h.fixed_at) - aware(h.first_reported_at)).total_seconds() / 86400 for h in hs
                    if h.status == HazardStatus.fixed and h.fixed_at and aware(h.fixed_at) >= now - timedelta(days=90)]
        cohort = [h for h in hs if now - timedelta(days=120) <= aware(h.first_reported_at) <= now - timedelta(days=30)]
        fixed_fast = [h for h in cohort if h.fixed_at and h.status == HazardStatus.fixed
                      and aware(h.fixed_at) - aware(h.first_reported_at) <= timedelta(days=30)]
        ward_rows.append({
            "ward": ward,
            "active": sum(1 for h in hs if h.status in ACTIVE_STATUSES),
            "fixed_90d": len(fix_days),
            "median_days_to_fix_90d": round(statistics.median(fix_days), 1) if fix_days else None,
            "pct_fixed_30d": round(100 * len(fixed_fast) / len(cohort), 1) if cohort else None,
        })
    ward_rows.sort(key=lambda w: (-w["active"], w["ward"]))

    road_rows = [{"road": road, "ward": ward, "active": len(hs),
                  "health": max(0, 100 - sum(ROAD_SEVERITY_WEIGHT.get(h.severity, 10) for h in hs))}
                 for (road, ward), hs in roads.items()]
    road_rows.sort(key=lambda r: (r["health"], -r["active"]))
    return {"city": city, "wards": ward_rows, "worst_roads": road_rows[:10]}


@router.get("/meta/statuses")
def statuses():
    return [{"value": s.value, "active": s in ACTIVE_STATUSES, "public": s in PUBLIC_STATUSES} for s in HazardStatus]


@router.get("/hazards/{hazard_id}/complaint")
def complaint(hazard_id: int, request: Request, lang: Literal["en", "hi", "mr"] = "en",
              user: User | None = Depends(current_user_optional), db: Session = Depends(get_db)):
    from backend.services.complaint import build_complaint

    h = get_visible_hazard(db, hazard_id, None)  # complaint packs exist only for public hazards
    return build_complaint(h, lang, request.app.state.settings.app_url)


@router.get("/hazards/{hazard_id}/complaint.pdf")
def complaint_pdf(hazard_id: int, request: Request, db: Session = Depends(get_db)):
    from fastapi.responses import Response

    from backend.services.complaint_pdf import build_pdf

    h = get_visible_hazard(db, hazard_id, None)
    photo = request.app.state.storage.get(h.cover_image_key, public=True) if h.cover_image_key else None
    pdf = build_pdf(h, request.app.state.settings.app_url, photo)
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'inline; filename="CityPulse-complaint-CP-{h.id}.pdf"',
        "Cache-Control": "public, max-age=300"})
