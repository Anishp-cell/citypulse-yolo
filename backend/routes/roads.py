from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional, List
from backend.database import get_db
from backend.models import RoadSegment, PotholeCluster, RoadSubscription, Verification, User, IncidentStatus
from backend.dependencies import get_current_user
from backend.service.spatial_service import award_user_points, recalculate_road_health
from backend.service.government.pdf_generator import generate_grievance_pdf
from backend.service.government.email_pdf_adapter import EmailPdfGovernmentAdapter
import uuid
import os
from pathlib import Path

router = APIRouter(prefix="/api/v1", tags=["roads_and_rankings"])

PROCESSED_DIR = Path("static/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
email_adapter = EmailPdfGovernmentAdapter()

@router.get("/roads/rankings")
def get_road_rankings(
    city: str = Query("Pune"),
    ward: Optional[str] = Query(None),
    sort: str = Query("worst", pattern="^(worst|best)$"),
    db: Session = Depends(get_db),
):
    """
    Return Road Health Scorecard rankings (Worst Roads vs Best Roads).
    """
    query = db.query(RoadSegment).filter(RoadSegment.city.ilike(f"%{city}%"))
    if ward:
        query = query.filter(RoadSegment.ward.ilike(f"%{ward}%"))

    if sort == "worst":
        query = query.order_by(RoadSegment.health_score.asc())
    else:
        query = query.order_by(RoadSegment.health_score.desc())

    roads = query.limit(20).all()

    # If database is fresh with no roads, provide populated default records
    if not roads:
        defaults = [
            ("MG Road (Camp to Sector 4)", "Central", "Pune", 38.5, 14, 3),
            ("Hinjawadi Phase 1 Spine Road", "Hinjawadi", "Pune", 44.0, 11, 5),
            ("Baner-Pashan Link Road", "Aundh-Baner", "Pune", 72.0, 4, 8),
            ("Koregaon Park North Main Road", "Dhole Patil", "Pune", 92.5, 1, 12),
        ]
        created = []
        for name, w, c, score, active, resolved in defaults:
            r = RoadSegment(
                name=name,
                ward=w,
                city=c,
                health_score=score,
                active_potholes_count=active,
                resolved_potholes_count=resolved,
            )
            db.add(r)
            created.append(r)
        db.commit()
        roads = created

    return {
        "city": city,
        "sort": sort,
        "total": len(roads),
        "roads": [
            {
                "id": r.id,
                "name": r.name,
                "ward": r.ward,
                "city": r.city,
                "health_score": round(r.health_score, 1),
                "active_potholes_count": r.active_potholes_count,
                "resolved_potholes_count": r.resolved_potholes_count,
                "grade": "A" if r.health_score >= 85 else ("B" if r.health_score >= 60 else "F"),
            }
            for r in roads
        ],
    }

@router.post("/roads/{road_id}/watch")
def watch_road(
    road_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Subscribe to road follow-up notifications."""
    road = db.query(RoadSegment).filter(RoadSegment.id == road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road segment not found")

    existing = db.query(RoadSubscription).filter(
        RoadSubscription.user_id == current_user.id,
        RoadSubscription.road_id == road_id,
    ).first()

    if existing:
        db.delete(existing)
        db.commit()
        return {"success": True, "watching": False, "message": f"Unsubscribed from {road.name}"}

    sub = RoadSubscription(user_id=current_user.id, road_id=road_id)
    db.add(sub)
    award_user_points(db, current_user.id, 5, "ROAD_WATCH_SUBSCRIBE", road_id)
    db.commit()
    return {"success": True, "watching": True, "message": f"Subscribed to {road.name} updates"}

@router.post("/clusters/{cluster_id}/verify")
async def verify_pothole_fix(
    cluster_id: int,
    is_fixed: bool = Form(...),
    notes: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Citizen verification of a reported pothole repair.
    Awards +50 karma points upon verification.
    """
    cluster = db.query(PotholeCluster).filter(PotholeCluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Pothole cluster not found")

    image_url = None
    if file:
        file_ext = (file.filename or "verify.jpg").rsplit(".", 1)[-1].lower()
        filename = f"verify_{uuid.uuid4()}.{file_ext}"
        filepath = PROCESSED_DIR / filename
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        image_url = f"/static/processed/{filename}"

    verification = Verification(
        cluster_id=cluster_id,
        user_id=current_user.id,
        image_url=image_url,
        is_fixed=is_fixed,
        notes=notes,
    )
    db.add(verification)

    if is_fixed:
        cluster.status = IncidentStatus.confirmed_fixed
        if cluster.road:
            cluster.road.active_potholes_count = max(0, cluster.road.active_potholes_count - 1)
            cluster.road.resolved_potholes_count += 1
            recalculate_road_health(cluster.road)
        award_user_points(db, current_user.id, 50, "VERIFICATION_FIX", cluster_id)
    else:
        cluster.status = IncidentStatus.disputed
        cluster.priority_score = min(100.0, cluster.priority_score * 1.5)
        award_user_points(db, current_user.id, 20, "VERIFICATION_DISPUTE", cluster_id)

    db.commit()
    return {
        "success": True,
        "cluster_id": cluster_id,
        "new_status": cluster.status.value,
        "points_awarded": 50 if is_fixed else 20,
    }

@router.get("/clusters/{cluster_id}/dossier")
def download_grievance_dossier(
    cluster_id: int,
    db: Session = Depends(get_db),
):
    """
    Generate and download the official Municipal Infrastructure Grievance PDF Dossier.
    """
    cluster = db.query(PotholeCluster).filter(PotholeCluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Pothole cluster not found")

    road_name = cluster.road.name if cluster.road else (cluster.address_text or "Municipal Road")
    ward_name = cluster.road.ward if cluster.road else "Central"
    city = cluster.road.city if cluster.road else "Pune"

    # Find evidence image
    incident = cluster.incidents[0] if cluster.incidents else None
    image_path = None
    if incident and incident.image_url:
        local_rel = incident.image_url.lstrip("/")
        if os.path.exists(local_rel):
            image_path = local_rel

    pdf_file = generate_grievance_pdf(
        ticket_id=f"CP-{cluster.id}",
        road_name=road_name,
        ward_name=ward_name,
        city=city,
        lat=cluster.lat,
        lng=cluster.lng,
        severity=cluster.severity,
        priority_score=cluster.priority_score,
        report_count=cluster.report_count,
        image_path=image_path,
    )

    return FileResponse(
        pdf_file,
        media_type="application/pdf",
        filename=os.path.basename(pdf_file),
    )

@router.post("/clusters/{cluster_id}/dispatch")
async def dispatch_grievance_to_government(
    cluster_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Dispatch official grievance PDF packet to Municipal Ward Engineer.
    """
    cluster = db.query(PotholeCluster).filter(PotholeCluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Pothole cluster not found")

    road_name = cluster.road.name if cluster.road else (cluster.address_text or "Municipal Road")
    ward_name = cluster.road.ward if cluster.road else "Central"
    city = cluster.road.city if cluster.road else "Pune"

    incident = cluster.incidents[0] if cluster.incidents else None
    image_path = incident.image_url.lstrip("/") if (incident and incident.image_url) else None

    dispatch_result = await email_adapter.submit_complaint({
        "cluster_id": cluster.id,
        "ticket_id": f"CP-{cluster.id}",
        "road_name": road_name,
        "ward_name": ward_name,
        "city": city,
        "lat": cluster.lat,
        "lng": cluster.lng,
        "severity": cluster.severity,
        "priority_score": cluster.priority_score,
        "report_count": cluster.report_count,
        "image_path": image_path,
    })

    cluster.status = IncidentStatus.routed
    cluster.government_ticket_id = dispatch_result["ticket_id"]
    cluster.government_dept = "Roads & Traffic Infrastructure Department"
    award_user_points(db, current_user.id, 25, "MUNICIPAL_DISPATCH_BONUS", cluster_id)

    db.commit()
    return dispatch_result

@router.get("/leaderboard")
def get_leaderboard(
    db: Session = Depends(get_db),
    limit: int = 10,
):
    """Get top civic champions by Karma points."""
    top_users = db.query(User).order_by(User.points_balance.desc()).limit(limit).all()
    return {
        "leaderboard": [
            {
                "rank": idx + 1,
                "id": u.id,
                "name": u.full_name or u.email.split('@')[0],
                "points": u.points_balance,
                "trust_score": round(u.trust_score, 2),
                "ward": u.home_ward or "Central",
            }
            for idx, u in enumerate(top_users)
        ]
    }
