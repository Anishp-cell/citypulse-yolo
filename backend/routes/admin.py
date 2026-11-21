from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date
from datetime import datetime, timedelta
from backend.database import get_db
from backend.models import Incident, User, UserRole, IncidentStatus
from backend.dependencies import require_officer_or_admin, require_admin

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats")
def get_admin_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_officer_or_admin),
):
    """Dashboard statistics for admin/officer users."""
    total = db.query(Incident).count()

    # Active = detected or notified (not yet acknowledged/resolved/closed)
    active = db.query(Incident).filter(
        Incident.status.in_([IncidentStatus.detected, IncidentStatus.notified])
    ).count()

    # Resolved today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    resolved_today = db.query(Incident).filter(
        Incident.status == IncidentStatus.resolved,
        Incident.created_at >= today_start,
    ).count()

    # By severity
    severity_rows = db.query(
        Incident.severity, func.count(Incident.id)
    ).group_by(Incident.severity).all()
    by_severity = {row[0]: row[1] for row in severity_rows}

    # Daily trend (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    trend_rows = db.query(
        cast(Incident.created_at, Date).label("date"),
        func.count(Incident.id).label("count"),
    ).filter(
        Incident.created_at >= thirty_days_ago
    ).group_by(
        cast(Incident.created_at, Date)
    ).order_by("date").all()

    daily_trend = [
        {"date": row.date.isoformat() if row.date else "", "count": row.count}
        for row in trend_rows
    ]

    # Hotspots — top locations by incident count
    hotspot_rows = db.query(
        Incident.address_text, func.count(Incident.id).label("count")
    ).filter(
        Incident.address_text.isnot(None),
        Incident.address_text != "",
    ).group_by(
        Incident.address_text
    ).order_by(
        func.count(Incident.id).desc()
    ).limit(10).all()

    hotspots = [
        {"location": row[0], "count": row[1]}
        for row in hotspot_rows
    ]

    return {
        "total": total,
        "active": active,
        "resolved_today": resolved_today,
        "avg_response": "—",  # placeholder — need timestamp tracking for real value
        "by_severity": by_severity,
        "daily_trend": daily_trend,
        "hotspots": hotspots,
    }


@router.get("/heatmap")
def get_heatmap_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_officer_or_admin),
):
    """Return incident coordinates for the heatmap layer."""
    rows = db.query(
        Incident.lat, Incident.lng, Incident.severity
    ).filter(
        Incident.lat.isnot(None),
        Incident.lng.isnot(None),
    ).all()

    # Map severity to intensity for heatmap weighting
    intensity_map = {
        "no_accident": 0.1,
        "minor_accident": 0.3,
        "moderate_accident": 0.5,
        "severe_accident": 0.8,
        "totaled_vehicle": 1.0,
        "pothole": 0.4,
    }

    points = [
        {
            "lat": row.lat,
            "lng": row.lng,
            "intensity": intensity_map.get(row.severity, 0.5),
        }
        for row in rows
    ]

    return {"points": points, "count": len(points)}


@router.get("/users")
def list_users(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List all users. Admin only."""
    total = db.query(User).count()
    users = db.query(User).order_by(User.created_at.desc()).offset(skip).limit(min(limit, 100)).all()

    return {
        "total": total,
        "users": [
            {
                "id": u.id,
                "email": u.email,
                "phone": u.phone,
                "role": u.role.value if u.role else "citizen",
                "district": u.district,
                "created_at": u.created_at.isoformat() if u.created_at else None,
            }
            for u in users
        ],
    }


@router.patch("/users/{user_id}/role")
def update_user_role(
    user_id: int,
    data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Change a user's role. Admin only."""
    new_role = data.get("role")
    if not new_role:
        raise HTTPException(status_code=400, detail="Missing 'role' field")
    try:
        role_val = UserRole(new_role)
    except ValueError:
        valid = [r.value for r in UserRole]
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {valid}")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.role = role_val
    db.commit()
    return {"success": True, "user_id": user_id, "role": new_role}
