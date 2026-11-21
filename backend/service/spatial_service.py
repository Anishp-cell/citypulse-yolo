import math
from datetime import datetime, timezone
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from backend.models import PotholeCluster, RoadSegment, Incident, PointTransaction, User, IncidentStatus

def haversine_distance_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compute distance in meters between two lat/lng points using Haversine formula."""
    R = 6371000  # Radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def find_or_create_road_segment(db: Session, road_name: Optional[str], ward: Optional[str] = "Central", city: str = "Pune") -> RoadSegment:
    """Find or register a road segment."""
    clean_name = (road_name or "MG Road").strip()
    road = db.query(RoadSegment).filter(RoadSegment.name.ilike(f"%{clean_name}%")).first()
    if not road:
        road = RoadSegment(
            name=clean_name,
            ward=ward or "Central",
            city=city,
            health_score=100.0,
            active_potholes_count=0,
            resolved_potholes_count=0,
        )
        db.add(road)
        db.commit()
        db.refresh(road)
    return road

def calculate_priority_score(severity: str, report_count: int, days_old: float = 0.0, is_disputed: bool = False) -> float:
    """
    Mathematical Prioritization Scoring:
    P = min(100, (W_sev * S_sev + W_vol * S_traffic + W_age * ln(1+t)) * M_rep * M_dispute)
    """
    severity_scores = {
        'totaled_vehicle': 40.0,
        'severe_accident': 35.0,
        'moderate_accident': 25.0,
        'pothole': 25.0,
        'minor_accident': 15.0,
        'no_accident': 5.0,
    }
    s_sev = severity_scores.get(severity.lower(), 20.0)
    s_traffic = 20.0  # City arterial default
    s_critical = 10.0  # Proximity index
    s_age = 3.0 * math.log(1.0 + max(0.0, days_old))

    base_score = s_sev + s_traffic + s_critical + s_age
    m_rep = 1.0 + 0.15 * min(4, max(0, report_count - 1))
    m_dispute = 1.5 if is_disputed else 1.0

    priority = min(100.0, base_score * m_rep * m_dispute)
    return round(priority, 1)

def find_or_merge_cluster(
    db: Session,
    lat: float,
    lng: float,
    severity: str,
    address_text: Optional[str] = None,
    distance_threshold_m: float = 15.0
) -> PotholeCluster:
    """
    Query nearby clusters within distance_threshold_m (15m).
    If found, increments report_count and merges. Otherwise, creates a new cluster.
    """
    now = datetime.now(timezone.utc)
    
    # Query candidate clusters within ~0.001 deg (~100m bounding box for efficiency)
    delta_deg = 0.001
    candidates = db.query(PotholeCluster).filter(
        PotholeCluster.lat >= lat - delta_deg,
        PotholeCluster.lat <= lat + delta_deg,
        PotholeCluster.lng >= lng - delta_deg,
        PotholeCluster.lng <= lng + delta_deg,
    ).all()

    matched_cluster = None
    for cand in candidates:
        dist = haversine_distance_meters(lat, lng, cand.lat, cand.lng)
        if dist <= distance_threshold_m:
            matched_cluster = cand
            break

    if matched_cluster:
        matched_cluster.report_count += 1
        matched_cluster.last_reported_at = now
        matched_cluster.priority_score = calculate_priority_score(
            matched_cluster.severity,
            matched_cluster.report_count,
            days_old=0.0
        )
        db.commit()
        db.refresh(matched_cluster)
        return matched_cluster

    # Create new cluster
    road = find_or_create_road_segment(db, address_text)
    new_priority = calculate_priority_score(severity, 1, 0.0)

    cluster = PotholeCluster(
        road_id=road.id,
        lat=lat,
        lng=lng,
        address_text=address_text,
        severity=severity,
        priority_score=new_priority,
        status=IncidentStatus.submitted,
        report_count=1,
        first_reported_at=now,
        last_reported_at=now,
    )
    db.add(cluster)

    # Update road metrics
    road.active_potholes_count += 1
    recalculate_road_health(road)

    db.commit()
    db.refresh(cluster)
    return cluster

def recalculate_road_health(road: RoadSegment):
    """
    RHI = 100 - min(100, (active_potholes / length_km) * 12.5)
    """
    length = max(0.5, road.length_km)
    penalty = (road.active_potholes_count / length) * 12.5
    road.health_score = max(0.0, round(100.0 - min(100.0, penalty), 1))

def award_user_points(db: Session, user_id: int, points: int, reason: str, reference_id: Optional[int] = None):
    """Award karma points and log transaction."""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.points_balance += points
        user.trust_score = min(1.0, user.trust_score + 0.02)
        tx = PointTransaction(
            user_id=user_id,
            amount=points,
            reason=reason,
            reference_id=reference_id,
        )
        db.add(tx)
        db.commit()
