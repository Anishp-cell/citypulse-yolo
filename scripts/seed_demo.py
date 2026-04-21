"""
seed_demo.py — Populate the database with realistic demo incidents for presentations.

Usage:
    python scripts/seed_demo.py

Requires DATABASE_URL to be set (or defaults to localhost citypulse).
Set SEED_USER_EMAIL / SEED_USER_PASSWORD to create/use a specific demo account,
otherwise defaults to demo@citypulse.ai / DemoPass123!
"""

import os
import sys
import random
from datetime import datetime, timedelta

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import engine, get_db, Base
from backend.models import Incident, User, IncidentStatus, UserRole
from backend.security import get_password_hash
from sqlalchemy.orm import Session

# ── Config ────────────────────────────────────────────────────────────────────

DEMO_EMAIL    = os.getenv("SEED_USER_EMAIL", "demo@citypulse.ai")
DEMO_PASSWORD = os.getenv("SEED_USER_PASSWORD", "DemoPass123!")
N_INCIDENTS   = int(os.getenv("SEED_COUNT", "40"))

# Realistic city locations worldwide (lat, lng, address)
LOCATIONS = [
    (40.7128, -74.0060, "Broadway & Canal St, New York, USA"),
    (40.7580, -73.9855, "Times Square, New York, USA"),
    (51.5074, -0.1278,  "Westminster Bridge, London, UK"),
    (51.5007, -0.1246,  "Lambeth Bridge Rd, London, UK"),
    (48.8566,  2.3522,  "Rue de Rivoli, Paris, France"),
    (48.8738,  2.2950,  "Avenue des Champs-Elysees, Paris, France"),
    (35.6762, 139.6503, "Shibuya Crossing, Tokyo, Japan"),
    (35.6895, 139.6917, "Shinjuku, Tokyo, Japan"),
    (1.3521,  103.8198, "Orchard Road, Singapore"),
    (1.2966,  103.8520, "Marina Bay, Singapore"),
    (-33.8688, 151.2093,"George St, Sydney, Australia"),
    (37.7749, -122.4194,"Market St, San Francisco, USA"),
    (37.3382, -121.8863,"San Jose Downtown, USA"),
    (19.0760,  72.8777, "Western Express Highway, Mumbai"),
    (28.6139,  77.2090, "Connaught Place, New Delhi"),
    (12.9716,  77.5946, "MG Road, Bangalore"),
    (22.5726,  88.3639, "Park Street, Kolkata"),
    (-23.5505, -46.6333,"Avenida Paulista, Sao Paulo, Brazil"),
    (55.7558,  37.6176, "Tverskaya Street, Moscow, Russia"),
    (31.2304, 121.4737, "Nanjing Road, Shanghai, China"),
]

SEVERITIES = [
    "no_accident",
    "minor_accident",
    "minor_accident",
    "moderate_accident",
    "moderate_accident",
    "severe_accident",
    "totaled_vehicle",
    "pothole",
    "pothole",
    "pothole",
]

STATUSES = [
    IncidentStatus.detected,
    IncidentStatus.detected,
    IncidentStatus.notified,
    IncidentStatus.acknowledged,
    IncidentStatus.resolved,
    IncidentStatus.closed,
]

STATUS_WEIGHTS = [0.30, 0.25, 0.20, 0.12, 0.08, 0.05]

SAMPLE_DETECTIONS = {
    "minor_accident": [
        {"class": "minor_accident", "confidence": 0.78, "bbox": [120, 80, 380, 290]},
    ],
    "moderate_accident": [
        {"class": "moderate_accident", "confidence": 0.81, "bbox": [90, 60, 420, 310]},
        {"class": "minor_accident",    "confidence": 0.55, "bbox": [200, 100, 340, 240]},
    ],
    "severe_accident": [
        {"class": "severe_accident",   "confidence": 0.91, "bbox": [60, 40, 500, 380]},
        {"class": "moderate_accident", "confidence": 0.62, "bbox": [150, 90, 350, 270]},
    ],
    "totaled_vehicle": [
        {"class": "totaled_vehicle",   "confidence": 0.95, "bbox": [40, 30, 560, 420]},
        {"class": "severe_accident",   "confidence": 0.70, "bbox": [100, 70, 440, 330]},
    ],
    "pothole": [
        {"class": "pothole", "confidence": 0.88, "bbox": [180, 200, 340, 310]},
    ],
    "no_accident": [],
}

SAMPLE_GUIDANCE = {
    "minor_accident":    {"severity": "Minor",    "emergency_level": 1, "llm_enhanced": "Demo: Minor collision detected. Ensure hazard lights are on, check for injuries, and exchange information."},
    "moderate_accident": {"severity": "Moderate", "emergency_level": 2, "llm_enhanced": "Demo: Moderate accident detected. Call emergency services immediately and do not move injured persons."},
    "severe_accident":   {"severity": "Severe",   "emergency_level": 3, "llm_enhanced": "Demo: Severe accident. Call emergency services now. Start CPR if victim is unresponsive and not breathing."},
    "totaled_vehicle":   {"severity": "Critical", "emergency_level": 4, "llm_enhanced": "Demo: Critical multi-casualty incident. Call emergency services. Check for fire/fuel leaks before approaching."},
    "pothole":           {"severity": "Infrastructure Hazard", "emergency_level": 0, "llm_enhanced": "Demo: Significant pothole detected. Report to municipal road department for urgent repair."},
    "no_accident":       {"severity": "None", "emergency_level": 0},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def random_datetime_last_30_days():
    delta = timedelta(days=random.uniform(0, 30))
    return datetime.utcnow() - delta

def weighted_choice(choices, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    cumulative = 0
    for choice, weight in zip(choices, weights):
        cumulative += weight
        if r <= cumulative:
            return choice
    return choices[-1]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("CityPulse Demo Seed")
    print("=" * 40)

    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    db: Session = next(get_db())

    # Create or fetch demo user
    user = db.query(User).filter(User.email == DEMO_EMAIL).first()
    if not user:
        user = User(
            email=DEMO_EMAIL,
            hashed_password=get_password_hash(DEMO_PASSWORD),
            role=UserRole.officer,
            district="Demo District",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Created demo user: {DEMO_EMAIL} / {DEMO_PASSWORD}")
    else:
        print(f"Using existing user: {DEMO_EMAIL}")

    # Seed incidents
    created = 0
    for i in range(N_INCIDENTS):
        lat, lng, address = random.choice(LOCATIONS)
        # Add small jitter so pins don't stack
        lat += random.uniform(-0.008, 0.008)
        lng += random.uniform(-0.008, 0.008)

        severity = random.choice(SEVERITIES)
        status   = weighted_choice(STATUSES, STATUS_WEIGHTS)
        ts       = random_datetime_last_30_days()

        incident = Incident(
            image_url=None,
            severity=severity,
            status=status,
            lat=round(lat, 6),
            lng=round(lng, 6),
            address_text=address,
            detection_results=SAMPLE_DETECTIONS.get(severity, []),
            guidance=SAMPLE_GUIDANCE.get(severity, {}),
            reported_by=user.id,
            created_at=ts,
        )
        db.add(incident)
        created += 1

    db.commit()
    print(f"Seeded {created} demo incidents.")
    print(f"\nSign in at http://localhost:8000 with:")
    print(f"  Email:    {DEMO_EMAIL}")
    print(f"  Password: {DEMO_PASSWORD}")
    print("\nOpen the Dashboard tab to see charts and the incident map.")

if __name__ == "__main__":
    main()
