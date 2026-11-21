import io
import uuid

from PIL import Image

from backend.ml.detector import Detection, FakeDetector

PUNE = (18.5236, 73.8410)
M_PER_DEG_LAT = 111_320.0


def offset(lat, lng, north_m=0.0, east_m=0.0):
    import math

    return lat + north_m / M_PER_DEG_LAT, lng + east_m / (M_PER_DEG_LAT * math.cos(math.radians(lat)))


def jpeg(size=(800, 600), color=(90, 90, 90)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def detect_pothole(app, confidence=0.87, box=(300, 200, 500, 400)):
    app.state.detector = FakeDetector([Detection("pothole", confidence, box)])


def detect_nothing(app):
    app.state.detector = FakeDetector([])


def analyze(client, lat=PUNE[0], lng=PUNE[1], accuracy_m=8.0, source="gps", data=None):
    return client.post(
        "/api/reports/analyze",
        files={"file": ("photo.jpg", data if data is not None else jpeg(), "image/jpeg")},
        data={"lat": str(lat), "lng": str(lng), "accuracy_m": str(accuracy_m), "location_source": source},
    )


def submit(client, upload_id, hazard_type="pothole", severity="medium", key=None, **extra):
    return client.post(
        "/api/reports",
        json={"upload_id": upload_id, "hazard_type": hazard_type, "severity": severity, **extra},
        headers={"Idempotency-Key": key or str(uuid.uuid4())},
    )


def report(client, lat=PUNE[0], lng=PUNE[1], hazard_type="pothole", **extra):
    """Analyze + submit in one go; returns the submit JSON."""
    a = analyze(client, lat, lng)
    assert a.status_code == 200, a.text
    r = submit(client, a.json()["upload_id"], hazard_type=hazard_type, **extra)
    assert r.status_code == 200, r.text
    return r.json()
