import uuid

from fastapi.testclient import TestClient

from backend.models import Follow, Hazard, Report, StatusEvent, Upload
from tests.helpers import PUNE, analyze, detect_nothing, detect_pothole, offset, report, submit

# ---------- analyze ----------

def test_analyze_returns_ai_and_place(app, citizen):
    c, _ = citizen
    detect_pothole(app, 0.87)
    r = analyze(c)
    assert r.status_code == 200, r.text
    body = r.json()
    assert uuid.UUID(body["upload_id"])
    assert body["ai"]["is_pothole"] is True
    assert body["ai"]["confidence"] == 0.87
    assert body["ai"]["severity_suggestion"] in {"low", "medium", "high"}
    assert body["ai"]["boxes"] == [[300, 200, 500, 400]]
    assert body["place"]["road"] == "FC Road" and body["place"]["ward"] == "Shivajinagar"
    assert body["nearby"] == []
    assert body["preview_url"].startswith("/media/public/")


def test_analyze_without_pothole(app, citizen):
    c, _ = citizen
    detect_nothing(app)
    ai = analyze(c).json()["ai"]
    assert ai == {"is_pothole": False, "confidence": 0.0, "severity_suggestion": "medium", "boxes": []}


def test_analyze_lists_nearby_within_30m(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    near = report(c, *offset(*PUNE, north_m=10))
    report(c, *offset(*PUNE, north_m=100))
    nearby = analyze(c).json()["nearby"]
    assert [n["id"] for n in nearby] == [near["hazard_id"]]
    assert 9 <= nearby[0]["distance_m"] <= 11
    assert nearby[0]["report_count"] == 1 and nearby[0]["type"] == "pothole"


def test_analyze_requires_auth(app):
    with TestClient(app, headers={"X-Requested-With": "citypulse"}) as anon:
        assert analyze(anon).status_code == 401


def test_analyze_rejects_non_image_415(citizen):
    c, _ = citizen
    r = analyze(c, data=b"not an image at all")
    assert r.status_code == 415 and r.json()["detail"] == "unsupported_image"


def test_lat_lng_out_of_range_422(citizen):
    c, _ = citizen
    assert analyze(c, lat=95).status_code == 422
    assert analyze(c, lng=-200).status_code == 422


def test_zero_zero_coordinates_rejected(citizen):
    c, _ = citizen
    assert analyze(c, lat=0, lng=0).status_code == 422


def test_analyze_rate_limited(app, citizen):
    c, _ = citizen
    app.state.settings.rate_limits_enabled = True
    codes = [analyze(c).status_code for _ in range(21)]
    assert codes[:20] == [200] * 20 and codes[20] == 429


# ---------- submit ----------

def _hazard(app, hid):
    with app.state.SessionLocal() as db:
        h = db.get(Hazard, hid)
        db.expunge(h)
        return h


def test_creates_hazard_open_when_ai_confident(app, citizen):
    c, user = citizen
    detect_pothole(app, 0.6)
    r = report(c)
    assert r["created_new"] is True and r["status"] == "open"
    h = _hazard(app, r["hazard_id"])
    assert (h.type, h.ai_verified, h.road, h.ward, h.city, h.report_count) == (
        "pothole", True, "FC Road", "Shivajinagar", "Pune", 1)
    assert h.priority_score > 0 and h.cover_image_key
    with app.state.SessionLocal() as db:
        assert db.query(Follow).filter_by(user_id=user["id"], hazard_id=h.id).count() == 1
        ev = db.query(StatusEvent).filter_by(hazard_id=h.id).one()
        assert ev.from_status is None and ev.to_status == "open"


def test_low_confidence_goes_to_pending_review(app, citizen):
    c, _ = citizen
    detect_nothing(app)
    r = report(c)
    assert r["created_new"] is True and r["status"] == "pending_review"
    assert _hazard(app, r["hazard_id"]).ai_verified is False


def test_manual_type_goes_to_pending_review(app, citizen):
    c, _ = citizen
    detect_pothole(app, 0.9)
    r = report(c, hazard_type="open_manhole")
    assert r["status"] == "pending_review"


def test_auto_merge_within_20m_same_type(app, make_user):
    a, _ = make_user()
    b, _ = make_user()
    detect_pothole(app)
    first = report(a)
    second = report(b, *offset(*PUNE, east_m=15))
    assert second["created_new"] is False and second["hazard_id"] == first["hazard_id"]
    assert _hazard(app, first["hazard_id"]).report_count == 2


def test_no_merge_beyond_20m(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    first = report(c)
    assert report(c, *offset(*PUNE, east_m=25))["hazard_id"] != first["hazard_id"]


def test_no_merge_different_type(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    first = report(c)
    assert report(c, hazard_type="debris")["hazard_id"] != first["hazard_id"]


def test_no_merge_into_fixed_hazard(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    first = report(c)
    with app.state.SessionLocal() as db:
        db.get(Hazard, first["hazard_id"]).status = "fixed"
        db.commit()
    assert report(c)["created_new"] is True


def test_same_user_plus_one_counted_once(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    first = report(c)
    again = report(c)
    assert again["hazard_id"] == first["hazard_id"]
    assert _hazard(app, first["hazard_id"]).report_count == 1


def test_report_idempotent_retry(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    up = analyze(c).json()["upload_id"]
    r1 = submit(c, up, key="retry-key-1")
    r2 = submit(c, up, key="retry-key-1")
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    with app.state.SessionLocal() as db:
        assert db.query(Report).count() == 1


def test_upload_single_use(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    up = analyze(c).json()["upload_id"]
    assert submit(c, up).status_code == 200
    assert submit(c, up).status_code == 409


def test_cannot_use_someone_elses_upload(app, make_user):
    a, _ = make_user()
    b, _ = make_user()
    detect_pothole(app)
    up = analyze(a).json()["upload_id"]
    assert submit(b, up).status_code == 404


def test_attach_to_existing_hazard(app, make_user):
    a, _ = make_user()
    b, _ = make_user()
    detect_pothole(app)
    first = report(a)
    up = analyze(b, *offset(*PUNE, north_m=40)).json()["upload_id"]
    r = submit(b, up, attach_to_hazard_id=first["hazard_id"])
    assert r.status_code == 200 and r.json()["hazard_id"] == first["hazard_id"]


def test_attach_too_far_422(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    first = report(c)
    up = analyze(c, *offset(*PUNE, north_m=80)).json()["upload_id"]
    r = submit(c, up, attach_to_hazard_id=first["hazard_id"])
    assert r.status_code == 422 and r.json()["detail"] == "too_far"


def test_invalid_hazard_type_422(app, citizen):
    c, _ = citizen
    up = analyze(c).json()["upload_id"]
    assert submit(c, up, hazard_type="alien").status_code == 422


def test_submit_marks_upload_consumed_and_copies_images(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    up = analyze(c).json()["upload_id"]
    submit(c, up, note="near the bus stop")
    with app.state.SessionLocal() as db:
        u = db.get(Upload, up)
        rep = db.query(Report).one()
        assert u.consumed_at is not None
        assert (rep.image_key, rep.public_image_key, rep.thumb_key) == (u.image_key, u.public_image_key, u.thumb_key)
        assert rep.note == "near the bus stop" and rep.ai_label == "pothole"
