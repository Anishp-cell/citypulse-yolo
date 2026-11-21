from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from backend.domain.status import HazardStatus
from backend.models import Hazard
from tests.helpers import PUNE, detect_nothing, detect_pothole, offset, report

BBOX = f"{PUNE[1] - 0.01},{PUNE[0] - 0.01},{PUNE[1] + 0.01},{PUNE[0] + 0.01}"


def anon(app):
    return TestClient(app, headers={"X-Requested-With": "citypulse"})


def set_status(app, hid, status, **fields):
    with app.state.SessionLocal() as db:
        h = db.get(Hazard, hid)
        h.status = status
        for k, v in fields.items():
            setattr(h, k, v)
        db.commit()


def test_bbox_filters(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    inside = report(c)
    report(c, *offset(*PUNE, north_m=5000))
    with anon(app) as a:
        r = a.get("/api/hazards", params={"bbox": BBOX})
    assert r.status_code == 200
    assert [h["id"] for h in r.json()["hazards"]] == [inside["hazard_id"]]


def test_bad_bbox_422(app):
    with anon(app) as a:
        assert a.get("/api/hazards", params={"bbox": "1,2,3"}).status_code == 422
        assert a.get("/api/hazards", params={"bbox": "a,b,c,d"}).status_code == 422


def test_status_filter(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    active = report(c)["hazard_id"]
    fixed = report(c, *offset(*PUNE, north_m=100))["hazard_id"]
    set_status(app, fixed, HazardStatus.fixed)
    with anon(app) as a:
        ids = lambda s: {h["id"] for h in a.get("/api/hazards", params={"bbox": BBOX, "status": s}).json()["hazards"]}  # noqa: E731
        assert ids("active") == {active}
        assert ids("fixed") == {fixed}
        assert ids("all") == {active, fixed}


def test_pending_and_rejected_hidden(app, citizen):
    c, _ = citizen
    detect_nothing(app)
    pending = report(c)["hazard_id"]
    detect_pothole(app)
    rejected = report(c, *offset(*PUNE, north_m=100))["hazard_id"]
    set_status(app, rejected, HazardStatus.rejected)
    with anon(app) as a:
        assert a.get("/api/hazards", params={"bbox": BBOX, "status": "all"}).json()["hazards"] == []
        assert a.get(f"/api/hazards/{pending}").status_code == 404
        assert a.get(f"/api/hazards/{rejected}").status_code == 404


def test_reporter_and_staff_can_see_own_pending_hazard(app, citizen, officer):
    c, _ = citizen
    o, _ = officer
    detect_nothing(app)
    pending = report(c)["hazard_id"]
    assert c.get(f"/api/hazards/{pending}").status_code == 200
    assert o.get(f"/api/hazards/{pending}").status_code == 200


def test_hazard_detail_shape(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    hid = report(c, note="deep")["hazard_id"]
    with anon(app) as a:
        d = a.get(f"/api/hazards/{hid}").json()
    assert d["id"] == hid and d["type"] == "pothole" and d["status"] == "open"
    assert d["road"] == "FC Road" and d["ward"] == "Shivajinagar" and d["report_count"] == 1
    assert d["days_open"] == 0 and d["photo_url"].startswith("/media/public/")
    assert [e["status"] for e in d["timeline"]] == ["open"]
    assert d["is_following"] is False and d["my_vote"] is None
    assert d["share_url"] == f"http://testserver/h/{hid}"


def test_public_payload_has_no_reporter_pii(app, make_user):
    c, user = make_user(email="secret.person@example.com")
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    with anon(app) as a:
        texts = a.get(f"/api/hazards/{hid}").text + a.get("/api/hazards", params={"bbox": BBOX}).text
    assert "secret.person" not in texts and "Test User" not in texts
    assert '"user_id"' not in texts and "full_photos" not in texts


def test_full_image_url_only_for_reporter_and_officer(app, make_user):
    reporter, _ = make_user()
    other, _ = make_user()
    officer, _ = make_user("officer")
    detect_pothole(app)
    hid = report(reporter)["hazard_id"]
    assert len(reporter.get(f"/api/hazards/{hid}").json()["full_photos"]) == 1
    assert len(officer.get(f"/api/hazards/{hid}").json()["full_photos"]) == 1
    assert "full_photos" not in other.get(f"/api/hazards/{hid}").json()
    url = officer.get(f"/api/hazards/{hid}").json()["full_photos"][0]
    assert officer.get(url).status_code == 200


def test_follow_flag_reflects_reporter(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    assert c.get(f"/api/hazards/{hid}").json()["is_following"] is True


def test_scorecard_wards_and_roads(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    now = datetime.now(UTC)
    report(c)
    report(c, *offset(*PUNE, north_m=100), severity="high")
    fixed = report(c, *offset(*PUNE, north_m=200))["hazard_id"]
    set_status(app, fixed, HazardStatus.fixed, first_reported_at=now - timedelta(days=50),
               fixed_at=now - timedelta(days=40))
    old_fixed = report(c, *offset(*PUNE, north_m=300))["hazard_id"]
    set_status(app, old_fixed, HazardStatus.fixed, first_reported_at=now - timedelta(days=60),
               fixed_at=now - timedelta(days=20))
    with anon(app) as cl:
        s = cl.get("/api/scorecard", params={"city": "Pune"}).json()
    ward = s["wards"][0]
    assert ward["ward"] == "Shivajinagar" and ward["active"] == 2
    assert ward["median_days_to_fix_90d"] == 25.0  # took 10 and 40 days
    assert ward["pct_fixed_30d"] == 50.0  # of the 2 hazards reported 30-120 days ago, 1 fixed within 30d
    road = s["worst_roads"][0]
    assert road["road"] == "FC Road" and road["active"] == 2 and road["health"] == 100 - 10 - 20


def test_meta_statuses(app):
    with anon(app) as a:
        statuses = a.get("/api/meta/statuses").json()
    by = {s["value"]: s for s in statuses}
    assert set(by) == {s.value for s in HazardStatus}
    assert by["open"] == {"value": "open", "active": True, "public": True}
    assert by["pending_review"]["public"] is False


def _write_index(settings):
    dist = Path(settings.frontend_dist)
    (dist / "assets").mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><head><title>CityPulse</title><!-- OG_TAGS --></head><body>app</body></html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")


def test_spa_fallback_and_assets(app, settings):
    _write_index(settings)
    with anon(app) as a:
        assert "app" in a.get("/map").text
        assert a.get("/assets/app.js").text == "console.log(1)"
        assert "immutable" in a.get("/assets/app.js").headers["cache-control"]
        assert a.get("/api/does-not-exist").status_code == 404
        leak = a.get("/assets/..%2F..%2F..%2Fbackend%2Fsettings.py")
        assert "SECRET" not in leak.text


def test_og_tags_injected_and_escaped(app, settings, citizen):
    _write_index(settings)
    c, _ = citizen
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    set_status(app, hid, HazardStatus.open, road='<script>alert(1)</script>')
    with anon(app) as a:
        html = a.get(f"/h/{hid}").text
    assert 'property="og:title"' in html and 'property="og:image" content="http://testserver/media/public/' in html
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_og_page_for_unknown_hazard_still_serves_app(app, settings):
    _write_index(settings)
    with anon(app) as a:
        r = a.get("/h/99999")
    assert r.status_code == 200 and "og:title" not in r.text
