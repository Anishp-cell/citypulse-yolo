from backend.models import Flag, Follow, Hazard, Report
from tests.helpers import PUNE, detect_nothing, detect_pothole, jpeg, offset, report


def status(client, hid, to, note=None, photo=False):
    files = {"photo": ("p.jpg", jpeg(), "image/jpeg")} if photo else None
    data = {"to": to}
    if note:
        data["note"] = note
    return client.post(f"/api/admin/hazards/{hid}/status", data=data, files=files)


def hazard(app, hid):
    with app.state.SessionLocal() as db:
        h = db.get(Hazard, hid)
        db.expunge(h)
        return h


def test_citizen_forbidden_admin_routes(citizen):
    c, _ = citizen
    for path in ("/api/admin/queue", "/api/admin/moderation", "/api/admin/stats", "/api/admin/users"):
        assert c.get(path).status_code == 403
    assert status(c, 1, "acknowledged").status_code == 403


def test_queue_sorted_by_priority_and_filtered(app, citizen, officer):
    c, _ = citizen
    o, _ = officer
    detect_pothole(app)
    low = report(c, severity="low")["hazard_id"]
    high = report(c, *offset(*PUNE, north_m=200), severity="high")["hazard_id"]
    q = o.get("/api/admin/queue").json()
    assert [h["id"] for h in q["hazards"]] == [high, low] and q["total"] == 2
    assert o.get("/api/admin/queue", params={"ward": "Nowhere"}).json()["hazards"] == []
    assert o.get("/api/admin/queue", params={"status": "filed"}).json()["hazards"] == []


def test_officer_workflow_with_notes(app, citizen, officer):
    c, _ = citizen
    o, _ = officer
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    assert status(o, hid, "acknowledged", note="Crew assigned").status_code == 200
    assert status(o, hid, "in_progress").status_code == 200
    timeline = c.get(f"/api/hazards/{hid}").json()["timeline"]
    assert [e["status"] for e in timeline] == ["open", "acknowledged", "in_progress"]
    assert timeline[1]["note"] == "Crew assigned"


def test_invalid_transition_409(app, citizen, officer):
    c, _ = citizen
    o, _ = officer
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    assert status(o, hid, "fixed").status_code == 409


def test_officer_fixed_requires_photo(app, citizen, officer):
    c, _ = citizen
    o, _ = officer
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    r = status(o, hid, "fixed_unverified")
    assert r.status_code == 422 and r.json()["detail"] == "photo_required"
    assert status(o, hid, "fixed_unverified", photo=True).status_code == 200
    h = hazard(app, hid)
    assert h.status.value == "fixed_unverified" and h.fixed_by_officer is True


def test_moderation_lists_pending_and_flagged(app, make_user, officer):
    c, _ = make_user()
    other, _ = make_user()
    o, _ = officer
    detect_nothing(app)
    pending = report(c)["hazard_id"]
    detect_pothole(app)
    flagged = report(c, *offset(*PUNE, north_m=300))["hazard_id"]
    other.post(f"/api/hazards/{flagged}/flag", json={"reason": "not_a_hazard"})
    m = o.get("/api/admin/moderation").json()
    assert [h["id"] for h in m["pending"]] == [pending]
    assert [h["id"] for h in m["flagged"]] == [flagged]
    assert m["flagged"][0]["flags"] == ["not_a_hazard"]


def test_approve_pending_and_dismiss_flags(app, make_user, officer):
    c, _ = make_user()
    other, _ = make_user()
    o, _ = officer
    detect_nothing(app)
    pending = report(c)["hazard_id"]
    assert o.post(f"/api/admin/hazards/{pending}/approve").json()["status"] == "open"
    other.post(f"/api/hazards/{pending}/flag", json={"reason": "duplicate"})
    o.post(f"/api/admin/hazards/{pending}/approve")
    with app.state.SessionLocal() as db:
        assert db.query(Flag).filter_by(resolved=False).count() == 0


def test_reject_rules(app, citizen, officer, admin):
    c, _ = citizen
    o, _ = officer
    a, _ = admin
    detect_nothing(app)
    pending = report(c)["hazard_id"]
    detect_pothole(app)
    public = report(c, *offset(*PUNE, north_m=300))["hazard_id"]
    assert o.post(f"/api/admin/hazards/{pending}/reject", json={"reason": "not_a_hazard"}).status_code == 200
    assert o.post(f"/api/admin/hazards/{public}/reject", json={"reason": "abuse"}).status_code == 409
    assert a.post(f"/api/admin/hazards/{public}/reject", json={"reason": "abuse"}).status_code == 200
    assert a.post(f"/api/admin/hazards/{public}/reject", json={"reason": "made up"}).status_code == 422


def test_merge_moves_reports_and_recounts(app, make_user, admin):
    a_c, _ = make_user()
    b_c, b_user = make_user()
    adm, _ = admin
    detect_pothole(app)
    target = report(a_c)["hazard_id"]
    dup = report(b_c, *offset(*PUNE, north_m=30))["hazard_id"]
    r = adm.post(f"/api/admin/hazards/{dup}/merge", json={"into_id": target})
    assert r.status_code == 200
    with app.state.SessionLocal() as db:
        assert db.query(Report).filter_by(hazard_id=target).count() == 2
        assert db.get(Hazard, target).report_count == 2
        assert db.get(Hazard, dup).status.value == "rejected"
        assert db.get(Follow, (b_user["id"], target)) is not None
    assert adm.post(f"/api/admin/hazards/{target}/merge", json={"into_id": target}).status_code == 422


def test_officer_cannot_merge(app, citizen, officer):
    c, _ = citizen
    o, _ = officer
    detect_pothole(app)
    a = report(c)["hazard_id"]
    b = report(c, *offset(*PUNE, north_m=30))["hazard_id"]
    assert o.post(f"/api/admin/hazards/{b}/merge", json={"into_id": a}).status_code == 403


def test_stats_counts(app, citizen, officer):
    c, _ = citizen
    o, _ = officer
    detect_pothole(app)
    report(c)
    report(c, *offset(*PUNE, north_m=300), hazard_type="debris")
    s = o.get("/api/admin/stats").json()
    assert s["by_status"]["open"] == 1 and s["by_status"]["pending_review"] == 1
    assert s["by_type"] == {"pothole": 1, "debris": 1}
    assert sum(d["count"] for d in s["daily_reports"]) == 2 and len(s["daily_reports"]) == 30
    assert len(s["heat"]) == 1 and set(s["heat"][0]) == {"lat", "lng", "weight"}
    assert s["totals"]["active"] == 1


def test_users_and_roles(app, make_user, officer, admin):
    _, target = make_user()
    o, _ = officer
    a, admin_user = admin
    assert o.patch(f"/api/admin/users/{target['id']}/role", json={"role": "officer"}).status_code == 403
    users = a.get("/api/admin/users", params={"q": target["email"]}).json()
    assert [u["id"] for u in users["users"]] == [target["id"]]
    assert a.patch(f"/api/admin/users/{target['id']}/role", json={"role": "officer"}).json()["role"] == "officer"
    assert a.patch(f"/api/admin/users/{target['id']}/role", json={"role": "king"}).status_code == 422
    assert a.patch(f"/api/admin/users/{admin_user['id']}/role", json={"role": "citizen"}).status_code == 409


def test_reject_for_abuse_removes_public_photos(app, citizen, admin):
    # Review I6: abusive photos must disappear from public storage, not just from listings.
    c, _ = citizen
    a, _ = admin
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    with app.state.SessionLocal() as db:
        rep = db.query(Report).filter_by(hazard_id=hid).one()
        public_key, thumb_key = rep.public_image_key, rep.thumb_key
    assert a.post(f"/api/admin/hazards/{hid}/reject", json={"reason": "abuse"}).status_code == 200
    assert app.state.storage.get(public_key, public=True) is None
    assert app.state.storage.get(thumb_key, public=True) is None
    d = a.get(f"/api/hazards/{hid}").json()
    assert d["photo_url"] is None and d["photos"] == [] and d["thumb_url"] is None


def test_remove_photos_keeps_hazard_public(app, citizen, officer):
    c, _ = citizen
    o, _ = officer
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    assert o.post(f"/api/admin/hazards/{hid}/remove-photos").status_code == 200
    d = c.get(f"/api/hazards/{hid}").json()
    assert d["status"] == "open" and d["photo_url"] is None


def test_long_address_truncated(app, citizen):
    # Review I8: Hazard.address is String(300); Nominatim display names can be longer.
    import httpx

    from backend.services.geocode import Geocoder
    from tests.conftest import FAKE_PLACE

    long_place = {**FAKE_PLACE, "display_name": "A" * 450}
    app.state.geocoder = Geocoder(app.state.settings, min_interval_s=0,
                                  transport=httpx.MockTransport(lambda r: httpx.Response(200, json=long_place)))
    c, _ = citizen
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    with app.state.SessionLocal() as db:
        assert len(db.get(Hazard, hid).address) <= 300
