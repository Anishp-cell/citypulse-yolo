from backend.models import Follow, Hazard, Verification
from tests.helpers import PUNE, detect_pothole, jpeg, offset, report


def vote(client, hid, fixed, photo=False):
    files = {"photo": ("p.jpg", jpeg(), "image/jpeg")} if photo else None
    return client.post(f"/api/hazards/{hid}/verify", data={"is_fixed": "true" if fixed else "false"}, files=files)


def hazard(app, hid):
    with app.state.SessionLocal() as db:
        h = db.get(Hazard, hid)
        db.expunge(h)
        return h


def new_hazard(app, client):
    detect_pothole(app)
    return report(client)["hazard_id"]


def test_follow_and_unfollow(app, make_user):
    reporter, _ = make_user()
    other, user = make_user()
    hid = new_hazard(app, reporter)
    assert other.post(f"/api/hazards/{hid}/follow").status_code == 200
    assert other.get(f"/api/hazards/{hid}").json()["is_following"] is True
    assert other.delete(f"/api/hazards/{hid}/follow").status_code == 200
    with app.state.SessionLocal() as db:
        assert db.get(Follow, (user["id"], hid)) is None


def test_follow_requires_login(app, client):
    assert client.post("/api/hazards/1/follow").status_code == 401


def test_filed_sets_ticket_and_status(app, citizen):
    c, _ = citizen
    hid = new_hazard(app, c)
    r = c.post(f"/api/hazards/{hid}/filed", json={"portal": "PMC CARE", "ticket_id": "PMC-12345"})
    assert r.status_code == 200 and r.json()["status"] == "filed"
    h = hazard(app, hid)
    assert (h.status.value, h.official_ticket_id, h.official_portal) == ("filed", "PMC-12345", "PMC CARE")
    assert c.get(f"/api/hazards/{hid}").json()["timeline"][-1]["status"] == "filed"


def test_filed_again_updates_ticket_without_new_event(app, citizen):
    c, _ = citizen
    hid = new_hazard(app, c)
    c.post(f"/api/hazards/{hid}/filed", json={"portal": "PMC CARE", "ticket_id": "A1"})
    r = c.post(f"/api/hazards/{hid}/filed", json={"portal": "PMC CARE", "ticket_id": "B2"})
    assert r.status_code == 200 and hazard(app, hid).official_ticket_id == "B2"
    assert len(c.get(f"/api/hazards/{hid}").json()["timeline"]) == 2


def test_filed_rejects_blank_ticket(app, citizen):
    c, _ = citizen
    hid = new_hazard(app, c)
    assert c.post(f"/api/hazards/{hid}/filed", json={"portal": "PMC", "ticket_id": "  "}).status_code == 422


def test_fixed_vote_moves_to_fixed_unverified(app, citizen):
    c, _ = citizen
    hid = new_hazard(app, c)
    r = vote(c, hid, True, photo=True)
    assert r.status_code == 200 and r.json()["status"] == "fixed_unverified"
    d = c.get(f"/api/hazards/{hid}").json()
    assert d["my_vote"] == "fixed" and d["verify_round"] == 1


def test_not_fixed_vote_on_active_hazard_409(app, citizen):
    c, _ = citizen
    hid = new_hazard(app, c)
    assert vote(c, hid, False).status_code == 409


def test_two_distinct_fixed_votes_confirm(app, make_user):
    a, _ = make_user()
    b, _ = make_user()
    hid = new_hazard(app, a)
    vote(a, hid, True)
    r = vote(b, hid, True)
    assert r.json()["status"] == "fixed"
    assert hazard(app, hid).fixed_at is not None


def test_same_user_cannot_vote_twice_per_round(app, citizen):
    c, _ = citizen
    hid = new_hazard(app, c)
    vote(c, hid, True)
    assert vote(c, hid, True).status_code == 409
    assert hazard(app, hid).status.value == "fixed_unverified"


def test_two_not_fixed_votes_reopen_and_boost_priority(app, make_user):
    a, _ = make_user()
    b, _ = make_user()
    d, _ = make_user()
    hid = new_hazard(app, a)
    before = hazard(app, hid).priority_score
    vote(a, hid, True)
    vote(b, hid, False)
    r = vote(d, hid, False)
    assert r.json()["status"] == "open"
    h = hazard(app, hid)
    assert h.dispute_count == 1 and h.priority_score > before


def test_new_round_after_reopen_allows_voting_again(app, make_user):
    a, _ = make_user()
    b, _ = make_user()
    d, _ = make_user()
    hid = new_hazard(app, a)
    vote(a, hid, True)
    vote(b, hid, False)
    vote(d, hid, False)
    assert vote(a, hid, True).status_code == 200  # round 2
    with app.state.SessionLocal() as db:
        assert db.query(Verification).filter_by(hazard_id=hid, round=2).count() == 1


def test_officer_proof_plus_one_citizen_vote_confirms(app, make_user):
    a, _ = make_user()
    o, _ = make_user("officer")
    hid = new_hazard(app, a)
    r = o.post(f"/api/admin/hazards/{hid}/status", data={"to": "fixed_unverified", "note": "patched"},
               files={"photo": ("p.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200
    assert vote(a, hid, True).json()["status"] == "fixed"


def test_flag_once_per_user(app, make_user):
    a, _ = make_user()
    b, _ = make_user()
    hid = new_hazard(app, a)
    assert b.post(f"/api/hazards/{hid}/flag", json={"reason": "not_a_hazard"}).status_code == 200
    assert b.post(f"/api/hazards/{hid}/flag", json={"reason": "duplicate"}).status_code == 409
    assert b.post(f"/api/hazards/{hid}/flag", json={"reason": "made-up"}).status_code in (409, 422)


def test_flag_rate_limited(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    ids = [report(c, *offset(*PUNE, north_m=100 * i))["hazard_id"] for i in range(11)]
    app.state.settings.rate_limits_enabled = True
    codes = [c.post(f"/api/hazards/{h}/flag", json={"reason": "other"}).status_code for h in ids]
    assert codes[:10] == [200] * 10 and codes[10] == 429


def test_my_reports_following_and_impact(app, make_user):
    a, _ = make_user()
    b, _ = make_user()
    mine = new_hazard(app, a)
    theirs = report(b, *offset(*PUNE, north_m=500))["hazard_id"]
    a.post(f"/api/hazards/{theirs}/follow")
    assert [h["id"] for h in a.get("/api/me/reports").json()["hazards"]] == [mine]
    assert {h["id"] for h in a.get("/api/me/following").json()["hazards"]} == {mine, theirs}
    vote(a, theirs, True)
    vote(b, theirs, True)  # confirmed fixed
    impact = a.get("/api/me/impact").json()
    assert impact == {"reports": 1, "hazards_reported": 1, "hazards_fixed": 0, "verifications": 1}
    assert b.get("/api/me/impact").json()["hazards_fixed"] == 1


def test_officer_proof_disputed_by_citizen_reopens(app, make_user):
    # Review I5: an officer "fix" that the only follower disputes must not hide the hazard as fixed.
    a, _ = make_user()
    o, _ = make_user("officer")
    hid = new_hazard(app, a)
    o.post(f"/api/admin/hazards/{hid}/status", data={"to": "fixed_unverified"},
           files={"photo": ("p.jpg", jpeg(), "image/jpeg")})
    r = vote(a, hid, False)
    assert r.status_code == 200 and r.json()["status"] == "open"


def test_long_flag_details_stored_within_column_limit(app, make_user):
    # Review I8: Flag.reason is String(300); Postgres rejects longer values with a 500.
    a, _ = make_user()
    b, _ = make_user()
    hid = new_hazard(app, a)
    r = b.post(f"/api/hazards/{hid}/flag", json={"reason": "inappropriate_photo", "details": "x" * 300})
    assert r.status_code == 200
    from backend.models import Flag

    with app.state.SessionLocal() as db:
        assert len(db.query(Flag).one().reason) <= 300
