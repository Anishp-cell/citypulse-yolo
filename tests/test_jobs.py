from datetime import datetime, timedelta, timezone

import pytest

from backend.domain.status import HazardStatus
from backend.models import Hazard, StatusEvent, Upload, Verification
from backend.services.jobs import run_daily
from backend.services.push import PushService
from tests.helpers import analyze, detect_pothole, report
from tests.test_push import FakeSender, subscribe

NOW = datetime.now(timezone.utc)


@pytest.fixture
def sender(app, settings):
    settings.vapid_public_key, settings.vapid_private_key = "pub", "priv"
    fake = FakeSender()
    app.state.push = PushService(settings, app.state.SessionLocal, sender=fake)
    return fake


def run(app):
    with app.state.SessionLocal() as db:
        return run_daily(db, app.state.push, app.state.storage, NOW)


def age(app, hid, days, status=None, event_to=None):
    """Make a hazard look `days` old; optionally backdate its last event into `event_to`."""
    with app.state.SessionLocal() as db:
        h = db.get(Hazard, hid)
        h.first_reported_at = NOW - timedelta(days=days)
        if status:
            h.status = status
        if event_to:
            db.add(StatusEvent(hazard_id=hid, from_status=HazardStatus.open, to_status=event_to,
                               created_at=NOW - timedelta(days=days)))
        db.commit()


def test_daily_endpoint_requires_token(app, client, settings):
    settings.job_token = "s3cret-token"
    assert client.post("/api/internal/jobs/daily").status_code == 401
    assert client.post("/api/internal/jobs/daily", headers={"X-Job-Token": "wrong"}).status_code == 401
    r = client.post("/api/internal/jobs/daily", headers={"X-Job-Token": "s3cret-token"})
    assert r.status_code == 200 and "priorities_updated" in r.json()


def test_daily_endpoint_disabled_without_configured_token(client):
    assert client.post("/api/internal/jobs/daily", headers={"X-Job-Token": ""}).status_code == 401


def test_file_reminder_sent_once(app, citizen, sender):
    c, _ = citizen
    subscribe(c, "https://push.example/me")
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    age(app, hid, 8)
    assert run(app)["reminders_file"] == 1
    assert run(app)["reminders_file"] == 0
    assert [p["title"] for _, p in sender.sent] == ["Send it to the city"]
    assert [e["status"] for e in c.get(f"/api/hazards/{hid}").json()["timeline"]] == ["open"]  # reminders hidden


def test_no_file_reminder_for_new_hazard(app, citizen, sender):
    c, _ = citizen
    detect_pothole(app)
    report(c)
    assert run(app)["reminders_file"] == 0


def test_escalation_reminder_after_21_days_filed(app, citizen, sender):
    c, _ = citizen
    subscribe(c, "https://push.example/me")
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    age(app, hid, 22, status=HazardStatus.filed, event_to=HazardStatus.filed)
    assert run(app)["reminders_escalate"] == 1
    assert run(app)["reminders_escalate"] == 0


def test_auto_confirm_after_14d(app, citizen, sender):
    c, _ = citizen
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    age(app, hid, 15, status=HazardStatus.fixed_unverified, event_to=HazardStatus.fixed_unverified)
    assert run(app)["auto_confirmed"] == 1
    with app.state.SessionLocal() as db:
        h = db.get(Hazard, hid)
        assert h.status == HazardStatus.fixed and h.fixed_at is not None


def test_no_auto_confirm_when_disputed(app, citizen, sender):
    c, user = citizen
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    age(app, hid, 15, status=HazardStatus.fixed_unverified, event_to=HazardStatus.fixed_unverified)
    with app.state.SessionLocal() as db:
        h = db.get(Hazard, hid)
        db.add(Verification(hazard_id=hid, user_id=user["id"], round=h.verify_round, is_fixed=False))
        db.commit()
    assert run(app)["auto_confirmed"] == 0


def test_purges_stale_uploads(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    stale = analyze(c).json()["upload_id"]
    fresh = analyze(c).json()["upload_id"]
    with app.state.SessionLocal() as db:
        u = db.get(Upload, stale)
        u.created_at = NOW - timedelta(hours=25)
        stale_key = u.public_image_key
        db.commit()
    assert run(app)["uploads_purged"] == 1
    with app.state.SessionLocal() as db:
        assert db.get(Upload, stale) is None and db.get(Upload, fresh) is not None
    assert app.state.storage.get(stale_key, public=True) is None


def test_consumed_upload_rows_purged_but_report_images_kept(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    a = analyze(c).json()
    from tests.helpers import submit

    submit(c, a["upload_id"])
    with app.state.SessionLocal() as db:
        u = db.get(Upload, a["upload_id"])
        u.created_at = NOW - timedelta(hours=25)
        key = u.public_image_key
        db.commit()
    run(app)
    with app.state.SessionLocal() as db:
        assert db.get(Upload, a["upload_id"]) is None
    assert app.state.storage.get(key, public=True) is not None


def test_recomputes_priority_with_age(app, citizen):
    c, _ = citizen
    detect_pothole(app)
    hid = report(c)["hazard_id"]
    with app.state.SessionLocal() as db:
        before = db.get(Hazard, hid).priority_score
    age(app, hid, 60)
    run(app)
    with app.state.SessionLocal() as db:
        assert db.get(Hazard, hid).priority_score > before
