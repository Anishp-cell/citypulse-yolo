import json
import re

import pytest
from pywebpush import WebPushException

from backend.models import PushSubscription
from backend.services.push import PushService
from tests.helpers import detect_pothole, jpeg, report


class FakeSender:
    def __init__(self):
        self.sent = []
        self.fail_with = {}

    def __call__(self, subscription_info, data, vapid_private_key, vapid_claims, ttl):
        endpoint = subscription_info["endpoint"]
        if endpoint in self.fail_with:
            resp = type("R", (), {"status_code": self.fail_with[endpoint], "text": "gone"})()
            raise WebPushException("push failed", response=resp)
        self.sent.append((endpoint, json.loads(data)))


@pytest.fixture
def sender(app, settings):
    settings.vapid_public_key, settings.vapid_private_key = "pub-key", "priv-key"
    fake = FakeSender()
    app.state.push = PushService(settings, app.state.SessionLocal, sender=fake)
    return fake


def subscribe(client, endpoint):
    r = client.post("/api/push/subscribe", json={"endpoint": endpoint, "keys": {"p256dh": "p" * 20, "auth": "a" * 10}})
    assert r.status_code == 200, r.text


def test_vapid_key_endpoint(app, client, sender):
    assert client.get("/api/push/vapid-public-key").json() == {"enabled": True, "key": "pub-key"}


def test_push_disabled_without_keys(client):
    assert client.get("/api/push/vapid-public-key").json() == {"enabled": False, "key": None}


def test_subscribe_upserts_and_unsubscribe(app, make_user, sender):
    a, _ = make_user()
    b, b_user = make_user()
    subscribe(a, "https://push.example/1")
    subscribe(b, "https://push.example/1")  # same browser, new account
    with app.state.SessionLocal() as db:
        subs = db.query(PushSubscription).all()
        assert len(subs) == 1 and subs[0].user_id == b_user["id"]
    assert b.request("DELETE", "/api/push/subscribe", json={"endpoint": "https://push.example/1"}).status_code == 200
    with app.state.SessionLocal() as db:
        assert db.query(PushSubscription).count() == 0


def test_subscribe_rejects_non_https(citizen, sender):
    c, _ = citizen
    r = c.post("/api/push/subscribe", json={"endpoint": "http://evil.example", "keys": {"p256dh": "x", "auth": "y"}})
    assert r.status_code == 422


def test_status_change_pushes_followers_not_actor(app, make_user, sender):
    reporter, _ = make_user()
    follower, _ = make_user()
    officer, _ = make_user("officer")
    subscribe(reporter, "https://push.example/reporter")
    subscribe(follower, "https://push.example/follower")
    subscribe(officer, "https://push.example/officer")
    detect_pothole(app)
    hid = report(reporter)["hazard_id"]
    follower.post(f"/api/hazards/{hid}/follow")
    officer.post(f"/api/hazards/{hid}/follow")
    officer.post(f"/api/admin/hazards/{hid}/status", data={"to": "acknowledged"})
    endpoints = sorted(e for e, _ in sender.sent)
    assert endpoints == ["https://push.example/follower", "https://push.example/reporter"]
    payload = sender.sent[0][1]
    assert payload["url"] == f"/h/{hid}" and payload["title"] and "FC Road" in payload["body"]


def test_fixed_unverified_asks_followers_to_confirm(app, make_user, sender):
    reporter, _ = make_user()
    neighbour, _ = make_user()
    subscribe(reporter, "https://push.example/reporter")
    subscribe(neighbour, "https://push.example/neighbour")
    detect_pothole(app)
    hid = report(reporter)["hazard_id"]
    neighbour.post(f"/api/hazards/{hid}/follow")
    neighbour.post(f"/api/hazards/{hid}/verify", data={"is_fixed": "true"},
                   files={"photo": ("p.jpg", jpeg(), "image/jpeg")})
    assert [e for e, _ in sender.sent] == ["https://push.example/reporter"]
    assert "confirm" in sender.sent[0][1]["title"].lower()
    assert sender.sent[0][1]["tag"] == f"hazard-{hid}"


def test_push_payload_localized(app, make_user, sender):
    reporter, _ = make_user()
    reporter.patch("/api/auth/me", json={"language": "hi"})
    officer, _ = make_user("officer")
    subscribe(reporter, "https://push.example/hi")
    detect_pothole(app)
    hid = report(reporter)["hazard_id"]
    officer.post(f"/api/admin/hazards/{hid}/status", data={"to": "acknowledged"})
    assert re.search(r"[ऀ-ॿ]", sender.sent[0][1]["title"])


def test_gone_subscription_deleted(app, make_user, sender):
    reporter, _ = make_user()
    officer, _ = make_user("officer")
    subscribe(reporter, "https://push.example/gone")
    sender.fail_with["https://push.example/gone"] = 410
    detect_pothole(app)
    hid = report(reporter)["hazard_id"]
    officer.post(f"/api/admin/hazards/{hid}/status", data={"to": "acknowledged"})
    with app.state.SessionLocal() as db:
        assert db.query(PushSubscription).count() == 0


def test_transient_push_error_keeps_subscription(app, make_user, sender):
    reporter, _ = make_user()
    officer, _ = make_user("officer")
    subscribe(reporter, "https://push.example/flaky")
    sender.fail_with["https://push.example/flaky"] = 500
    detect_pothole(app)
    hid = report(reporter)["hazard_id"]
    r = officer.post(f"/api/admin/hazards/{hid}/status", data={"to": "acknowledged"})
    assert r.status_code == 200
    with app.state.SessionLocal() as db:
        assert db.query(PushSubscription).count() == 1
