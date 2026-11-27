from datetime import UTC, datetime, timedelta

import jwt

from backend.security import ALGORITHM, SESSION_COOKIE


def _second_device(app, email, password="correct-horse-1"):
    from fastapi.testclient import TestClient

    c = TestClient(app, headers={"X-Requested-With": "citypulse"})
    assert c.post("/api/auth/login", json={"email": email, "password": password}).status_code == 200
    return c


def test_logout_all_ends_sessions_on_every_device(app, make_user):
    phone, user = make_user()
    laptop = _second_device(app, user["email"])
    assert laptop.get("/api/auth/me").status_code == 200

    assert phone.post("/api/auth/logout-all").status_code == 200

    assert phone.get("/api/auth/me").status_code == 401
    assert laptop.get("/api/auth/me").status_code == 401
    # Signing in again works and gives a fresh, valid session.
    again = _second_device(app, user["email"])
    assert again.get("/api/auth/me").status_code == 200


def test_logout_all_requires_login(client):
    assert client.post("/api/auth/logout-all").status_code == 401


def test_password_reset_ends_existing_sessions(app, make_user, client):
    phone, user = make_user()
    client.post("/api/auth/forgot-password", json={"email": user["email"]})
    token = app.state.mailer.outbox[-1]["body"].split("token=")[1].split()[0]
    assert client.post("/api/auth/reset-password",
                       json={"token": token, "new_password": "brand-new-pass"}).status_code == 200

    assert phone.get("/api/auth/me").status_code == 401
    assert _second_device(app, user["email"], "brand-new-pass").get("/api/auth/me").status_code == 200


def test_tokens_issued_before_session_versions_still_work(app, make_user, settings):
    """Deploying this change must not sign everyone out: a token without "sv" counts as version 0."""
    from fastapi.testclient import TestClient

    _, user = make_user()
    now = datetime.now(UTC)
    legacy = jwt.encode({"sub": str(user["id"]), "purpose": "session", "iat": now, "exp": now + timedelta(days=1)},
                        settings.secret_key, algorithm=ALGORITHM)
    c = TestClient(app, cookies={SESSION_COOKIE: legacy})
    assert c.get("/api/auth/me").status_code == 200
