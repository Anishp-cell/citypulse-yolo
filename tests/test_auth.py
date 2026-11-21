from fastapi.testclient import TestClient

from backend.models import Follow, Hazard, Report, User


def register(client, email="a@example.com", password="correct-horse-1", **extra):
    return client.post("/api/auth/register", json={"email": email, "password": password, **extra})


def test_register_returns_user_and_sets_session(client):
    r = register(client, full_name="Asha")
    assert r.status_code == 200
    body = r.json()
    assert body["email"] == "a@example.com" and body["role"] == "citizen" and body["full_name"] == "Asha"
    assert "hashed_password" not in body
    assert client.get("/api/auth/me").json()["id"] == body["id"]


def test_register_ignores_role_field(client):
    register(client, role="admin")
    assert client.get("/api/auth/me").json()["role"] == "citizen"


def test_register_duplicate_email_409(client):
    register(client)
    assert register(client, email="A@example.com").status_code == 409


def test_password_min_8(client):
    r = register(client, password="short")
    assert r.status_code == 422


def test_invalid_email_422(client):
    assert register(client, email="not-an-email").status_code == 422


def test_email_case_insensitive(app):
    with TestClient(app, headers={"X-Requested-With": "citypulse"}) as c:
        register(c, email="Mixed@Example.COM")
        c.post("/api/auth/logout")
        r = c.post("/api/auth/login", json={"email": "mixed@example.com", "password": "correct-horse-1"})
        assert r.status_code == 200


def test_login_sets_httponly_cookie(client):
    register(client)
    client.post("/api/auth/logout")
    r = client.post("/api/auth/login", json={"email": "a@example.com", "password": "correct-horse-1"})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "cp_session=" in cookie and "httponly" in cookie and "samesite=lax" in cookie


def test_login_wrong_password_401(client):
    register(client)
    r = client.post("/api/auth/login", json={"email": "a@example.com", "password": "wrong-password"})
    assert r.status_code == 401


def test_me_requires_auth(client):
    assert client.get("/api/auth/me").status_code == 401


def test_logout_clears_session(client):
    register(client)
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401


def test_unsafe_request_without_csrf_header_403(app):
    with TestClient(app) as bare:
        r = bare.post("/api/auth/login", json={"email": "a@example.com", "password": "x" * 8})
        assert r.status_code == 403


def test_patch_me_updates_profile(client):
    register(client)
    r = client.patch("/api/auth/me", json={"full_name": "New Name", "language": "mr", "role": "admin"})
    assert r.status_code == 200
    assert r.json()["full_name"] == "New Name" and r.json()["language"] == "mr" and r.json()["role"] == "citizen"


def test_patch_me_rejects_unknown_language(client):
    register(client)
    assert client.patch("/api/auth/me", json={"language": "xx"}).status_code == 422


def test_forgot_password_always_200(app, client):
    r = client.post("/api/auth/forgot-password", json={"email": "nobody@example.com"})
    assert r.status_code == 200
    assert app.state.mailer.outbox == []


def _reset_token_from_outbox(app):
    body = app.state.mailer.outbox[-1]["body"]
    return body.split("token=")[1].split()[0]


def test_reset_password_flow_and_single_use(app, client):
    register(client)
    client.post("/api/auth/forgot-password", json={"email": "a@example.com"})
    assert app.state.mailer.outbox[-1]["to"] == "a@example.com"
    token = _reset_token_from_outbox(app)
    r = client.post("/api/auth/reset-password", json={"token": token, "new_password": "brand-new-pass"})
    assert r.status_code == 200
    again = client.post("/api/auth/reset-password", json={"token": token, "new_password": "another-pass1"})
    assert again.status_code == 400
    login = client.post("/api/auth/login", json={"email": "a@example.com", "password": "brand-new-pass"})
    assert login.status_code == 200


def test_reset_with_garbage_token_400(client):
    r = client.post("/api/auth/reset-password", json={"token": "nope", "new_password": "brand-new-pass"})
    assert r.status_code == 400


def test_delete_account_anonymizes_reports(app, client):
    uid = register(client).json()["id"]
    with app.state.SessionLocal() as db:
        h = Hazard(type="pothole", lat=18.5, lng=73.8)
        db.add(h)
        db.flush()
        db.add(Report(hazard_id=h.id, user_id=uid, image_key="a", public_image_key="b", thumb_key="c",
                      lat=18.5, lng=73.8))
        db.add(Follow(user_id=uid, hazard_id=h.id))
        db.commit()
    r = client.delete("/api/auth/me")
    assert r.status_code == 200
    assert client.get("/api/auth/me").status_code == 401
    with app.state.SessionLocal() as db:
        assert db.query(Report).one().user_id is None
        assert db.query(Follow).count() == 0
        u = db.get(User, uid)
        assert u.is_active is False and u.email != "a@example.com" and u.full_name is None
    login = client.post("/api/auth/login", json={"email": "a@example.com", "password": "correct-horse-1"})
    assert login.status_code == 401


def test_auth_rate_limited(app, client):
    app.state.settings.rate_limits_enabled = True
    codes = [client.post("/api/auth/login", json={"email": "x@example.com", "password": "whatever1"}).status_code
             for _ in range(11)]
    assert codes[:10] == [401] * 10
    assert codes[10] == 429


def test_demo_login_disabled_by_default(client):
    assert client.post("/api/auth/demo-login").status_code == 404


def test_cli_create_admin(app, settings):
    from backend.cli import create_admin

    create_admin(settings, "Boss@Example.com", "admin-password-1")
    with app.state.SessionLocal() as db:
        u = db.query(User).filter_by(email="boss@example.com").one()
        assert u.role == "admin"
    # Running it again promotes/updates instead of failing.
    create_admin(settings, "boss@example.com", "admin-password-2")


def test_rate_limit_ignores_spoofed_forwarded_for(app, client):
    # Review I2: the client controls the left-most X-Forwarded-For entry; the proxy appends the real one.
    app.state.settings.rate_limits_enabled = True
    codes = [client.post("/api/auth/login", json={"email": "x@example.com", "password": "whatever1"},
                         headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.7"}).status_code for i in range(11)]
    assert codes[10] == 429


def test_login_limited_per_email_across_ips(app, client):
    app.state.settings.rate_limits_enabled = True
    codes = [client.post("/api/auth/login", json={"email": "victim@example.com", "password": "guess-pass"},
                         headers={"X-Forwarded-For": f"198.51.100.{i}"}).status_code for i in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429


def test_forgot_password_limited_per_email(app, client):
    app.state.settings.rate_limits_enabled = True
    codes = [client.post("/api/auth/forgot-password", json={"email": "victim@example.com"},
                         headers={"X-Forwarded-For": f"198.51.100.{i}"}).status_code for i in range(4)]
    assert codes[:3] == [200] * 3 and codes[3] == 429


def test_console_mailer_never_logs_reset_token_in_production_like_mode(caplog):
    from backend.services.mailer import ConsoleMailer

    ConsoleMailer(log_body=False).send("a@example.com", "Reset", "link token=SECRET123")
    assert "SECRET123" not in caplog.text
