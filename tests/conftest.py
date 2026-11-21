import itertools

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.settings import Settings

_counter = itertools.count(1)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        app_env="test",
        app_url="http://testserver",
        secret_key="test-secret-key-0123456789abcdef",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        media_dir=str(tmp_path / "media"),
        detector="fake",
        rate_limits_enabled=False,
        frontend_dist=str(tmp_path / "dist"),
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


def _new_client(app):
    return TestClient(app, headers={"X-Requested-With": "citypulse"})


@pytest.fixture
def client(app):
    with _new_client(app) as c:
        yield c


@pytest.fixture
def make_user(app):
    """Factory: make_user(role="citizen") -> (TestClient logged in, user dict)."""
    clients = []

    def _make(role="citizen", email=None, password="correct-horse-1"):
        c = _new_client(app)
        c.__enter__()
        clients.append(c)
        email = email or f"user{next(_counter)}@example.com"
        r = c.post("/api/auth/register", json={"email": email, "password": password, "full_name": "Test User"})
        assert r.status_code == 200, r.text
        user = r.json()
        if role != "citizen":
            from backend.models import User

            with app.state.SessionLocal() as db:
                u = db.get(User, user["id"])
                u.role = role
                db.commit()
            user["role"] = role
        return c, user

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def citizen(make_user):
    return make_user("citizen")


@pytest.fixture
def officer(make_user):
    return make_user("officer")


@pytest.fixture
def admin(make_user):
    return make_user("admin")
