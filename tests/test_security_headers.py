from fastapi.testclient import TestClient

from backend.security_headers import build_headers, content_security_policy
from backend.settings import Settings


def test_every_response_gets_browser_security_headers(client):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    csp = r.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert "https://tile.openstreetmap.org" in csp
    assert "Strict-Transport-Security" not in r.headers  # only in production, where https is guaranteed


def test_csrf_rejections_also_get_the_headers(app):
    with TestClient(app) as bare:  # no X-Requested-With header
        r = bare.post("/api/auth/logout")
    assert r.status_code == 403
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in r.headers


def test_api_docs_are_exempt_from_csp(client):
    r = client.get("/docs")  # Swagger UI loads from a CDN; it's disabled in production anyway
    assert r.status_code == 200
    assert "Content-Security-Policy" not in r.headers
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_csp_allows_the_public_photo_bucket():
    csp = content_security_policy(Settings(_env_file=None, r2_public_base_url="https://pub-abc.r2.dev/photos"))
    img_src = next(d for d in csp.split("; ") if d.startswith("img-src"))
    connect_src = next(d for d in csp.split("; ") if d.startswith("connect-src"))
    assert "https://pub-abc.r2.dev" in img_src and "https://pub-abc.r2.dev" in connect_src
    assert "/photos" not in csp  # CSP sources are origins


def test_hsts_in_production_only():
    assert "Strict-Transport-Security" in build_headers(Settings(_env_file=None, app_env="production"))
    assert "Strict-Transport-Security" not in build_headers(Settings(_env_file=None, app_env="dev"))
