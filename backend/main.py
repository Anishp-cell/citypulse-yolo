"""CityPulse API + PWA host. `create_app()` builds a fully wired FastAPI instance.

Run locally:  uvicorn backend.main:create_app --factory --reload
"""

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend import observability
from backend.db import Base, can_connect, make_engine, make_sessionmaker
from backend.ml.detector import FakeDetector
from backend.ratelimit import RateLimiter
from backend.services.geocode import Geocoder
from backend.services.mailer import build_mailer
from backend.services.push import PushService
from backend.services.storage import build_storage
from backend.settings import Settings

log = logging.getLogger("citypulse")
access_log = logging.getLogger("citypulse.access")

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "citypulse"
CSRF_EXEMPT_PREFIXES = ("/api/internal/",)  # machine-to-machine, authenticated by token header


def _build_detector(settings: Settings):
    if settings.detector == "fake":
        return FakeDetector()
    from backend.ml.detector import OnnxPotholeDetector

    return OnnxPotholeDetector(settings.model_path)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.validate_for_env()
    observability.configure(settings)

    app = FastAPI(title="CityPulse API", docs_url=None if settings.is_production else "/docs",
                  redoc_url=None, openapi_url=None if settings.is_production else "/openapi.json")
    app.state.settings = settings
    engine = make_engine(settings.database_url)
    if not settings.database_url.startswith("sqlite") and not can_connect(engine):
        if settings.app_env != "dev":
            raise RuntimeError("database is unreachable; check DATABASE_URL")
        log.warning("PostgreSQL at DATABASE_URL is unreachable — falling back to SQLite at %s (dev only)",
                    settings.dev_sqlite_path)
        engine.dispose()
        engine = make_engine(f"sqlite:///{settings.dev_sqlite_path}")
    app.state.engine = engine
    app.state.SessionLocal = make_sessionmaker(app.state.engine)
    app.state.detector = _build_detector(settings)
    app.state.mailer = build_mailer(settings)
    app.state.rate_limiter = RateLimiter()
    app.state.storage = build_storage(settings)
    app.state.geocoder = Geocoder(settings)
    app.state.push = PushService(settings, app.state.SessionLocal)

    if not settings.is_production:
        # Production schema is managed by Alembic (`alembic upgrade head` runs before the server starts).
        import backend.models  # noqa: F401  (registers tables on Base.metadata)

        Base.metadata.create_all(app.state.engine)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        if request.url.path.startswith("/api/"):
            access_log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code,
                            (time.perf_counter() - start) * 1000, extra={"request_id": rid})
        return response

    @app.middleware("http")
    async def csrf_guard(request: Request, call_next):
        # The session cookie is SameSite=Lax; requiring a custom header on unsafe API calls blocks
        # cross-site form posts, which cannot set custom headers.
        path = request.url.path
        if (request.method in UNSAFE_METHODS and path.startswith("/api/")
                and not path.startswith(CSRF_EXEMPT_PREFIXES)
                and request.headers.get(CSRF_HEADER) != CSRF_VALUE):
            return JSONResponse({"detail": "csrf_header_missing"}, status_code=403)
        return await call_next(request)

    from backend.routes import actions, admin, auth, hazards, internal, me, media, push, reports, spa

    app.include_router(auth.router)
    app.include_router(reports.router)
    app.include_router(hazards.router)
    app.include_router(actions.router)
    app.include_router(me.router)
    app.include_router(admin.router)
    app.include_router(push.router)
    app.include_router(internal.router)
    app.include_router(media.router)

    @app.get("/health")
    def health():
        db_ok = True
        try:
            with app.state.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            log.exception("health: database check failed")
            db_ok = False
        model_ok = app.state.detector is not None
        body = {"status": "ok" if db_ok and model_ok else "error",
                "db": "ok" if db_ok else "error",
                "model": "ok" if model_ok else "error"}
        return JSONResponse(body, status_code=200 if db_ok and model_ok else 503)

    app.include_router(spa.router)  # last: catch-all for the PWA
    return app
