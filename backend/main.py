"""CityPulse API + PWA host. `create_app()` builds a fully wired FastAPI instance.

Run locally:  uvicorn backend.main:create_app --factory --reload
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.db import Base, make_engine, make_sessionmaker
from backend.ml.detector import FakeDetector
from backend.ratelimit import RateLimiter
from backend.services.geocode import Geocoder
from backend.services.mailer import build_mailer
from backend.services.push import PushService
from backend.services.storage import build_storage
from backend.settings import Settings

log = logging.getLogger("citypulse")

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

    app = FastAPI(title="CityPulse API", docs_url=None if settings.is_production else "/docs",
                  redoc_url=None, openapi_url=None if settings.is_production else "/openapi.json")
    app.state.settings = settings
    app.state.engine = make_engine(settings.database_url)
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
