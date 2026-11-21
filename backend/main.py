"""CityPulse API + PWA host. `create_app()` builds a fully wired FastAPI instance."""

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.db import Base, make_engine, make_sessionmaker
from backend.ml.detector import FakeDetector
from backend.settings import Settings

log = logging.getLogger("citypulse")


def _build_detector(settings: Settings):
    if settings.detector == "fake":
        return FakeDetector()
    from backend.ml.detector import OnnxPotholeDetector

    return OnnxPotholeDetector(settings.model_path)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.validate_for_env()

    app = FastAPI(title="CityPulse API", docs_url=None if settings.is_production else "/docs")
    app.state.settings = settings
    app.state.engine = make_engine(settings.database_url)
    app.state.SessionLocal = make_sessionmaker(app.state.engine)
    app.state.detector = _build_detector(settings)

    if not settings.is_production:
        # Production schema is managed by Alembic (`alembic upgrade head` runs before the server starts).
        import backend.models  # noqa: F401  (registers tables on Base.metadata)

        Base.metadata.create_all(app.state.engine)

    @app.get("/health")
    def health():
        db_ok = model_ok = True
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

    return app

