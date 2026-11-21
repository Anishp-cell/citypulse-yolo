import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# Use environment variable or default to local DB
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/citypulse_db"
)

_SQLITE_FALLBACK = "sqlite:///" + str(Path(__file__).resolve().parent.parent / "citypulse.db")

def _build_engine(url: str):
    """Create an engine, optionally with SQLite-specific settings."""
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url)

def _try_connect(url: str) -> bool:
    """Return True if the database at *url* is reachable."""
    try:
        eng = _build_engine(url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False

# Try configured URL first; fall back to SQLite for local/demo use
if DATABASE_URL.startswith("sqlite") or _try_connect(DATABASE_URL):
    engine = _build_engine(DATABASE_URL)
    print(f"Database: connected to {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
else:
    print(f"[WARNING] PostgreSQL unavailable -- falling back to local SQLite at {_SQLITE_FALLBACK}")
    DATABASE_URL = _SQLITE_FALLBACK
    engine = _build_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
