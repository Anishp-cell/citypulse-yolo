"""Application configuration, loaded from environment variables (and `.env` in dev)."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SECRET = "dev-insecure-secret-change-me"


class SettingsError(RuntimeError):
    """Raised when the configuration is unsafe for the current environment."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "test", "production"] = "dev"
    app_url: str = "http://localhost:8000"
    secret_key: str = DEFAULT_SECRET
    database_url: str = f"sqlite:///{ROOT / 'citypulse-dev.db'}"

    # Media storage
    storage_backend: Literal["local", "r2"] = "local"
    media_dir: str = str(ROOT / "media")
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_public_bucket: str = ""
    r2_private_bucket: str = ""
    r2_public_base_url: str = ""

    # ML
    detector: Literal["onnx", "fake"] = "onnx"
    model_path: str = str(ROOT / "backend" / "ml" / "models" / "pothole-v1.onnx")

    # Web push (VAPID)
    vapid_public_key: str = ""
    vapid_private_key: str = ""
    vapid_subject: str = "mailto:admin@example.com"

    # Outgoing email (password reset only)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""

    job_token: str = ""
    sentry_dsn: str = ""
    locationiq_key: str = ""
    enable_demo: bool = False
    rate_limits_enabled: bool = True
    frontend_dist: str = str(ROOT / "app" / "dist")

    @field_validator("database_url")
    @classmethod
    def _normalise_db_url(cls, v: str) -> str:
        # Neon/Render hand out postgres:// URLs; SQLAlchemy needs the psycopg3 driver name.
        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix):]
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def secure_cookies(self) -> bool:
        return self.is_production

    def validate_for_env(self) -> None:
        if not self.is_production:
            return
        if self.secret_key == DEFAULT_SECRET or len(self.secret_key) < 32:
            raise SettingsError("SECRET_KEY must be set to a random value of at least 32 characters in production")
        if not self.database_url.startswith("postgresql"):
            raise SettingsError("DATABASE_URL must point to PostgreSQL in production")
        if self.storage_backend != "r2":
            raise SettingsError("STORAGE_BACKEND must be 'r2' (Cloudflare R2) in production; local disk is ephemeral")
        required = ["r2_account_id", "r2_access_key_id", "r2_secret_access_key", "r2_public_bucket",
                    "r2_private_bucket", "r2_public_base_url"]
        missing = [name.upper() for name in required if not getattr(self, name)]
        if missing:
            raise SettingsError(f"R2 storage is not fully configured; missing: {', '.join(missing)}")
