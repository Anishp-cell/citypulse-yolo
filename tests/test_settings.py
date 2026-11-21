import pytest

from backend.settings import DEFAULT_SECRET, Settings, SettingsError


def prod(**kw):
    base = dict(
        _env_file=None,
        app_env="production",
        secret_key="x" * 40,
        database_url="postgresql+psycopg://u:p@db.example.com/citypulse",
        storage_backend="r2",
        r2_account_id="acc",
        r2_access_key_id="key",
        r2_secret_access_key="secret",
        r2_public_bucket="pub",
        r2_private_bucket="priv",
        r2_public_base_url="https://media.example.com",
        smtp_host="smtp.example.com",
        smtp_from="noreply@example.com",
        app_url="https://citypulse.example.com",
    )
    base.update(kw)
    return Settings(**base)


def test_valid_production_config_passes():
    prod().validate_for_env()


def test_prod_rejects_default_secret():
    with pytest.raises(SettingsError, match="SECRET_KEY"):
        prod(secret_key=DEFAULT_SECRET).validate_for_env()


def test_prod_rejects_short_secret():
    with pytest.raises(SettingsError, match="SECRET_KEY"):
        prod(secret_key="short").validate_for_env()


def test_prod_rejects_sqlite():
    with pytest.raises(SettingsError, match="DATABASE_URL"):
        prod(database_url="sqlite:///x.db").validate_for_env()


def test_prod_rejects_local_storage():
    with pytest.raises(SettingsError, match="R2"):
        prod(storage_backend="local").validate_for_env()


def test_dev_allows_sqlite():
    Settings(_env_file=None, app_env="dev", database_url="sqlite:///x.db").validate_for_env()


def test_postgres_url_normalised_to_psycopg3():
    s = Settings(_env_file=None, database_url="postgres://u:p@h/db")
    assert s.database_url.startswith("postgresql+psycopg://")


def test_geocoder_off_returns_coordinates(tmp_path):
    from backend.services.geocode import Geocoder

    s = Settings(_env_file=None, geocoder="off")
    assert Geocoder(s).reverse(None, 18.5, 73.8).display == "18.50000, 73.80000"


def test_dev_falls_back_to_sqlite_when_postgres_unreachable(tmp_path, caplog):
    from backend.main import create_app

    s = Settings(_env_file=None, app_env="dev", detector="fake", media_dir=str(tmp_path / "m"),
                 database_url="postgresql+psycopg://u:p@127.0.0.1:1/nope",
                 dev_sqlite_path=str(tmp_path / "fallback.db"))
    app = create_app(s)
    assert str(app.state.engine.url).startswith("sqlite")
    assert "falling back to SQLite" in caplog.text


def test_production_never_falls_back(tmp_path):
    import pytest as _pytest

    from backend.main import create_app

    s = prod(database_url="postgresql+psycopg://u:p@127.0.0.1:1/nope", detector="fake")
    with _pytest.raises(RuntimeError, match="unreachable"):
        create_app(s)


def test_prod_requires_smtp_for_password_reset():
    # Review I3: without SMTP, reset links would only exist in logs.
    with pytest.raises(SettingsError, match="SMTP"):
        prod(smtp_host="").validate_for_env()


def test_prod_requires_https_app_url():
    with pytest.raises(SettingsError, match="APP_URL"):
        prod(app_url="http://localhost:8000").validate_for_env()


def test_prod_accepts_s3_endpoint_instead_of_r2_account():
    prod(r2_account_id="", s3_endpoint_url="https://storage.googleapis.com").validate_for_env()
