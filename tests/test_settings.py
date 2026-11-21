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
