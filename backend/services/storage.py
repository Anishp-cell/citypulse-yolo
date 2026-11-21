"""Media storage. Public objects (crops, thumbnails) are served directly; private ones (full photos)
only through short-lived signed URLs."""

import hashlib
import hmac
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode

SIGNED_URL_TTL_S = 3600


def new_key(prefix: str, ext: str = "jpg") -> str:
    month = datetime.now(timezone.utc).strftime("%Y%m")
    return f"{prefix}/{month}/{uuid.uuid4().hex}.{ext}"


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str, public: bool) -> None: ...
    def get(self, key: str, public: bool) -> bytes | None: ...
    def url(self, key: str, public: bool) -> str: ...
    def delete(self, key: str) -> None: ...


def _safe_key(key: str) -> str:
    parts = Path(key).parts
    if not key or key.startswith(("/", "\\")) or any(p in ("..", "") for p in parts) or ":" in key:
        raise ValueError("invalid storage key")
    return key


class LocalStorage:
    """Disk storage for development and tests. Private URLs are HMAC-signed like R2 presigned URLs."""

    def __init__(self, root: str, secret: str):
        self.root = Path(root)
        self.secret = secret.encode()

    def _path(self, key: str, public: bool) -> Path:
        return self.root / ("public" if public else "private") / _safe_key(key)

    def put(self, key: str, data: bytes, content_type: str, public: bool) -> None:
        path = self._path(key, public)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read(self, key: str, public: bool) -> bytes | None:
        try:
            path = self._path(key, public)
        except ValueError:
            return None
        return path.read_bytes() if path.is_file() else None

    get = read

    def _sig(self, key: str, exp: str) -> str:
        return hmac.new(self.secret, f"{key}:{exp}".encode(), hashlib.sha256).hexdigest()

    def verify_signature(self, key: str, exp: str, sig: str) -> bool:
        try:
            if int(exp) < time.time():
                return False
        except ValueError:
            return False
        return hmac.compare_digest(self._sig(key, exp), sig)

    def url(self, key: str, public: bool) -> str:
        if public:
            return f"/media/public/{key}"
        exp = str(int(time.time()) + SIGNED_URL_TTL_S)
        return f"/media/private/{key}?{urlencode({'exp': exp, 'sig': self._sig(key, exp)})}"

    def delete(self, key: str) -> None:
        for public in (True, False):
            try:
                self._path(key, public).unlink(missing_ok=True)
            except ValueError:
                pass


class R2Storage:
    """Cloudflare R2 via its S3-compatible API."""

    def __init__(self, account_id: str, access_key_id: str, secret_access_key: str, public_bucket: str,
                 private_bucket: str, public_base_url: str):
        import boto3
        from botocore.config import Config

        self.client = boto3.client(
            "s3", endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,
            region_name="auto", config=Config(signature_version="s3v4", retries={"max_attempts": 3}))
        self.public_bucket, self.private_bucket = public_bucket, private_bucket
        self.public_base_url = public_base_url.rstrip("/")

    def _bucket(self, public: bool) -> str:
        return self.public_bucket if public else self.private_bucket

    def put(self, key: str, data: bytes, content_type: str, public: bool) -> None:
        extra = {"CacheControl": "public, max-age=31536000, immutable"} if public else {}
        self.client.put_object(Bucket=self._bucket(public), Key=_safe_key(key), Body=data,
                               ContentType=content_type, **extra)

    def get(self, key: str, public: bool) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self._bucket(public), Key=key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def url(self, key: str, public: bool) -> str:
        if public:
            return f"{self.public_base_url}/{key}"
        return self.client.generate_presigned_url("get_object", Params={"Bucket": self.private_bucket, "Key": key},
                                                  ExpiresIn=SIGNED_URL_TTL_S)

    def delete(self, key: str) -> None:
        for bucket in (self.public_bucket, self.private_bucket):
            self.client.delete_object(Bucket=bucket, Key=key)


def build_storage(settings) -> Storage:
    if settings.storage_backend == "r2":
        return R2Storage(settings.r2_account_id, settings.r2_access_key_id, settings.r2_secret_access_key,
                         settings.r2_public_bucket, settings.r2_private_bucket, settings.r2_public_base_url)
    return LocalStorage(settings.media_dir, settings.secret_key)
