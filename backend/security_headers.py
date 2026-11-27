"""Browser security headers for every response (CSP, HSTS, framing, referrer, permissions)."""

from urllib.parse import urlsplit

from backend.settings import Settings

MAP_TILES = "https://tile.openstreetmap.org"
# FastAPI's Swagger UI (dev only, disabled in production) loads its assets from a CDN.
CSP_EXEMPT_PREFIXES = ("/docs", "/openapi.json")


def _origin(url: str) -> str | None:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else None


def content_security_policy(settings: Settings) -> str:
    # Public photos may be served from the storage bucket's own domain (R2 / GCS) rather than ours.
    media = " ".join(o for o in [MAP_TILES, _origin(settings.r2_public_base_url)] if o)
    return "; ".join([
        "default-src 'self'",
        "script-src 'self'",
        # React and Leaflet set element styles; 'unsafe-inline' for styles carries little risk.
        "style-src 'self' 'unsafe-inline'",
        f"img-src 'self' data: blob: {media}",
        # The service worker fetches (and caches) map tiles and public photos itself.
        f"connect-src 'self' {media}",
        "worker-src 'self' blob:",  # the image-compression worker runs from a blob: URL
        "font-src 'self'",
        "manifest-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


def build_headers(settings: Settings) -> dict[str, str]:
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(self), geolocation=(self), microphone=(), payment=(), usb=()",
        "Cross-Origin-Opener-Policy": "same-origin",
    }
    if settings.is_production:  # only sent over https, which production guarantees (APP_URL check)
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return headers
