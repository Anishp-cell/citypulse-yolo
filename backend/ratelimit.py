"""In-memory sliding-window rate limiting (CityPulse runs as a single instance)."""

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from backend.security import SESSION_COOKIE, decode_session_token


class RateLimiter:
    def __init__(self):
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: float) -> bool:
        """Record a hit; return False if the key is over its limit."""
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 50_000:  # drop idle keys so spoofed keys cannot grow memory forever
                for k in [k for k, v in self._hits.items() if not v or v[-1] <= now - 86400]:
                    del self._hits[k]
            q = self._hits[key]
            while q and q[0] <= now - window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def client_key(request: Request) -> str:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        uid = decode_session_token(token, request.app.state.settings.secret_key)
        if uid is not None:
            return f"user:{uid}"
    return f"ip:{client_ip(request)}"


def client_ip(request: Request) -> str:
    # Render's proxy appends the address it saw; everything to its left is client-supplied and untrusted.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def check_limit(request: Request, bucket: str, key: str, limit: int, window_s: float) -> None:
    """Imperative variant for keys only known inside the handler (e.g. the email being logged into)."""
    if not request.app.state.settings.rate_limits_enabled:
        return
    if not request.app.state.rate_limiter.hit(f"{bucket}:{key}", limit, window_s):
        raise HTTPException(status_code=429, detail="rate_limited")


def rate_limit(bucket: str, limit: int, window_s: float):
    """FastAPI dependency: `Depends(rate_limit("analyze", 20, 3600))`."""

    def _dep(request: Request) -> None:
        if not request.app.state.settings.rate_limits_enabled:
            return
        if not request.app.state.rate_limiter.hit(f"{bucket}:{client_key(request)}", limit, window_s):
            raise HTTPException(status_code=429, detail="rate_limited")

    return _dep


HOUR = 3600
DAY = 86400
