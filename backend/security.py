"""Password hashing and signed tokens (session cookie + password reset)."""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

ALGORITHM = "HS256"
SESSION_COOKIE = "cp_session"
SESSION_DAYS = 30
RESET_MINUTES = 30


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:  # malformed hash (e.g. anonymised account)
        return False


def _encode(payload: dict, secret: str, ttl: timedelta) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({**payload, "iat": now, "exp": now + ttl}, secret, algorithm=ALGORITHM)


def _decode(token: str, secret: str) -> dict | None:
    try:
        return jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None


def create_session_token(user_id: int, secret: str) -> str:
    return _encode({"sub": str(user_id), "purpose": "session"}, secret, timedelta(days=SESSION_DAYS))


def decode_session_token(token: str, secret: str) -> int | None:
    data = _decode(token, secret)
    if not data or data.get("purpose") != "session":
        return None
    return int(data["sub"])


def _password_fingerprint(hashed_password: str) -> str:
    # Changes whenever the password changes, which makes reset links single-use.
    return hashed_password[-10:]


def create_reset_token(user_id: int, hashed_password: str, secret: str) -> str:
    return _encode({"sub": str(user_id), "purpose": "reset", "pwd_fp": _password_fingerprint(hashed_password)},
                   secret, timedelta(minutes=RESET_MINUTES))


def decode_reset_token(token: str, secret: str) -> tuple[int, str] | None:
    data = _decode(token, secret)
    if not data or data.get("purpose") != "reset":
        return None
    return int(data["sub"]), data.get("pwd_fp", "")


def reset_token_matches(fingerprint: str, hashed_password: str) -> bool:
    return fingerprint == _password_fingerprint(hashed_password)
