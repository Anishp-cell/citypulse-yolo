"""Request dependencies: current user and role checks."""

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import User
from backend.security import SESSION_COOKIE, decode_session_token


def current_user_optional(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    uid = decode_session_token(token, request.app.state.settings.secret_key)
    if uid is None:
        return None
    user = db.get(User, uid)
    if user is None or not user.is_active:
        return None
    return user


def current_user(user: User | None = Depends(current_user_optional)) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="not_authenticated")
    return user


def require_roles(*roles: str):
    def _dep(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="forbidden")
        return user

    return _dep


require_staff = require_roles("officer", "admin")
require_admin = require_roles("admin")
