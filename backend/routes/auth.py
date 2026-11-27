"""Accounts: register, login (cookie session), profile, password reset, account deletion."""

import re
import secrets
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import update
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.deps import current_user
from backend.models import Flag, Follow, PushSubscription, Report, Upload, User, Verification
from backend.ratelimit import HOUR, check_limit, rate_limit
from backend.security import (
    SESSION_COOKIE,
    SESSION_DAYS,
    create_reset_token,
    create_session_token,
    decode_reset_token,
    hash_password,
    reset_token_matches,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])
auth_limit = Depends(rate_limit("auth", 10, 60))

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
Language = Literal["en", "hi", "mr"]


def _email(v: str) -> str:
    v = v.strip().lower()
    if len(v) > 320 or not EMAIL_RE.match(v):
        raise ValueError("invalid email")
    return v


class RegisterIn(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=120)
    language: Language = "en"

    _norm = field_validator("email")(lambda cls, v: _email(v))


class LoginIn(BaseModel):
    email: str
    password: str = Field(max_length=128)

    _norm = field_validator("email")(lambda cls, v: v.strip().lower())


class ProfileIn(BaseModel):
    full_name: str | None = Field(default=None, max_length=120)
    language: Language | None = None
    city: str | None = Field(default=None, max_length=80)
    ward: str | None = Field(default=None, max_length=120)


class ForgotIn(BaseModel):
    email: str


class ResetIn(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str | None
    role: str
    language: str
    city: str | None
    ward: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


def _set_session(response: Response, request: Request, user: User) -> None:
    settings = request.app.state.settings
    response.set_cookie(SESSION_COOKIE, create_session_token(user.id, settings.secret_key, user.session_version or 0),
                        max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax",
                        secure=settings.secure_cookies, path="/")


@router.post("/register", response_model=UserOut, dependencies=[auth_limit])
def register(body: RegisterIn, request: Request, response: Response, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=409, detail="email_taken")
    user = User(email=body.email, hashed_password=hash_password(body.password),
                full_name=body.full_name, language=body.language, role="citizen")
    db.add(user)
    db.commit()
    _set_session(response, request, user)
    return user


@router.post("/login", response_model=UserOut, dependencies=[auth_limit])
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    check_limit(request, "login-email", body.email, 10, 15 * 60)  # stops password guessing across many IPs
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not user.is_active or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    _set_session(response, request, user)
    return user


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.post("/logout-all")
def logout_all(response: Response, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """End every session of this account, on every device (e.g. after losing a phone)."""
    user.session_version = (user.session_version or 0) + 1
    db.commit()
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.patch("/me", response_model=UserOut)
def update_me(body: ProfileIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    for field, value in body.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(user, field, value)
    db.commit()
    return user


@router.delete("/me")
def delete_me(response: Response, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Right to erasure: personal data goes, public hazard reports stay but are no longer linked to anyone."""
    uid = user.id
    for model in (Report, Verification, Flag):
        db.execute(update(model).where(model.user_id == uid).values(user_id=None))
    for model in (Follow, PushSubscription, Upload):
        db.query(model).filter(model.user_id == uid).delete()
    user.email = f"deleted-{uid}-{secrets.token_hex(4)}@deleted.invalid"
    user.full_name = None
    user.city = user.ward = None
    user.hashed_password = "!" + secrets.token_hex(16)
    user.is_active = False
    user.deleted_at = datetime.now(UTC)
    db.commit()
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.post("/forgot-password", dependencies=[auth_limit])
def forgot_password(body: ForgotIn, request: Request, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    check_limit(request, "forgot-email", email, 3, HOUR)  # protects inboxes and the SMTP quota
    user = db.query(User).filter(User.email == email, User.is_active.is_(True)).first()
    if user:
        settings = request.app.state.settings
        token = create_reset_token(user.id, user.hashed_password, settings.secret_key)
        link = f"{settings.app_url.rstrip('/')}/reset-password?token={token}"
        request.app.state.mailer.send(
            user.email, "Reset your CityPulse password",
            f"Someone asked to reset the password for your CityPulse account.\n\n"
            f"Open this link within 30 minutes to choose a new password:\n{link}\n\n"
            f"If this wasn't you, you can ignore this email.\n")
    # Same response either way, so the endpoint can't be used to discover accounts.
    return {"ok": True}


@router.post("/reset-password", dependencies=[auth_limit])
def reset_password(body: ResetIn, request: Request, db: Session = Depends(get_db)):
    decoded = decode_reset_token(body.token, request.app.state.settings.secret_key)
    user = db.get(User, decoded[0]) if decoded else None
    if not user or not user.is_active or not reset_token_matches(decoded[1], user.hashed_password):
        raise HTTPException(status_code=400, detail="invalid_or_expired_token")
    user.hashed_password = hash_password(body.new_password)
    user.session_version = (user.session_version or 0) + 1  # sign out everywhere, including an attacker
    db.commit()
    return {"ok": True}


@router.post("/demo-login", response_model=UserOut, include_in_schema=False)
def demo_login(request: Request, response: Response, db: Session = Depends(get_db)):
    if not request.app.state.settings.enable_demo:
        raise HTTPException(status_code=404)
    user = db.query(User).filter(User.email == "demo@citypulse.local").first()
    if not user:
        user = User(email="demo@citypulse.local", hashed_password=hash_password(secrets.token_hex(16)),
                    full_name="Demo Citizen")
        db.add(user)
        db.commit()
    _set_session(response, request, user)
    return user
