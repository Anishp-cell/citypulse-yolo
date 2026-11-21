"""Things citizens do to a hazard: follow it, record that they filed it with the city, vote on a fix, flag it."""

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.deps import current_user
from backend.domain.priority import hazard_priority
from backend.domain.status import ACTIVE_STATUSES, HazardStatus, apply_transition
from backend.models import Flag, Follow, Hazard, User, Verification
from backend.ratelimit import DAY, rate_limit
from backend.routes.hazards import get_visible_hazard
from backend.services.images import MAX_BYTES, ImageTooLarge, UnsupportedImage, process_upload
from backend.services.notify import notify_event
from backend.services.storage import new_key

router = APIRouter(prefix="/api/hazards", tags=["actions"])

FLAG_REASONS = ("not_a_hazard", "duplicate", "wrong_location", "inappropriate_photo", "other")
FILED_STATUSES = frozenset({HazardStatus.filed, HazardStatus.acknowledged, HazardStatus.in_progress})


def _follow(db: Session, user: User, hazard: Hazard) -> None:
    if db.get(Follow, (user.id, hazard.id)) is None:
        db.add(Follow(user_id=user.id, hazard_id=hazard.id))


def store_photo(request: Request, photo: UploadFile | None, prefix: str) -> str | None:
    """Validate and store an optional proof photo publicly; returns its storage key."""
    if photo is None or not photo.filename:
        return None
    try:
        img = process_upload(photo.file.read(MAX_BYTES + 1))
    except ImageTooLarge:
        raise HTTPException(status_code=413, detail="image_too_large")
    except UnsupportedImage:
        raise HTTPException(status_code=415, detail="unsupported_image")
    key = new_key(prefix)
    request.app.state.storage.put(key, img.full_jpeg, "image/jpeg", public=True)
    return key


@router.post("/{hazard_id}/follow")
def follow(hazard_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    h = get_visible_hazard(db, hazard_id, user)
    _follow(db, user, h)
    db.commit()
    return {"following": True}


@router.delete("/{hazard_id}/follow")
def unfollow(hazard_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.query(Follow).filter_by(user_id=user.id, hazard_id=hazard_id).delete()
    db.commit()
    return {"following": False}


class FiledIn(BaseModel):
    portal: str = Field(min_length=1, max_length=120)
    ticket_id: str = Field(max_length=80)

    @field_validator("ticket_id", "portal")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


@router.post("/{hazard_id}/filed")
def mark_filed(hazard_id: int, body: FiledIn, background: BackgroundTasks, request: Request,
               user: User = Depends(current_user), db: Session = Depends(get_db)):
    h = get_visible_hazard(db, hazard_id, user)
    event = None
    if h.status == HazardStatus.open:
        event = apply_transition(db, h, HazardStatus.filed, actor=user, role="citizen",
                                 note=f"Filed with {body.portal}: {body.ticket_id}")
    elif h.status not in FILED_STATUSES:
        raise HTTPException(status_code=409, detail="cannot_file_in_this_status")
    h.official_portal, h.official_ticket_id = body.portal, body.ticket_id
    h.updated_at = datetime.now(UTC)
    _follow(db, user, h)
    db.commit()
    if event is not None:
        notify_event(background, request.app, event.id, exclude_user_id=user.id)
    return {"status": h.status.value, "official_ticket_id": h.official_ticket_id}


def _tally(db: Session, h: Hazard) -> tuple[int, int]:
    votes = db.query(Verification).filter_by(hazard_id=h.id, round=h.verify_round).all()
    return sum(v.is_fixed for v in votes), sum(not v.is_fixed for v in votes)


@router.post("/{hazard_id}/verify", dependencies=[Depends(rate_limit("verify", 30, DAY))])
def verify(hazard_id: int, request: Request, background: BackgroundTasks,
           is_fixed: bool = Form(...), photo: UploadFile | None = File(None),
           user: User = Depends(current_user), db: Session = Depends(get_db)):
    h = get_visible_hazard(db, hazard_id, user)
    events = []
    if h.status in ACTIVE_STATUSES:
        if not is_fixed:
            raise HTTPException(status_code=409, detail="not_awaiting_verification")
        events.append(apply_transition(db, h, HazardStatus.fixed_unverified, actor=user, role="citizen",
                                       note="A citizen reported it fixed"))
    elif h.status != HazardStatus.fixed_unverified:
        raise HTTPException(status_code=409, detail="not_awaiting_verification")

    if db.query(Verification).filter_by(hazard_id=h.id, user_id=user.id, round=h.verify_round).first():
        raise HTTPException(status_code=409, detail="already_voted")
    image_key = store_photo(request, photo, "verify")
    db.add(Verification(hazard_id=h.id, user_id=user.id, round=h.verify_round, is_fixed=is_fixed,
                        image_key=image_key))
    db.flush()

    fixed_votes, not_fixed_votes = _tally(db, h)
    if fixed_votes >= 2 or (h.fixed_by_officer and fixed_votes >= 1):
        events.append(apply_transition(db, h, HazardStatus.fixed, role="system", note="Confirmed fixed by citizens"))
    elif not_fixed_votes >= 2 or (h.fixed_by_officer and not_fixed_votes >= 1):
        events.append(apply_transition(db, h, HazardStatus.open, role="system",
                                       note="Citizens say it is still there; reopened"))
    h.priority_score = hazard_priority(h, datetime.now(UTC))
    _follow(db, user, h)
    db.commit()
    for ev in events:
        notify_event(background, request.app, ev.id, exclude_user_id=user.id)
    return {"status": h.status.value, "verify_round": h.verify_round, "my_vote": "fixed" if is_fixed else "not_fixed"}


class FlagIn(BaseModel):
    reason: Literal[FLAG_REASONS]  # type: ignore[valid-type]
    details: str | None = Field(default=None, max_length=300)


@router.post("/{hazard_id}/flag", dependencies=[Depends(rate_limit("flag", 10, DAY))])
def flag(hazard_id: int, body: FlagIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    h = get_visible_hazard(db, hazard_id, user)
    if db.query(Flag).filter_by(hazard_id=h.id, user_id=user.id).first():
        raise HTTPException(status_code=409, detail="already_flagged")
    reason = (body.reason if not body.details else f"{body.reason}: {body.details}")[:300]
    db.add(Flag(hazard_id=h.id, user_id=user.id, reason=reason))
    db.commit()
    return {"flagged": True}
