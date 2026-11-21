"""Officer/admin console: work queue, status updates, moderation, stats, user roles."""

from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, selectinload

from backend.db import get_db
from backend.deps import require_admin, require_staff
from backend.domain.priority import hazard_priority
from backend.domain.status import ACTIVE_STATUSES, HazardStatus, InvalidTransition, apply_transition
from backend.models import ROLES, Flag, Follow, Hazard, Report, User, Verification, aware
from backend.routes.actions import store_photo
from backend.services.notify import notify_event
from backend.services.present import summary

router = APIRouter(prefix="/api/admin", tags=["admin"])

REJECT_REASONS = ("duplicate", "not_a_hazard", "abuse")
QUEUE_DEFAULT = ACTIVE_STATUSES | {HazardStatus.fixed_unverified}


def _get(db: Session, hazard_id: int) -> Hazard:
    h = db.get(Hazard, hazard_id)
    if h is None:
        raise HTTPException(status_code=404, detail="hazard_not_found")
    return h


def _transition(db, h, to, actor, note=None, image_key=None, role=None):
    try:
        return apply_transition(db, h, to, actor=actor, role=role, note=note, image_key=image_key)
    except InvalidTransition as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/queue")
def queue(request: Request, ward: str | None = None, status: HazardStatus | None = None,
          sort: Literal["priority", "age"] = "priority", limit: int = Query(50, ge=1, le=200),
          offset: int = Query(0, ge=0), _: User = Depends(require_staff), db: Session = Depends(get_db)):
    q = db.query(Hazard).options(selectinload(Hazard.reports))
    q = q.filter(Hazard.status == status) if status else q.filter(Hazard.status.in_(list(QUEUE_DEFAULT)))
    if ward:
        q = q.filter(Hazard.ward.ilike(ward))
    total = q.count()
    order = Hazard.priority_score.desc() if sort == "priority" else Hazard.first_reported_at.asc()
    rows = q.order_by(order, Hazard.id).offset(offset).limit(limit).all()
    return {"total": total, "hazards": [summary(h, request.app.state.storage) for h in rows]}


@router.post("/hazards/{hazard_id}/status")
def set_status(hazard_id: int, request: Request, background: BackgroundTasks,
               to: HazardStatus = Form(...), note: str | None = Form(None, max_length=500),
               photo: UploadFile | None = File(None),
               user: User = Depends(require_staff), db: Session = Depends(get_db)):
    h = _get(db, hazard_id)
    if to == HazardStatus.rejected:
        raise HTTPException(status_code=422, detail="use_reject_endpoint")
    if to == HazardStatus.fixed_unverified and (photo is None or not photo.filename):
        raise HTTPException(status_code=422, detail="photo_required")
    image_key = store_photo(request, photo, "proof")
    event = _transition(db, h, to, user, note=note, image_key=image_key)
    h.priority_score = hazard_priority(h, datetime.now(UTC))
    db.commit()
    notify_event(background, request.app, event.id, exclude_user_id=user.id)
    return {"status": h.status.value}


def _resolve_flags(db: Session, hazard_id: int) -> None:
    db.query(Flag).filter_by(hazard_id=hazard_id, resolved=False).update({"resolved": True})


@router.get("/moderation")
def moderation(request: Request, _: User = Depends(require_staff), db: Session = Depends(get_db)):
    storage = request.app.state.storage
    pending = (db.query(Hazard).options(selectinload(Hazard.reports))
               .filter(Hazard.status == HazardStatus.pending_review).order_by(Hazard.first_reported_at).all())
    flagged_ids = [hid for (hid,) in db.query(Flag.hazard_id).filter(Flag.resolved.is_(False)).distinct()]
    flagged = []
    for h in db.query(Hazard).filter(Hazard.id.in_(flagged_ids), Hazard.status != HazardStatus.rejected).all():
        item = summary(h, storage)
        item["flags"] = [f.reason for f in db.query(Flag).filter_by(hazard_id=h.id, resolved=False)]
        flagged.append(item)

    def with_notes(h):
        item = summary(h, storage)
        item["notes"] = [r.note for r in h.reports if r.note]
        item["photo_url"] = storage.url(h.cover_image_key, public=True) if h.cover_image_key else None
        return item

    return {"pending": [with_notes(h) for h in pending], "flagged": flagged}


@router.post("/hazards/{hazard_id}/approve")
def approve(hazard_id: int, request: Request, background: BackgroundTasks,
            user: User = Depends(require_staff), db: Session = Depends(get_db)):
    h = _get(db, hazard_id)
    event = None
    if h.status == HazardStatus.pending_review:
        event = _transition(db, h, HazardStatus.open, user, note="Approved by moderator")
    _resolve_flags(db, h.id)
    db.commit()
    if event is not None:
        notify_event(background, request.app, event.id, exclude_user_id=user.id)
    return {"status": h.status.value}


class RejectIn(BaseModel):
    reason: Literal[REJECT_REASONS]  # type: ignore[valid-type]


@router.post("/hazards/{hazard_id}/reject")
def reject(hazard_id: int, body: RejectIn, user: User = Depends(require_staff), db: Session = Depends(get_db)):
    h = _get(db, hazard_id)
    _transition(db, h, HazardStatus.rejected, user, note=f"Rejected: {body.reason}")
    _resolve_flags(db, h.id)
    db.commit()
    return {"status": h.status.value}


class MergeIn(BaseModel):
    into_id: int


@router.post("/hazards/{hazard_id}/merge")
def merge(hazard_id: int, body: MergeIn, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    if body.into_id == hazard_id:
        raise HTTPException(status_code=422, detail="cannot_merge_into_itself")
    source, target = _get(db, hazard_id), _get(db, body.into_id)
    if target.status == HazardStatus.rejected:
        raise HTTPException(status_code=422, detail="target_rejected")
    db.query(Report).filter_by(hazard_id=source.id).update({"hazard_id": target.id})
    for f in db.query(Follow).filter_by(hazard_id=source.id).all():
        if db.get(Follow, (f.user_id, target.id)) is None:
            db.add(Follow(user_id=f.user_id, hazard_id=target.id))
        db.delete(f)
    db.query(Verification).filter_by(hazard_id=source.id).delete()
    db.flush()
    reporters = db.query(Report.user_id).filter(Report.hazard_id == target.id)
    target.report_count = max(1, reporters.filter(Report.user_id.isnot(None)).distinct().count()
                              + reporters.filter(Report.user_id.is_(None)).count())
    target.first_reported_at = min(aware(target.first_reported_at), aware(source.first_reported_at))
    target.priority_score = hazard_priority(target, datetime.now(UTC))
    _transition(db, source, HazardStatus.rejected, user, note=f"Rejected: duplicate of #{target.id}")
    db.commit()
    return {"merged_into": target.id, "report_count": target.report_count}


@router.get("/stats")
def stats(_: User = Depends(require_staff), db: Session = Depends(get_db)):
    now = datetime.now(UTC)
    by_status = {s.value: 0 for s in HazardStatus}
    by_status.update({s.value: n for s, n in db.query(Hazard.status, func.count()).group_by(Hazard.status)})
    by_type = {t: n for t, n in db.query(Hazard.type, func.count()).filter(Hazard.status != HazardStatus.rejected)
               .group_by(Hazard.type)}
    since = now - timedelta(days=29)
    per_day = Counter(aware(r.created_at).date() for r in
                      db.query(Report.created_at).filter(Report.created_at >= since.replace(hour=0, minute=0)))
    start: date = since.date()
    daily = [{"date": (start + timedelta(days=i)).isoformat(), "count": per_day.get(start + timedelta(days=i), 0)}
             for i in range(30)]
    active = db.query(Hazard).filter(Hazard.status.in_(list(ACTIVE_STATUSES))).all()
    fixed = db.query(Hazard).filter(Hazard.status == HazardStatus.fixed, Hazard.fixed_at.isnot(None)).all()
    fix_days = [(aware(h.fixed_at) - aware(h.first_reported_at)).total_seconds() / 86400 for h in fixed]
    return {
        "totals": {"active": len(active), "fixed": len(fixed),
                   "avg_days_to_fix": round(sum(fix_days) / len(fix_days), 1) if fix_days else None},
        "by_status": by_status,
        "by_type": by_type,
        "daily_reports": daily,
        "heat": [{"lat": h.lat, "lng": h.lng, "weight": round(h.priority_score / 100, 2)} for h in active],
    }


@router.get("/users")
def users(q: str | None = Query(None, max_length=120), limit: int = Query(50, ge=1, le=200),
          offset: int = Query(0, ge=0), _: User = Depends(require_admin), db: Session = Depends(get_db)):
    query = db.query(User).filter(User.is_active.is_(True))
    if q:
        like = f"%{q.lower()}%"
        query = query.filter(or_(User.email.ilike(like), User.full_name.ilike(like)))
    total = query.count()
    rows = query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
    return {"total": total, "users": [{"id": u.id, "email": u.email, "full_name": u.full_name, "role": u.role,
                                       "created_at": aware(u.created_at).isoformat()} for u in rows]}


class RoleIn(BaseModel):
    role: Literal[ROLES]  # type: ignore[valid-type]


@router.patch("/users/{user_id}/role")
def set_role(user_id: int, body: RoleIn, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    if user_id == admin.id:
        raise HTTPException(status_code=409, detail="cannot_change_own_role")
    u = db.get(User, user_id)
    if u is None or not u.is_active:
        raise HTTPException(status_code=404, detail="user_not_found")
    u.role = body.role
    db.commit()
    return {"id": u.id, "role": u.role}
