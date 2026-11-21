"""Turns hazards into API payloads. Public payloads never contain reporter identities."""

from datetime import UTC, datetime

from backend.domain.status import ACTIVE_STATUSES, PUBLIC_STATUSES, HazardStatus
from backend.models import Follow, Hazard, User, Verification, aware

REMINDER_PREFIX = "reminder:"


def days_open(h: Hazard, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    end = aware(h.fixed_at) if h.status == HazardStatus.fixed and h.fixed_at else now
    return max(0, (end - aware(h.first_reported_at)).days)


def summary(h: Hazard, storage) -> dict:
    thumb = h.reports[0].thumb_key if h.reports else None
    return {
        "id": h.id, "type": h.type, "severity": h.severity, "status": h.status.value,
        "lat": h.lat, "lng": h.lng, "road": h.road, "ward": h.ward, "city": h.city,
        "report_count": h.report_count, "days_open": days_open(h), "ai_verified": h.ai_verified,
        "priority_score": h.priority_score,
        "thumb_url": storage.url(thumb, public=True) if thumb else None,
        "first_reported_at": aware(h.first_reported_at).isoformat(),
    }


def is_staff(user: User | None) -> bool:
    return user is not None and user.role in ("officer", "admin")


def can_view(h: Hazard, user: User | None) -> bool:
    if h.status in PUBLIC_STATUSES:
        return True
    if is_staff(user):
        return True
    return user is not None and h.status == HazardStatus.pending_review and any(
        r.user_id == user.id for r in h.reports)


def timeline(h: Hazard) -> list[dict]:
    return [{"status": e.to_status.value, "from": e.from_status.value if e.from_status else None,
             "at": aware(e.created_at).isoformat(), "note": e.note}
            for e in h.events if not (e.note or "").startswith(REMINDER_PREFIX)]


def detail(db, h: Hazard, user: User | None, storage, app_url: str) -> dict:
    out = summary(h, storage)
    out.update({
        "address": h.address,
        "photo_url": storage.url(h.cover_image_key, public=True) if h.cover_image_key else None,
        "photos": [storage.url(r.public_image_key, public=True) for r in h.reports if r.public_image_key][:12],
        "timeline": timeline(h),
        "official_ticket_id": h.official_ticket_id,
        "official_portal": h.official_portal,
        "fixed_at": aware(h.fixed_at).isoformat() if h.fixed_at else None,
        "verify_round": h.verify_round,
        "is_active": h.status in ACTIVE_STATUSES,
        "share_url": f"{app_url.rstrip('/')}/h/{h.id}",
        "is_following": False,
        "my_vote": None,
        "is_reporter": False,
    })
    if user is not None:
        out["is_following"] = db.get(Follow, (user.id, h.id)) is not None
        vote = db.query(Verification).filter_by(hazard_id=h.id, user_id=user.id, round=h.verify_round).first()
        out["my_vote"] = None if vote is None else ("fixed" if vote.is_fixed else "not_fixed")
        own = [r for r in h.reports if r.user_id == user.id]
        out["is_reporter"] = bool(own)
        visible = h.reports if is_staff(user) else own
        if visible:
            out["full_photos"] = [storage.url(r.image_key, public=False) for r in visible]
        if is_staff(user):
            out["notes"] = [r.note for r in h.reports if r.note]
    return out
