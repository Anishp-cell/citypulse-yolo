"""Daily maintenance, triggered by a scheduled GitHub Actions workflow. Every step is idempotent."""

import logging
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from backend.domain.priority import hazard_priority
from backend.domain.status import ACTIVE_STATUSES, HazardStatus, apply_transition
from backend.models import Follow, Hazard, Report, StatusEvent, Upload, Verification, aware
from backend.services.present import REMINDER_PREFIX

log = logging.getLogger("citypulse.jobs")

FILE_REMINDER_AFTER = timedelta(days=7)
ESCALATE_AFTER = timedelta(days=21)
AUTO_CONFIRM_AFTER = timedelta(days=14)
UPLOAD_TTL = timedelta(hours=24)


def _entered_at(h: Hazard, status: HazardStatus) -> datetime | None:
    times = [aware(e.created_at) for e in h.events if e.to_status == status and e.from_status != status]
    return max(times) if times else None


def _reminded(h: Hazard, kind: str) -> bool:
    return any(e.note == f"{REMINDER_PREFIX}{kind}" for e in h.events)


def _mark_reminded(db: Session, h: Hazard, kind: str, now: datetime) -> None:
    db.add(StatusEvent(hazard_id=h.id, from_status=h.status, to_status=h.status,
                       note=f"{REMINDER_PREFIX}{kind}", created_at=now))


def run_daily(db: Session, push, storage, now: datetime) -> dict:
    report = {"priorities_updated": 0, "reminders_file": 0, "reminders_escalate": 0, "auto_confirmed": 0,
              "uploads_purged": 0}

    for h in db.query(Hazard).filter(Hazard.status.in_(list(ACTIVE_STATUSES))).all():
        h.priority_score = hazard_priority(h, now)
        report["priorities_updated"] += 1

        if (h.status == HazardStatus.open and not h.official_ticket_id and not _reminded(h, "file")
                and now - aware(h.first_reported_at) >= FILE_REMINDER_AFTER):
            reporters = {uid for (uid,) in db.query(Report.user_id).filter(Report.hazard_id == h.id,
                                                                           Report.user_id.isnot(None))}
            push.send_to_users(db, reporters, "remind_file", h)
            _mark_reminded(db, h, "file", now)
            report["reminders_file"] += 1

        filed_at = _entered_at(h, HazardStatus.filed)
        if (h.status == HazardStatus.filed and filed_at and now - filed_at >= ESCALATE_AFTER
                and not _reminded(h, "escalate")):
            followers = {uid for (uid,) in db.query(Follow.user_id).filter(Follow.hazard_id == h.id)}
            push.send_to_users(db, followers, "remind_escalate", h)
            _mark_reminded(db, h, "escalate", now)
            report["reminders_escalate"] += 1

    for h in db.query(Hazard).filter(Hazard.status == HazardStatus.fixed_unverified).all():
        since = _entered_at(h, HazardStatus.fixed_unverified)
        disputed = db.query(Verification).filter_by(hazard_id=h.id, round=h.verify_round, is_fixed=False).count()
        if since and now - since >= AUTO_CONFIRM_AFTER and disputed == 0:
            apply_transition(db, h, HazardStatus.fixed, role="system",
                             note="Confirmed automatically: nobody disputed the fix for 14 days")
            followers = {uid for (uid,) in db.query(Follow.user_id).filter(Follow.hazard_id == h.id)}
            push.send_to_users(db, followers, "fixed", h)
            report["auto_confirmed"] += 1

    for u in db.query(Upload).all():
        if now - aware(u.created_at) < UPLOAD_TTL:
            continue
        if u.consumed_at is None:  # never turned into a report: its photos are orphans
            for key in (u.image_key, u.public_image_key, u.thumb_key):
                try:
                    storage.delete(key)
                except Exception:
                    log.exception("could not delete %s", key)
            report["uploads_purged"] += 1
        db.delete(u)

    db.commit()
    log.info("daily job: %s", report)
    return report
