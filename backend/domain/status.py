"""Hazard lifecycle: the single source of truth for which status changes are allowed and by whom."""

import enum
from datetime import UTC, datetime

from sqlalchemy.orm import Session


class HazardStatus(str, enum.Enum):
    pending_review = "pending_review"
    open = "open"
    filed = "filed"
    acknowledged = "acknowledged"
    in_progress = "in_progress"
    fixed_unverified = "fixed_unverified"
    fixed = "fixed"
    rejected = "rejected"


S = HazardStatus
ACTIVE_STATUSES = frozenset({S.open, S.filed, S.acknowledged, S.in_progress})
PUBLIC_STATUSES = frozenset(ACTIVE_STATUSES | {S.fixed_unverified, S.fixed})
#: Statuses a new report can be merged into (same physical hazard, still unresolved).
MERGEABLE_STATUSES = frozenset(ACTIVE_STATUSES | {S.pending_review})

STAFF = frozenset({"officer", "admin"})
ANYONE = frozenset({"citizen", "officer", "admin"})

# (from, to) -> roles allowed to make that change. "system" = automatic rules (votes, daily job).
_TRANSITIONS: dict[tuple[HazardStatus, HazardStatus], frozenset[str]] = {
    (S.pending_review, S.open): STAFF,
    (S.pending_review, S.rejected): STAFF,
    (S.open, S.filed): ANYONE,
    (S.open, S.acknowledged): STAFF,
    (S.filed, S.acknowledged): STAFF,
    (S.open, S.in_progress): STAFF,
    (S.filed, S.in_progress): STAFF,
    (S.acknowledged, S.in_progress): STAFF,
    (S.fixed_unverified, S.fixed): frozenset({"system", "admin"}),
    (S.fixed_unverified, S.open): frozenset({"system", "admin"}),
}
for _s in ACTIVE_STATUSES:
    _TRANSITIONS[(_s, S.fixed_unverified)] = ANYONE
for _s in S:
    if _s not in (S.rejected, S.pending_review):
        _TRANSITIONS.setdefault((_s, S.rejected), frozenset({"admin"}))


class InvalidTransition(ValueError):
    pass


def can_transition(from_: HazardStatus, to: HazardStatus, role: str) -> bool:
    return role in _TRANSITIONS.get((HazardStatus(from_), HazardStatus(to)), frozenset())


def apply_transition(db: Session, hazard, to: HazardStatus, actor=None, *, role: str | None = None,
                     note: str | None = None, image_key: str | None = None):
    """Move `hazard` to `to`, recording a StatusEvent. `role` defaults to the actor's role, or "system"."""
    from backend.models import StatusEvent

    to = HazardStatus(to)
    role = role or (actor.role if actor is not None else "system")
    from_ = HazardStatus(hazard.status)
    if not can_transition(from_, to, role):
        raise InvalidTransition(f"{role} cannot move hazard from {from_.value} to {to.value}")

    now = datetime.now(UTC)
    hazard.status = to
    hazard.updated_at = now
    if to == S.fixed_unverified:
        hazard.verify_round = (hazard.verify_round or 0) + 1
        hazard.fixed_by_officer = role in STAFF
    if to == S.fixed:
        hazard.fixed_at = now
    if to == S.open and from_ == S.fixed_unverified:
        hazard.dispute_count = (hazard.dispute_count or 0) + 1
        hazard.fixed_by_officer = False

    event = StatusEvent(hazard=hazard, from_status=from_, to_status=to,
                        actor_id=actor.id if actor is not None else None, note=note, image_key=image_key)
    db.add(event)
    db.flush()
    return event


def record_creation(db: Session, hazard, actor=None):
    """Write the first timeline entry for a newly created hazard."""
    from backend.models import StatusEvent

    event = StatusEvent(hazard=hazard, from_status=None, to_status=HazardStatus(hazard.status),
                        actor_id=actor.id if actor is not None else None, note="reported")
    db.add(event)
    db.flush()
    return event
