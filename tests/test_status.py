import pytest

from backend.domain.status import (
    ACTIVE_STATUSES,
    PUBLIC_STATUSES,
    HazardStatus as S,
    InvalidTransition,
    apply_transition,
    can_transition,
)
from backend.models import Hazard, StatusEvent, User

ALLOWED = [
    (S.pending_review, S.open, "officer"),
    (S.pending_review, S.open, "admin"),
    (S.pending_review, S.rejected, "officer"),
    (S.open, S.filed, "citizen"),
    (S.open, S.acknowledged, "officer"),
    (S.filed, S.acknowledged, "officer"),
    (S.acknowledged, S.in_progress, "officer"),
    (S.filed, S.in_progress, "admin"),
    (S.open, S.fixed_unverified, "citizen"),
    (S.in_progress, S.fixed_unverified, "officer"),
    (S.fixed_unverified, S.fixed, "system"),
    (S.fixed_unverified, S.open, "system"),
    (S.open, S.rejected, "admin"),
    (S.fixed, S.rejected, "admin"),
]


@pytest.mark.parametrize("from_,to,role", ALLOWED)
def test_allowed_transitions(from_, to, role):
    assert can_transition(from_, to, role)


def test_citizen_cannot_acknowledge():
    assert not can_transition(S.open, S.acknowledged, "citizen")


def test_officer_cannot_reject_public_hazard():
    assert not can_transition(S.open, S.rejected, "officer")


def test_citizen_cannot_confirm_fixed_directly():
    assert not can_transition(S.fixed_unverified, S.fixed, "citizen")


def test_rejected_is_terminal():
    for to in S:
        assert not can_transition(S.rejected, to, "admin")


def test_status_sets():
    assert ACTIVE_STATUSES == {S.open, S.filed, S.acknowledged, S.in_progress}
    assert S.pending_review not in PUBLIC_STATUSES
    assert S.rejected not in PUBLIC_STATUSES
    assert {S.fixed, S.fixed_unverified} <= PUBLIC_STATUSES


def _hazard(db, status=S.open):
    h = Hazard(type="pothole", severity="medium", status=status, lat=18.52, lng=73.85)
    db.add(h)
    db.commit()
    return h


def test_transition_writes_event(app):
    with app.state.SessionLocal() as db:
        user = User(email="o@example.com", hashed_password="x", role="officer")
        db.add(user)
        h = _hazard(db)
        ev = apply_transition(db, h, S.acknowledged, actor=user, note="on it")
        db.commit()
        assert h.status == S.acknowledged
        stored = db.query(StatusEvent).filter_by(hazard_id=h.id).one()
        assert (stored.from_status, stored.to_status, stored.actor_id, stored.note) == (
            S.open, S.acknowledged, user.id, "on it")
        assert ev.id == stored.id


def test_invalid_transition_raises_and_leaves_status(app):
    with app.state.SessionLocal() as db:
        citizen = User(email="c@example.com", hashed_password="x", role="citizen")
        db.add(citizen)
        h = _hazard(db)
        with pytest.raises(InvalidTransition):
            apply_transition(db, h, S.acknowledged, actor=citizen)
        assert h.status == S.open


def test_entering_fixed_unverified_starts_new_verify_round(app):
    with app.state.SessionLocal() as db:
        h = _hazard(db)
        assert h.verify_round == 0
        apply_transition(db, h, S.fixed_unverified, actor=None, role="citizen")
        assert h.verify_round == 1


def test_fixed_sets_fixed_at(app):
    with app.state.SessionLocal() as db:
        h = _hazard(db, S.fixed_unverified)
        apply_transition(db, h, S.fixed, actor=None, role="system")
        assert h.fixed_at is not None
