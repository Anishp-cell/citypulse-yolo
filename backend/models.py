"""Database schema. Roads and wards are derived from hazards, not stored separately."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base
from backend.domain.status import HazardStatus

HAZARD_TYPES = ("pothole", "open_manhole", "waterlogging", "debris", "other")
SEVERITIES = ("low", "medium", "high")
ROLES = ("citizen", "officer", "admin")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _enum(*values, name):
    return Enum(*values, name=name, native_enum=False, validate_strings=True)


_status_enum = Enum(HazardStatus, name="hazard_status", native_enum=False, validate_strings=True,
                    values_callable=lambda e: [m.value for m in e])


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(200))
    full_name: Mapped[str | None] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(_enum(*ROLES, name="user_role"), default="citizen")
    city: Mapped[str | None] = mapped_column(String(80))
    ward: Mapped[str | None] = mapped_column(String(120))
    language: Mapped[str] = mapped_column(String(5), default="en")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Hazard(Base):
    __tablename__ = "hazards"
    __table_args__ = (
        Index("ix_hazards_lat_lng", "lat", "lng"),
        Index("ix_hazards_city_status", "city", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(_enum(*HAZARD_TYPES, name="hazard_type"))
    severity: Mapped[str] = mapped_column(_enum(*SEVERITIES, name="hazard_severity"), default="medium")
    status: Mapped[HazardStatus] = mapped_column(_status_enum, default=HazardStatus.open)
    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    address: Mapped[str | None] = mapped_column(String(300))
    road: Mapped[str | None] = mapped_column(String(200), index=True)
    ward: Mapped[str | None] = mapped_column(String(120), index=True)
    city: Mapped[str | None] = mapped_column(String(80))
    priority_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    report_count: Mapped[int] = mapped_column(Integer, default=1)
    ai_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    cover_image_key: Mapped[str | None] = mapped_column(String(200))
    official_ticket_id: Mapped[str | None] = mapped_column(String(80))
    official_portal: Mapped[str | None] = mapped_column(String(120))
    verify_round: Mapped[int] = mapped_column(Integer, default=0)
    fixed_by_officer: Mapped[bool] = mapped_column(Boolean, default=False)
    dispute_count: Mapped[int] = mapped_column(Integer, default=0)
    first_reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    fixed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    reports: Mapped[list["Report"]] = relationship(back_populates="hazard", order_by="Report.id")
    events: Mapped[list["StatusEvent"]] = relationship(back_populates="hazard", order_by="StatusEvent.id")


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (UniqueConstraint("user_id", "idempotency_key", name="uq_reports_user_idem"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    hazard_id: Mapped[int] = mapped_column(ForeignKey("hazards.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    image_key: Mapped[str] = mapped_column(String(200))
    public_image_key: Mapped[str] = mapped_column(String(200))
    thumb_key: Mapped[str] = mapped_column(String(200))
    ai_label: Mapped[str | None] = mapped_column(String(40))
    ai_confidence: Mapped[float | None] = mapped_column(Float)
    ai_boxes: Mapped[list | None] = mapped_column(JSON)
    location_source: Mapped[str] = mapped_column(_enum("gps", "manual", name="location_source"), default="gps")
    gps_accuracy_m: Mapped[float | None] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    note: Mapped[str | None] = mapped_column(String(500))
    idempotency_key: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    hazard: Mapped[Hazard] = relationship(back_populates="reports")


class Upload(Base):
    """A photo that has been analysed but not yet turned into a report (purged after 24h)."""

    __tablename__ = "uploads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    image_key: Mapped[str] = mapped_column(String(200))
    public_image_key: Mapped[str] = mapped_column(String(200))
    thumb_key: Mapped[str] = mapped_column(String(200))
    ai: Mapped[dict] = mapped_column(JSON)
    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    accuracy_m: Mapped[float | None] = mapped_column(Float)
    location_source: Mapped[str] = mapped_column(String(10), default="gps")
    place: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class StatusEvent(Base):
    __tablename__ = "status_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    hazard_id: Mapped[int] = mapped_column(ForeignKey("hazards.id", ondelete="CASCADE"), index=True)
    from_status: Mapped[HazardStatus | None] = mapped_column(_status_enum)
    to_status: Mapped[HazardStatus] = mapped_column(_status_enum)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    note: Mapped[str | None] = mapped_column(String(500))
    image_key: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    hazard: Mapped[Hazard] = relationship(back_populates="events")


class Verification(Base):
    __tablename__ = "verifications"
    __table_args__ = (UniqueConstraint("hazard_id", "user_id", "round", name="uq_verification_round"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    hazard_id: Mapped[int] = mapped_column(ForeignKey("hazards.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    round: Mapped[int] = mapped_column(Integer)
    is_fixed: Mapped[bool] = mapped_column(Boolean)
    image_key: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Follow(Base):
    __tablename__ = "follows"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    hazard_id: Mapped[int] = mapped_column(ForeignKey("hazards.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Flag(Base):
    __tablename__ = "flags"

    id: Mapped[int] = mapped_column(primary_key=True)
    hazard_id: Mapped[int] = mapped_column(ForeignKey("hazards.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reason: Mapped[str] = mapped_column(String(300))
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PushSubscription(Base):
    __tablename__ = "push_subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    endpoint: Mapped[str] = mapped_column(Text, unique=True)
    p256dh: Mapped[str] = mapped_column(String(200))
    auth: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GeocodeCache(Base):
    __tablename__ = "geocode_cache"

    key: Mapped[str] = mapped_column(String(40), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
