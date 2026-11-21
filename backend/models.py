from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Enum, JSON, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from backend.database import Base

class UserRole(str, enum.Enum):
    citizen = "citizen"
    officer = "officer"
    admin = "admin"
    moderator = "moderator"

class IncidentStatus(str, enum.Enum):
    detected = "detected"
    submitted = "submitted"
    validating = "validating"
    routed = "routed"
    acknowledged = "acknowledged"
    in_progress = "in_progress"
    reported_fixed = "reported_fixed"
    citizen_verification = "citizen_verification"
    resolved = "resolved"
    confirmed_fixed = "confirmed_fixed"
    disputed = "disputed"
    closed = "closed"

class HazardSeverity(str, enum.Enum):
    minor = "minor"
    moderate = "moderate"
    severe = "severe"
    critical = "critical"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    phone = Column(String, unique=True, index=True, nullable=True)
    full_name = Column(String, nullable=True)
    role = Column(Enum(UserRole), default=UserRole.citizen, nullable=False)
    district = Column(String, nullable=True)
    home_ward = Column(String, nullable=True)
    trust_score = Column(Float, default=0.50, nullable=False)
    points_balance = Column(Integer, default=0, nullable=False)
    fcm_push_token = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    incidents = relationship("Incident", back_populates="reporter")
    verifications = relationship("Verification", back_populates="verifier")
    point_transactions = relationship("PointTransaction", back_populates="user")
    road_subscriptions = relationship("RoadSubscription", back_populates="user")

class RoadSegment(Base):
    __tablename__ = "road_segments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    ward = Column(String, index=True, nullable=False)
    city = Column(String, index=True, default="Pune", nullable=False)
    length_km = Column(Float, default=1.0, nullable=False)
    health_score = Column(Float, default=100.0, index=True, nullable=False) # 0 to 100
    active_potholes_count = Column(Integer, default=0, nullable=False)
    resolved_potholes_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    clusters = relationship("PotholeCluster", back_populates="road")
    subscriptions = relationship("RoadSubscription", back_populates="road")

class PotholeCluster(Base):
    __tablename__ = "pothole_clusters"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("road_segments.id"), nullable=True)
    lat = Column(Float, index=True, nullable=False)
    lng = Column(Float, index=True, nullable=False)
    address_text = Column(String, nullable=True)
    severity = Column(String, default="moderate", nullable=False)
    priority_score = Column(Float, default=0.0, index=True, nullable=False)
    status = Column(Enum(IncidentStatus), default=IncidentStatus.submitted, nullable=False)
    report_count = Column(Integer, default=1, nullable=False)
    first_reported_at = Column(DateTime(timezone=True), server_default=func.now())
    last_reported_at = Column(DateTime(timezone=True), server_default=func.now())
    last_followup_sent_at = Column(DateTime(timezone=True), nullable=True)
    government_ticket_id = Column(String, nullable=True)
    government_dept = Column(String, nullable=True)

    road = relationship("RoadSegment", back_populates="clusters")
    incidents = relationship("Incident", back_populates="cluster")
    verifications = relationship("Verification", back_populates="cluster")

class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    cluster_id = Column(Integer, ForeignKey("pothole_clusters.id"), nullable=True)
    image_url = Column(String, nullable=True)
    video_url = Column(String, nullable=True)
    severity = Column(String, index=True, nullable=False)
    
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    speed_kmh = Column(Float, default=0.0, nullable=True)
    
    address_text = Column(String, nullable=True)
    detection_results = Column(JSON, nullable=True)
    guidance = Column(JSON, nullable=True)
    
    status = Column(Enum(IncidentStatus), default=IncidentStatus.detected, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    reported_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reporter = relationship("User", back_populates="incidents")
    cluster = relationship("PotholeCluster", back_populates="incidents")
    
    notifications = relationship("Notification", back_populates="incident")

class Verification(Base):
    __tablename__ = "verifications"

    id = Column(Integer, primary_key=True, index=True)
    cluster_id = Column(Integer, ForeignKey("pothole_clusters.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_url = Column(String, nullable=True)
    is_fixed = Column(Boolean, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    cluster = relationship("PotholeCluster", back_populates="verifications")
    verifier = relationship("User", back_populates="verifications")

class RoadSubscription(Base):
    __tablename__ = "road_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    road_id = Column(Integer, ForeignKey("road_segments.id"), nullable=False)
    notify_on_repairs = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="road_subscriptions")
    road = relationship("RoadSegment", back_populates="subscriptions")

class PointTransaction(Base):
    __tablename__ = "point_transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount = Column(Integer, nullable=False) # e.g. +10, +25, +50
    reason = Column(String, nullable=False) # 'SUBMISSION', 'GOV_ACK', 'VERIFICATION_FIX'
    reference_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="point_transactions")

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False)
    channel = Column(String, nullable=False) # 'sms', 'email', 'push'
    recipient = Column(String, nullable=False)
    status = Column(String, nullable=False) # 'sent', 'failed'
    sent_at = Column(DateTime(timezone=True), server_default=func.now())

    incident = relationship("Incident", back_populates="notifications")
