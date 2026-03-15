from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
import enum
from backend.database import Base

class UserRole(str, enum.Enum):
    citizen = "citizen"
    officer = "officer"
    admin = "admin"

class IncidentStatus(str, enum.Enum):
    detected = "detected"
    notified = "notified"
    acknowledged = "acknowledged"
    resolved = "resolved"
    closed = "closed"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)  # Added password hash
    phone = Column(String, unique=True, index=True, nullable=True)
    role = Column(Enum(UserRole), default=UserRole.citizen, nullable=False)
    district = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    incidents = relationship("Incident", back_populates="reporter")

class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    image_url = Column(String, nullable=True)
    video_url = Column(String, nullable=True)
    severity = Column(String, index=True, nullable=False)
    
    # Store standard lat/lng for easy JSON serialization
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    
    # PostGIS Geography point
    location = Column(Geometry('POINT', srid=4326), nullable=True)
    
    address_text = Column(String, nullable=True)
    detection_results = Column(JSON, nullable=True)
    guidance = Column(JSON, nullable=True)
    
    status = Column(Enum(IncidentStatus), default=IncidentStatus.detected, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    reported_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reporter = relationship("User", back_populates="incidents")
    
    notifications = relationship("Notification", back_populates="incident")

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False)
    channel = Column(String, nullable=False) # e.g., 'sms', 'email', 'push'
    recipient = Column(String, nullable=False)
    status = Column(String, nullable=False) # e.g., 'sent', 'failed'
    sent_at = Column(DateTime(timezone=True), server_default=func.now())

    incident = relationship("Incident", back_populates="notifications")
