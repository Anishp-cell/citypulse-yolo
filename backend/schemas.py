from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime
from backend.models import UserRole, IncidentStatus

# ---- User Schemas ----
class UserBase(BaseModel):
    email: str
    phone: Optional[str] = None
    role: UserRole = UserRole.citizen
    district: Optional[str] = None

class UserCreate(UserBase):
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[UserRole] = None

class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

# ---- Incident Schemas ----
class IncidentBase(BaseModel):
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    severity: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    address_text: Optional[str] = None
    detection_results: Optional[List[Dict[str, Any]]] = None
    guidance: Optional[Dict[str, Any]] = None
    status: IncidentStatus = IncidentStatus.detected

class IncidentCreate(IncidentBase):
    reported_by: Optional[int] = None

class IncidentResponse(IncidentBase):
    id: int
    created_at: datetime
    reported_by: Optional[int] = None

    class Config:
        orm_mode = True

# ---- Notification Schemas ----
class NotificationBase(BaseModel):
    incident_id: int
    channel: str
    recipient: str
    status: str

class NotificationCreate(NotificationBase):
    pass

class NotificationResponse(NotificationBase):
    id: int
    sent_at: datetime

    class Config:
        orm_mode = True
