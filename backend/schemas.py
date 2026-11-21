from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any, Dict
from datetime import datetime
from backend.models import UserRole, IncidentStatus

# ---- User Schemas ----
class UserBase(BaseModel):
    email: str
    phone: Optional[str] = None
    full_name: Optional[str] = None
    role: UserRole = UserRole.citizen
    district: Optional[str] = None
    home_ward: Optional[str] = None

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
    trust_score: float = 0.50
    points_balance: int = 0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# ---- Incident Schemas ----
class IncidentBase(BaseModel):
    cluster_id: Optional[int] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    severity: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    speed_kmh: Optional[float] = 0.0
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

    model_config = ConfigDict(from_attributes=True)

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

    model_config = ConfigDict(from_attributes=True)
