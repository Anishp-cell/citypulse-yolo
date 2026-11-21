"""Browser push subscriptions."""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.deps import current_user
from backend.models import PushSubscription, User

router = APIRouter(prefix="/api/push", tags=["push"])


class Keys(BaseModel):
    p256dh: str = Field(min_length=1, max_length=200)
    auth: str = Field(min_length=1, max_length=100)


class SubscriptionIn(BaseModel):
    endpoint: str = Field(max_length=2000)
    keys: Keys

    @field_validator("endpoint")
    @classmethod
    def _https(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("push endpoint must be https")
        return v


class EndpointIn(BaseModel):
    endpoint: str = Field(max_length=2000)


@router.get("/vapid-public-key")
def vapid_public_key(request: Request):
    push = request.app.state.push
    return {"enabled": push.enabled, "key": request.app.state.settings.vapid_public_key if push.enabled else None}


@router.post("/subscribe")
def subscribe(body: SubscriptionIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    sub = db.query(PushSubscription).filter_by(endpoint=body.endpoint).first()
    if sub is None:
        sub = PushSubscription(endpoint=body.endpoint, user_id=user.id, p256dh=body.keys.p256dh, auth=body.keys.auth)
        db.add(sub)
    else:  # the same browser now belongs to whoever is signed in
        sub.user_id, sub.p256dh, sub.auth = user.id, body.keys.p256dh, body.keys.auth
    db.commit()
    return {"subscribed": True}


@router.delete("/subscribe")
def unsubscribe(body: EndpointIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.query(PushSubscription).filter_by(endpoint=body.endpoint, user_id=user.id).delete()
    db.commit()
    return {"subscribed": False}
