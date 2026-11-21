"""The signed-in citizen's own hazards and impact."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from backend.db import get_db
from backend.deps import current_user
from backend.domain.status import HazardStatus
from backend.models import Follow, Hazard, Report, User, Verification
from backend.services.present import summary

router = APIRouter(prefix="/api/me", tags=["me"])


@router.get("/reports")
def my_reports(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    last_report = (db.query(Report.hazard_id, func.max(Report.created_at).label("at"))
                   .filter(Report.user_id == user.id).group_by(Report.hazard_id).subquery())
    rows = (db.query(Hazard).join(last_report, last_report.c.hazard_id == Hazard.id)
            .options(selectinload(Hazard.reports)).filter(Hazard.status != HazardStatus.rejected)
            .order_by(last_report.c.at.desc()).all())
    return {"hazards": [summary(h, request.app.state.storage) for h in rows]}


@router.get("/following")
def following(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = (db.query(Hazard).join(Follow, Follow.hazard_id == Hazard.id).filter(Follow.user_id == user.id)
            .options(selectinload(Hazard.reports)).filter(Hazard.status != HazardStatus.rejected)
            .order_by(Hazard.updated_at.desc()).all())
    return {"hazards": [summary(h, request.app.state.storage) for h in rows]}


@router.get("/impact")
def impact(user: User = Depends(current_user), db: Session = Depends(get_db)):
    hazard_ids = db.query(Report.hazard_id).filter(Report.user_id == user.id).distinct()
    return {
        "reports": db.query(Report).filter(Report.user_id == user.id).count(),
        "hazards_reported": hazard_ids.count(),
        "hazards_fixed": db.query(Hazard).filter(Hazard.id.in_(hazard_ids),
                                                 Hazard.status == HazardStatus.fixed).count(),
        "verifications": db.query(Verification).filter(Verification.user_id == user.id).count(),
    }
