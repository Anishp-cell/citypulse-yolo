"""Machine-to-machine endpoints, authenticated with the JOB_TOKEN header (exempt from the CSRF check)."""

import hmac
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.services.jobs import run_daily

router = APIRouter(prefix="/api/internal", include_in_schema=False)


@router.post("/jobs/daily")
def daily(request: Request, x_job_token: str = Header(default=""), db: Session = Depends(get_db)):
    expected = request.app.state.settings.job_token
    if not expected or not hmac.compare_digest(x_job_token, expected):
        raise HTTPException(status_code=401, detail="invalid_job_token")
    return run_daily(db, request.app.state.push, request.app.state.storage, datetime.now(timezone.utc))
