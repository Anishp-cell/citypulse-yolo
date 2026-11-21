"""How urgently a hazard needs fixing, 0-100. Higher = more dangerous / more neglected."""

import math
from datetime import datetime

TYPE_WEIGHT = {"open_manhole": 40.0, "pothole": 25.0, "waterlogging": 20.0, "debris": 15.0, "other": 10.0}
SEVERITY_BONUS = {"low": 0.0, "medium": 10.0, "high": 20.0}
BASE_EXPOSURE = 30.0  # stands in for traffic volume until we have real road-class data


def compute_priority(hazard_type: str, severity: str, report_count: int, first_reported_at: datetime,
                     now: datetime, dispute_count: int = 0) -> float:
    days_open = max(0.0, (now - first_reported_at).total_seconds() / 86400)
    base = (TYPE_WEIGHT.get(hazard_type, 10.0) + SEVERITY_BONUS.get(severity, 10.0) + BASE_EXPOSURE
            + 3.0 * math.log1p(days_open))
    reports_mult = 1.0 + 0.15 * min(4, max(0, report_count - 1))
    dispute_mult = 1.5 if dispute_count > 0 else 1.0
    return round(min(100.0, base * reports_mult * dispute_mult), 1)


def hazard_priority(hazard, now: datetime) -> float:
    first = hazard.first_reported_at
    if first.tzinfo is None:  # SQLite returns naive datetimes
        from datetime import timezone

        first = first.replace(tzinfo=timezone.utc)
    return compute_priority(hazard.type, hazard.severity, hazard.report_count, first, now, hazard.dispute_count)
