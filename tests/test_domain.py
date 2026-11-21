from datetime import datetime, timedelta, timezone

import pytest

from backend.domain.geo import bbox_around, haversine_m
from backend.domain.priority import compute_priority
from tests.helpers import offset

NOW = datetime(2026, 11, 21, tzinfo=timezone.utc)


def test_haversine_known_distance():
    # Shaniwar Wada -> Pune railway station, ~2.3 km
    assert haversine_m(18.5195, 73.8553, 18.5289, 73.8744) == pytest.approx(2310, rel=0.05)


def test_offset_helper_matches_haversine():
    lat, lng = offset(18.52, 73.85, north_m=12, east_m=5)
    assert haversine_m(18.52, 73.85, lat, lng) == pytest.approx(13, abs=0.5)


def test_bbox_contains_radius():
    min_lat, max_lat, min_lng, max_lng = bbox_around(18.52, 73.85, 30)
    lat, lng = offset(18.52, 73.85, north_m=29, east_m=29)
    assert min_lat <= lat <= max_lat and min_lng <= lng <= max_lng


def p(**kw):
    base = dict(hazard_type="pothole", severity="medium", report_count=1, first_reported_at=NOW, now=NOW,
                dispute_count=0)
    base.update(kw)
    return compute_priority(**base)


def test_priority_increases_with_reports():
    assert p(report_count=3) > p(report_count=1)


def test_priority_report_bonus_caps_at_five_reports():
    assert p(report_count=5) == p(report_count=50)


def test_priority_increases_with_age():
    assert p(first_reported_at=NOW - timedelta(days=30)) > p()


def test_open_manhole_outranks_pothole():
    assert p(hazard_type="open_manhole") > p(hazard_type="pothole")


def test_high_severity_outranks_low():
    assert p(severity="high") > p(severity="low")


def test_dispute_boosts_priority():
    assert p(dispute_count=1) == pytest.approx(min(100.0, p() * 1.5), abs=0.1)


def test_priority_capped_at_100():
    assert p(hazard_type="open_manhole", severity="high", report_count=10, dispute_count=2,
             first_reported_at=NOW - timedelta(days=400)) == 100.0
