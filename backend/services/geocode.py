"""Reverse geocoding (lat/lng -> road, ward, city) with a database cache.

Uses Nominatim by default (max 1 request/second, identifying User-Agent, per its usage policy) or
LocationIQ when LOCATIONIQ_KEY is set. Failures never block a report: we fall back to coordinates.
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass

import httpx
from sqlalchemy.orm import Session

from backend.models import GeocodeCache

log = logging.getLogger("citypulse.geocode")


@dataclass(frozen=True)
class Place:
    road: str | None
    ward: str | None
    city: str | None
    display: str


def _fallback(lat: float, lng: float) -> Place:
    return Place(road=None, ward=None, city=None, display=f"{lat:.5f}, {lng:.5f}")


def _parse(data: dict, lat: float, lng: float) -> Place:
    a = data.get("address") or {}
    road = a.get("road") or a.get("pedestrian") or a.get("highway")
    ward = a.get("suburb") or a.get("city_district") or a.get("neighbourhood") or a.get("quarter")
    city = a.get("city") or a.get("town") or a.get("village") or a.get("county")
    return Place(road=road, ward=ward, city=city, display=data.get("display_name") or _fallback(lat, lng).display)


class Geocoder:
    def __init__(self, settings, transport: httpx.BaseTransport | None = None, min_interval_s: float = 1.0):
        self.settings = settings
        self.client = httpx.Client(transport=transport, timeout=6.0,
                                   headers={"User-Agent": f"CityPulse/1.0 (+{settings.app_url})"})
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_call = 0.0

    def _request(self, lat: float, lng: float) -> dict:
        if self.settings.locationiq_key:
            url = "https://us1.locationiq.com/v1/reverse"
            params = {"key": self.settings.locationiq_key, "lat": lat, "lon": lng, "format": "json",
                      "addressdetails": 1, "normalizeaddress": 0}
        else:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {"lat": lat, "lon": lng, "format": "jsonv2", "zoom": 18, "addressdetails": 1}
        with self._lock:  # stay within the provider's rate limit
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            resp = self.client.get(url, params=params, headers={"Accept-Language": "en"})
        resp.raise_for_status()
        return resp.json()

    def reverse(self, db: Session, lat: float, lng: float) -> Place:
        if self.settings.geocoder == "off":
            return _fallback(lat, lng)
        key = f"{round(lat, 4):.4f},{round(lng, 4):.4f}"
        cached = db.get(GeocodeCache, key)
        if cached:
            return Place(**cached.payload)
        try:
            place = _parse(self._request(lat, lng), lat, lng)
        except Exception as e:  # network error, 4xx/5xx, bad JSON
            log.warning("reverse geocode failed for %s: %s", key, e)
            return _fallback(lat, lng)
        db.merge(GeocodeCache(key=key, payload=asdict(place)))
        db.commit()
        return place
