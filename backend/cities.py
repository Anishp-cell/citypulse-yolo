"""Where complaints go, per city. Citizens send complaints themselves; this only tells them where.

Contact details are public information. VERIFY them on the city's official website before each release
(`verified` stays False until a human has checked them).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CityConfig:
    slug: str
    name: str
    bbox: tuple[float, float, float, float] | None  # min_lat, min_lng, max_lat, max_lng
    authority: str
    portal_name: str | None
    portal_url: str | None
    grievance_email: str | None
    whatsapp: str | None
    helpline: str | None
    escalation_name: str | None
    escalation_url: str | None
    verified: bool = False


CITIES = [
    CityConfig(
        slug="pune",
        name="Pune",
        bbox=(18.40, 73.70, 18.65, 74.05),
        authority="Pune Municipal Corporation (PMC)",
        portal_name="PMC CARE",
        portal_url="https://complaint.punecorporation.org/",
        grievance_email="feedback@punecorporation.org",
        whatsapp="+919689900002",
        helpline="18001030222",
        escalation_name="Aaple Sarkar (Government of Maharashtra grievance portal)",
        escalation_url="https://aaplesarkar.mahaonline.gov.in/",
        verified=False,
    ),
]

_GENERIC = CityConfig(
    slug="generic", name="your city", bbox=None, authority="your municipal corporation",
    portal_name=None, portal_url=None, grievance_email=None, whatsapp=None, helpline=None,
    escalation_name="CPGRAMS (national public grievance portal)", escalation_url="https://pgportal.gov.in/",
    verified=False,
)


def generic_city() -> CityConfig:
    return _GENERIC


def city_for(lat: float, lng: float) -> CityConfig:
    for city in CITIES:
        if city.bbox and city.bbox[0] <= lat <= city.bbox[2] and city.bbox[1] <= lng <= city.bbox[3]:
            return city
    return _GENERIC


def city_by_name(name: str | None) -> CityConfig:
    for city in CITIES:
        if name and city.name.lower() == name.lower():
            return city
    return _GENERIC
