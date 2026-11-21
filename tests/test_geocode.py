import httpx

from backend.cities import city_for, generic_city
from backend.services.geocode import Geocoder, Place

NOMINATIM_FC_ROAD = {
    "display_name": "Fergusson College Road, Shivajinagar, Pune, Maharashtra, India",
    "address": {"road": "Fergusson College Road", "suburb": "Shivajinagar", "city": "Pune"},
}


def make(handler, settings):
    return Geocoder(settings, transport=httpx.MockTransport(handler), min_interval_s=0)


def test_geocode_parses_road_ward_city(app, settings):
    g = make(lambda req: httpx.Response(200, json=NOMINATIM_FC_ROAD), settings)
    with app.state.SessionLocal() as db:
        p = g.reverse(db, 18.5236, 73.8410)
    assert p == Place(road="Fergusson College Road", ward="Shivajinagar", city="Pune",
                      display="Fergusson College Road, Shivajinagar, Pune, Maharashtra, India")


def test_geocode_sends_user_agent(app, settings):
    seen = {}

    def handler(req):
        seen["ua"] = req.headers["user-agent"]
        return httpx.Response(200, json=NOMINATIM_FC_ROAD)

    with app.state.SessionLocal() as db:
        make(handler, settings).reverse(db, 18.5, 73.8)
    assert seen["ua"].startswith("CityPulse/1.0")


def test_geocode_cached(app, settings):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json=NOMINATIM_FC_ROAD)

    g = make(handler, settings)
    with app.state.SessionLocal() as db:
        g.reverse(db, 18.52361, 73.84101)
        g.reverse(db, 18.52364, 73.84104)  # same 4-decimal cell
    assert len(calls) == 1


def test_geocode_failure_falls_back(app, settings):
    g = make(lambda req: httpx.Response(503), settings)
    with app.state.SessionLocal() as db:
        p = g.reverse(db, 18.52, 73.85)
    assert p == Place(road=None, ward=None, city=None, display="18.52000, 73.85000")


def test_city_for_pune_and_generic():
    assert city_for(18.52, 73.85).slug == "pune"
    assert city_for(28.61, 77.21) == generic_city()
