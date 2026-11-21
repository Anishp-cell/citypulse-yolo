import re

from backend.models import Hazard
from tests.helpers import detect_pothole, report


def make_hazard(app, client, **fields):
    detect_pothole(app)
    hid = report(client)["hazard_id"]
    if fields:
        with app.state.SessionLocal() as db:
            h = db.get(Hazard, hid)
            for k, v in fields.items():
                setattr(h, k, v)
            db.commit()
    return hid


def test_complaint_uses_city_config(app, client, citizen):
    c, _ = citizen
    hid = make_hazard(app, c)
    r = client.get(f"/api/hazards/{hid}/complaint")
    assert r.status_code == 200
    body = r.json()
    assert body["portal_url"] == "https://complaint.punecorporation.org/"
    assert body["email_to"] == "feedback@punecorporation.org"
    assert body["authority"].startswith("Pune Municipal Corporation")
    assert "FC Road" in body["subject"] and f"#{hid}" in body["subject"]
    assert f"http://testserver/h/{hid}" in body["body"]
    assert "https://maps.google.com/?q=" in body["body"]
    assert f"http://testserver/h/{hid}" in body["whatsapp_text"]
    assert body["pdf_url"] == f"/api/hazards/{hid}/complaint.pdf"
    assert body["escalation"]["name"].startswith("Aaple Sarkar")


def test_complaint_mentions_ticket_once_filed(app, client, citizen):
    c, _ = citizen
    hid = make_hazard(app, c, official_ticket_id="PMC-777", official_portal="PMC CARE")
    body = client.get(f"/api/hazards/{hid}/complaint").json()
    assert "PMC-777" in body["escalation"]["body"]


def test_generic_city_fallback(app, client, citizen):
    c, _ = citizen
    hid = make_hazard(app, c, lat=28.61, lng=77.21, city="Delhi")
    body = client.get(f"/api/hazards/{hid}/complaint").json()
    assert body["portal_url"] is None and body["email_to"] is None
    assert body["escalation"]["url"] == "https://pgportal.gov.in/"


def test_complaint_text_translated_hi(app, client, citizen):
    c, _ = citizen
    hid = make_hazard(app, c)
    body = client.get(f"/api/hazards/{hid}/complaint", params={"lang": "hi"}).json()
    assert re.search(r"[ऀ-ॿ]", body["body"]) and re.search(r"[ऀ-ॿ]", body["subject"])


def test_complaint_text_translated_mr(app, client, citizen):
    c, _ = citizen
    hid = make_hazard(app, c)
    body = client.get(f"/api/hazards/{hid}/complaint", params={"lang": "mr"}).json()
    assert re.search(r"[ऀ-ॿ]", body["body"])


def test_complaint_unknown_lang_422(app, client, citizen):
    c, _ = citizen
    hid = make_hazard(app, c)
    assert client.get(f"/api/hazards/{hid}/complaint", params={"lang": "fr"}).status_code == 422


def test_pdf_has_no_reporter_pii(app, client, make_user):
    c, _ = make_user(email="private.reporter@example.com")
    hid = make_hazard(app, c)
    r = client.get(f"/api/hazards/{hid}/complaint.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    pdf = r.content
    assert pdf.startswith(b"%PDF")
    assert f"CP-{hid}".encode() in pdf and b"FC Road" in pdf
    assert b"Citizen Road Hazard Complaint" in pdf
    assert b"private.reporter" not in pdf and b"Test User" not in pdf
    assert b"Official" not in pdf  # must not pose as a government document


def test_pdf_for_hidden_hazard_404(app, client, citizen):
    c, _ = citizen
    hid = make_hazard(app, c, status="pending_review")
    assert client.get(f"/api/hazards/{hid}/complaint.pdf").status_code == 404
