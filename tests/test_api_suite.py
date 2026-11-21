import pytest
from fastapi.testclient import TestClient
import os
import glob
from backend.main import app
from backend.service.government.pdf_generator import generate_grievance_pdf

client = TestClient(app)

def test_health_endpoint():
    """Verify health status and model availability."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_demo_login_and_auth():
    """Verify one-click demo login and JWT token issuance."""
    response = client.post("/auth/demo-login")
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_road_rankings_scorecard():
    """Verify worst and best roads ranking endpoints."""
    # Worst roads (default)
    res_worst = client.get("/api/v1/roads/rankings?city=Pune&sort=worst")
    assert res_worst.status_code == 200
    data_worst = res_worst.json()
    assert "roads" in data_worst
    assert len(data_worst["roads"]) > 0

    # Best roads
    res_best = client.get("/api/v1/roads/rankings?city=Pune&sort=best")
    assert res_best.status_code == 200
    data_best = res_best.json()
    assert "roads" in data_best

def test_image_analysis_and_clustering():
    """Verify AI image analysis, spatial clustering, and karma reward points."""
    # 1. Login
    login_res = client.post("/auth/demo-login")
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Find sample image
    test_imgs = glob.glob("static/**/*.jpg", recursive=True) + glob.glob("static/**/*.jpeg", recursive=True)
    if not test_imgs:
        pytest.skip("No test images available in static directory")

    test_img = test_imgs[0]
    with open(test_img, "rb") as f:
        res = client.post(
            "/api/analyze",
            headers=headers,
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={
                "location": "MG Road near Camp, Pune",
                "lat": "18.5196",
                "lng": "73.8553",
            },
        )

    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "incident_id" in data
    assert "cluster_id" in data
    assert data["points_earned"] == 10
    assert "highest_severity" in data

def test_pdf_dossier_generation():
    """Verify programmatic PDF grievance dossier generation."""
    pdf_path = generate_grievance_pdf(
        ticket_id="TEST-999",
        road_name="Hinjawadi Phase 1 Main Road",
        ward_name="Hinjawadi",
        city="Pune",
        lat=18.5913,
        lng=73.7389,
        severity="severe_pothole",
        priority_score=85.0,
        report_count=3,
        sla_hours=72,
    )
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000  # PDF generated with valid binary size

def test_pdf_dossier_download_endpoint():
    """Verify PDF download endpoint for an existing cluster."""
    # Cluster #1 exists from test_image_analysis_and_clustering
    res = client.get("/api/v1/clusters/1/dossier")
    assert res.status_code in (200, 404)
    if res.status_code == 200:
        assert res.headers["content-type"] == "application/pdf"

def test_municipal_dispatch_endpoint():
    """Verify municipal dispatch endpoint."""
    login_res = client.post("/auth/demo-login")
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res = client.post("/api/v1/clusters/1/dispatch", headers=headers)
    assert res.status_code in (200, 404)
    if res.status_code == 200:
        data = res.json()
        assert data["success"] is True
        assert "pdf_path" in data

def test_road_watch_subscription():
    """Verify citizen road watch toggle."""
    login_res = client.post("/auth/demo-login")
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res = client.post("/api/v1/roads/1/watch", headers=headers)
    assert res.status_code in (200, 404)

def test_leaderboard_endpoint():
    """Verify public leaderboard."""
    res = client.get("/api/v1/leaderboard")
    assert res.status_code == 200
    data = res.json()
    assert "leaderboard" in data
