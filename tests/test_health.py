def test_health_reports_db_and_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "db": "ok", "model": "ok"}


def test_health_503_when_model_missing(app, client):
    app.state.detector = None
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["model"] == "error"
