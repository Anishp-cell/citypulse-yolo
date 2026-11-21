def test_health_reports_db_and_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "db": "ok", "model": "ok"}


def test_health_503_when_model_missing(app, client):
    app.state.detector = None
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["model"] == "error"


def test_request_id_header(client):
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    assert len(client.get("/health").headers["x-request-id"]) >= 16


def test_health_caches_db_probe(app, client, monkeypatch):
    # Review I4: frequent platform health checks must not keep a scale-to-zero database awake.
    import backend.main as main

    calls = []
    monkeypatch.setattr(main, "can_connect", lambda engine: calls.append(1) or True)
    for _ in range(5):
        assert client.get("/health").status_code == 200
    assert len(calls) == 1
