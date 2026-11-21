from urllib.parse import parse_qs, urlparse

from backend.services.storage import LocalStorage, R2Storage, new_key


def test_new_key_layout():
    k = new_key("reports", "jpg")
    assert k.startswith("reports/") and k.endswith(".jpg") and len(k) > 30


def test_local_storage_roundtrip(tmp_path):
    s = LocalStorage(str(tmp_path), secret="s3cret")
    s.put("reports/a.jpg", b"abc", "image/jpeg", public=True)
    assert s.read("reports/a.jpg", public=True) == b"abc"
    assert s.url("reports/a.jpg", public=True) == "/media/public/reports/a.jpg"
    s.delete("reports/a.jpg")
    assert s.read("reports/a.jpg", public=True) is None


def test_local_private_url_is_signed(tmp_path):
    s = LocalStorage(str(tmp_path), secret="s3cret")
    s.put("full/b.jpg", b"xyz", "image/jpeg", public=False)
    url = s.url("full/b.jpg", public=False)
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).path == "/media/private/full/b.jpg"
    assert s.verify_signature("full/b.jpg", q["exp"][0], q["sig"][0])
    assert not s.verify_signature("full/other.jpg", q["exp"][0], q["sig"][0])


def test_private_media_route_checks_signature(app, client):
    storage = app.state.storage
    storage.put("full/c.jpg", b"secret-bytes", "image/jpeg", public=False)
    good = storage.url("full/c.jpg", public=False)
    assert client.get(good).content == b"secret-bytes"
    assert client.get("/media/private/full/c.jpg?exp=9999999999&sig=bad").status_code == 403


def test_public_media_route_serves_file(app, client):
    app.state.storage.put("reports/d.jpg", b"public-bytes", "image/jpeg", public=True)
    r = client.get("/media/public/reports/d.jpg")
    assert r.status_code == 200 and r.content == b"public-bytes"


def test_path_traversal_rejected(app, client):
    assert client.get("/media/public/..%2F..%2Fsecrets.txt").status_code in (400, 404)


def test_r2_public_url_uses_base_url():
    s = R2Storage(account_id="acc", access_key_id="k", secret_access_key="s", public_bucket="pub",
                  private_bucket="priv", public_base_url="https://media.example.com/")
    assert s.url("reports/a.jpg", public=True) == "https://media.example.com/reports/a.jpg"
    private = s.url("full/a.jpg", public=False)
    assert "priv" in private and "X-Amz-Signature" in private


def test_s3_compatible_endpoint_override_for_google_cloud_storage():
    s = R2Storage(account_id="", access_key_id="k", secret_access_key="s", public_bucket="pub",
                  private_bucket="priv", public_base_url="https://storage.googleapis.com/pub",
                  endpoint_url="https://storage.googleapis.com")
    assert s.client.meta.endpoint_url == "https://storage.googleapis.com"
    assert s.url("reports/a.jpg", public=True) == "https://storage.googleapis.com/pub/reports/a.jpg"
    assert "X-Amz-Signature" in s.url("full/a.jpg", public=False)


def test_delete_tolerates_key_missing_from_one_bucket():
    # Found against real GCS: unlike R2, it answers NoSuchKey for the bucket that never held the object.
    from botocore.stub import Stubber

    s = R2Storage(account_id="", access_key_id="k", secret_access_key="s", public_bucket="pub",
                  private_bucket="priv", public_base_url="https://x", endpoint_url="https://storage.googleapis.com")
    with Stubber(s.client) as stub:
        stub.add_response("delete_object", {}, {"Bucket": "pub", "Key": "a.jpg"})
        stub.add_client_error("delete_object", service_error_code="NoSuchKey", http_status_code=404,
                              expected_params={"Bucket": "priv", "Key": "a.jpg"})
        s.delete("a.jpg")
        stub.assert_no_pending_responses()
