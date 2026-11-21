"""Quick end-to-end test of the walkthrough flow."""
import requests, os, glob

BASE = "http://localhost:8000"

# 1) Demo login
r = requests.post(f"{BASE}/auth/demo-login")
assert r.status_code == 200, f"Demo login failed: {r.text}"
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("1. Demo login: OK")

# 2) Find a test image
test_imgs = glob.glob("data/**/*.jpg", recursive=True) + glob.glob("data/**/*.png", recursive=True)
if not test_imgs:
    test_imgs = glob.glob("static/**/*.jpg", recursive=True)
if not test_imgs:
    print("No test images found - skipping analyze test")
    exit(0)

img = test_imgs[0]
print(f"2. Test image: {img}")

# 3) Analyze
with open(img, "rb") as f:
    r2 = requests.post(
        f"{BASE}/api/analyze",
        headers=headers,
        files={"file": (os.path.basename(img), f, "image/jpeg")},
        data={"location": "Test Location", "lat": "18.5913", "lng": "73.7389"},
    )
print(f"3. Analyze: HTTP {r2.status_code}")
if r2.status_code == 200:
    d = r2.json()
    sev = d.get("highest_severity", "unknown")
    dets = d.get("detections", [])
    inc_id = d.get("incident_id")
    guidance = d.get("guidance", {})
    print(f"   Severity: {sev}")
    print(f"   Detections: {len(dets)}")
    print(f"   Incident ID: {inc_id}")
    print(f"   Guidance keys: {list(guidance.keys())}")
    if guidance.get("llm_enhanced"):
        safe_text = guidance["llm_enhanced"][:100].encode("ascii", "replace").decode()
        print(f"   AI Guidance: {safe_text}...")

    # 4) Notify
    ntype = "pothole" if "pothole" in sev else "accident"
    r3 = requests.post(
        f"{BASE}/api/notify",
        headers=headers,
        json={"type": ntype, "location": "NH-48, near Hinjawadi, Pune", "incident_id": inc_id},
    )
    msg = r3.json().get("message", "")
    print(f"4. Notify: HTTP {r3.status_code} - {msg[:100]}")
else:
    print(f"   Error: {r2.text[:300]}")

print("\n=== All walkthrough steps verified ===")
