# CityPulse v1 — Production Launch Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (recommended: Native, see bottom) or superpowers:subagent-driven-development. Use superpowers:test-driven-development inside every task, superpowers:systematic-debugging on any failure, superpowers:requesting-code-review at the end of each milestone, superpowers:verification-before-completion before claiming done, `security-review` before launch, and `run` to drive the app. Steps use `- [ ]` checkboxes.

**Goal:** Ship CityPulse as a production-ready, installable web app (PWA) where citizens report potholes and other road hazards, send the complaint to their city themselves, track it until it's fixed, and confirm the repair, with emergency first-aid guidance as a secondary mode.

**Architecture:** One FastAPI service serves both the JSON API and the built React PWA (same origin, cookie auth, no CORS). Pothole detection runs on CPU with **onnxruntime** using the existing `best.onnx`, with no torch at runtime. Postgres (Neon) holds the data, images go to Cloudflare R2, and the whole thing deploys as one Docker service on Render.

**Spec:** `docs/superpowers/specs/2026-11-21-citypulse-v1-design.md`

**Tech stack:** Python 3.11, FastAPI, SQLAlchemy 2 + Alembic, Postgres, onnxruntime + OpenCV-headless + Pillow, ReportLab, pywebpush, slowapi, boto3 (R2), sentry-sdk · React 19 + Vite + vite-plugin-pwa, react-router 7, TanStack Query, react-leaflet + marker clustering, i18next (en/hi/mr), idb-keyval, browser-image-compression · pytest, Vitest, Playwright, GitHub Actions.

---

## Context

The project stalled about 6 months ago at a demo stage: three separate frontends, a backend that crashes (`IncidentStatus.notified` was removed but `admin.py` still uses it), fake mobile login, alerts that report success when nothing was sent, demo code that sends **fake locations** to real authorities, a Docker image that's missing dependencies, and a **privilege escalation** (anyone can register as `admin`, because `UserCreate.role` is accepted from the client). The user wants it finished properly and then closed out: a reliable app that real people would use, not a feature list.

**Decisions the user made:**
1. **Potholes first.** Citizen reporting and city accountability is the product. Accident/emergency is a secondary mode.
2. **Web only (PWA).** Expo `mobile/` and vanilla `frontend/` are retired. The React `app/` becomes the single frontend (citizen + officer/admin).
3. **Free/cheap hosting:** Render (Docker), Neon Postgres, Cloudflare R2, all free tiers. Render Starter ($7/mo) recommended at launch to avoid cold starts.
4. **Citizens send their own complaints.** The app generates the complaint pack (PDF + text + portal link) and the citizen shares it via WhatsApp/email or files it on the city portal. The server never emails government addresses. Twilio and SendGrid are removed.

**Calls I made (with reasons, all can be overridden):**
- **No AI in emergency mode.** The accident classes are weak (overall mAP50 0.57), and wrong medical advice is a liability. Users pick their situation ("someone is injured / only vehicles damaged / not sure") and get vetted static first-aid steps plus one-tap 112/108. Gemini is removed (its SDK `google-generativeai` is deprecated, and hallucination is a risk here).
- **AI assists, it doesn't decide.** The model suggests "pothole + severity" and the user confirms. If the model finds nothing, the user can still submit; the hazard goes to `pending_review` instead of being rejected outright.
- **More hazard types, chosen manually:** open manhole, waterlogging, debris/obstruction, other. These kill people in Indian monsoons. They need no ML and always go to `pending_review`.
- **Photos are images only, no video.** Video upload is removed, since it was heavy on CPU and adds nothing for potholes.
- **Privacy by cropping:** the public sees a crop around the detected pothole (plus a margin) with EXIF stripped. The full photo is visible only to the reporter and officers. This avoids face/plate exposure without needing a blur model.
- **Leaderboard and karma currency are dropped.** They're replaced with a personal "Impact" summary (reports, fixes confirmed). Roads and wards get scorecards, because public accountability is the real motivation for using the app.
- **Marathi added** (Pune pilot) next to English and Hindi.
- **Server-side geocoding** with caching (Nominatim with a proper User-Agent; switches to LocationIQ if `LOCATIONIQ_KEY` is set). This gives a reliable road and ward and respects tile/geocode usage policies.
- **Map tiles** from CARTO basemaps (free, attribution required) instead of `tile.openstreetmap.org`, whose usage policy forbids production apps.
- **Password reset email** via generic SMTP (Brevo free tier works with a single verified sender; no domain needed).

---

## Product spec (what a user experiences)

**Citizen: report (target under 60 seconds)**
1. Home → big **Report a hazard** button → camera opens (`<input type=file accept=image/* capture=environment>`; the gallery is also allowed).
2. The photo is compressed on the device (≤1600px JPEG, ~0.5–1 MB, orientation fixed, HEIC converted). GPS is captured with its accuracy shown. A mini-map lets the user drag the pin. If accuracy is worse than 100 m or GPS is denied, placing the pin is **required**; the app never invents coordinates.
3. Upload → AI result: "Looks like a pothole (87%), severity: Medium", with editable type and severity and an optional note.
4. **Nearby check:** "3 people already reported a hazard 12 m away. Is this the same one?" [Yes, add my +1] [No, it's new].
5. Done screen → **Send to PMC** sheet: share the PDF + text via WhatsApp/email (Web Share API with the file; fallback is download + `mailto:` + copy text), and an **Open PMC grievance portal** link. Then "I filed it, here's my complaint number" (optional), which moves the hazard to `filed`.
6. If offline, the report is queued in IndexedDB and sent automatically when back online, without creating duplicates (idempotency key).

**Citizen: follow up**
- **My reports:** each hazard with its status, days open and a timeline (reported → filed #PMC123 → acknowledged → fixed → confirmed).
- Push notifications (Web Push) when a followed hazard changes status, when someone says it's fixed ("Can you confirm?"), and as reminders: day 7 "Not filed yet? File it with PMC"; day 21 after filing with no progress, "Escalate" (an escalation pack addressed to the next level from the city config).
- **Verify:** "Is it fixed?" [Yes, with an optional photo] [No, still there]. Two "fixed" votes from different users confirm it; two "not fixed" votes reopen it with a priority boost.

**Public**
- **Map:** clustered pins for active hazards, filter chips (Active / Fixed / All), tap to open a bottom sheet with a link to the hazard page.
- **Hazard page `/h/:id`:** a public, shareable link with Open Graph tags (the server injects them so WhatsApp previews show a photo and "Open 47 days on FC Road").
- **City scorecard `/city`:** per ward, the number of active hazards, the median days to fix (last 90 days) and the % fixed within 30 days; plus the 10 worst roads.

**Emergency `/emergency`** (works offline): big **Call 112** / **108 Ambulance** buttons, triage choice → step-by-step first aid, and **Share my location** (Web Share with a Google Maps link). Nothing is stored.

**Officer/admin (`/admin`, same app, lazy-loaded)**
- **Work queue:** filter by ward and status, sorted by priority. Hazard detail with the full photo and timeline. Status change with a note and an optional "fixed" proof photo.
- **Moderation:** `pending_review` items and abuse flags → approve / reject (with a reason) / merge a duplicate into another hazard.
- **Analytics:** the existing Recharts page, rewired to hazards. **Users:** list and role changes (admin only).
- First admin created with `python -m backend.cli create-admin --email ...`.

**Account:** email + password, forgot/reset password, language, notification toggle, **delete account** (reports are anonymized and stay public), Privacy and Terms pages.

### Hazard status machine (single source of truth: `backend/domain/status.py`)

| From → To | Who | Trigger |
|---|---|---|
| (new) → `pending_review` | system | manual type, or AI confidence < 0.25 |
| (new) → `open` | system | AI-detected pothole ≥ 0.25 |
| `pending_review` → `open` / `rejected` | officer/admin | moderation |
| `open` → `filed` | any follower | submits official ticket id |
| `open`/`filed` → `acknowledged` → `in_progress` | officer/admin | queue |
| any active → `fixed_unverified` | officer (proof photo), or a citizen "fixed" vote | |
| `fixed_unverified` → `fixed` | system | 2 distinct "fixed" votes, or officer proof + 1 vote, or 14 days with no dispute (daily job) |
| `fixed_unverified` → `open` | system | 2 distinct "not fixed" votes (priority ×1.5) |
| any → `rejected` | admin | reason required (`duplicate`, `not_a_hazard`, `abuse`) |

"Active" = `open, filed, acknowledged, in_progress`. `rejected`/`pending_review` are never public. Every transition writes a `status_events` row.

---

## Global constraints

- Runtime deps contain **no torch/ultralytics**; those move to `requirements-ml.txt` (training only).
- All config comes from `backend/settings.py` (pydantic-settings). `APP_ENV=production` makes the app **refuse to start** if `SECRET_KEY` is the default, if `DATABASE_URL` isn't Postgres, or if R2 is unconfigured. The silent SQLite fallback only exists when `APP_ENV=dev`/`test`.
- Clients can never set `role`. Only an admin changes roles.
- Every unsafe endpoint (POST/PATCH/DELETE) requires the session cookie (httpOnly, Secure in prod, SameSite=Lax) plus the header `X-Requested-With: citypulse` (CSRF guard).
- Rate limits (slowapi, per user, falling back to IP): analyze 20/hour, reports 30/day, auth 10/min, verify 30/day, flag 10/day.
- Uploads: JPEG/PNG/WebP decoded by Pillow (magic bytes, not extension), ≤10 MB, EXIF stripped, re-encoded as JPEG.
- Dedupe radius: **20 m**, same hazard type, active or `pending_review` hazards only. A user's +1 counts at most once per hazard (`report_count` = distinct reporters).
- No PII in public payloads: reporter names and emails never appear in hazard/map/PDF responses.
- Strings: everything user-visible goes through i18n with keys in `en.json`, `hi.json` and `mr.json` (the CI check fails on missing keys).
- The PDF must **not** claim to be an official government document. Title: "Citizen Road Hazard Complaint (prepared with CityPulse)".
- Third-party policies: geocoding sends a `User-Agent: CityPulse/1.0 (+<APP_URL>)` header and makes ≤1 request/s to Nominatim. Map attribution is shown.

## Review focus (failure modes that will bite real users; each has a test in its owning task)

1. **GPS denied, slow or inaccurate (desktop, indoors):** the flow must still finish via manual pin placement and must never submit 0,0 or demo coordinates. → Task 4.3 Playwright `report_without_gps_requires_pin`.
2. **Flaky or offline network and Render cold start mid-upload:** retry with backoff, show a "server waking up" message, queue offline, and never create a duplicate report on retry (`Idempotency-Key`). → Task 2.2 `test_report_idempotent_retry`, Task 4.6 `offlineQueue.test.ts`.
3. **HEIC, huge or rotated phone photos:** converted client-side; the server rejects undecodable files with 415 and a translatable error code. → Task 1.5 `test_rejects_non_image_bytes`, Task 4.3 compression unit test.
4. **The same user reporting the same pothole twice:** the count must not inflate, and the second attempt returns the existing hazard. → Task 2.2 `test_same_user_plus_one_counted_once`.
5. **The model misses a real pothole:** the user can still submit (goes to `pending_review`), with no dead end. → Task 2.2 `test_low_confidence_goes_to_pending_review`.

---

## Target file structure

```
backend/
  main.py                 app factory, routers, SPA + OG-tag serving
  settings.py             pydantic-settings, prod guards
  db.py                   engine/session (replaces database.py)
  models.py               rewritten schema (below)
  cli.py                  create-admin, seed-demo
  security.py             bcrypt, PyJWT session + reset tokens
  deps.py                 current_user, require_roles, csrf guard (replaces dependencies.py)
  ratelimit.py
  domain/status.py        transition table + can_transition()/apply_transition()
  domain/priority.py      priority score (from spatial_service.calculate_priority_score)
  domain/geo.py           haversine (from spatial_service) + bbox helpers
  ml/detector.py          Detector protocol, OnnxPotholeDetector, FakeDetector
  services/images.py      validate, strip EXIF, crop, thumbnail
  services/storage.py     Storage protocol: LocalStorage, R2Storage
  services/geocode.py     reverse_geocode() with cache
  services/complaint.py   complaint text + PDF (reuses government/pdf_generator.py)
  services/push.py        web push fan-out
  services/mailer.py      SMTP reset emails
  services/jobs.py        daily job
  cities.py               per-city config (Pune first) + generic fallback
  routes/auth.py, reports.py, hazards.py, me.py, scorecard.py, push.py, admin.py, internal.py
  alembic/ + alembic.ini
app/                      single PWA (citizen + admin)
tests/                    pytest (conftest with temp DB, FakeDetector, tmp LocalStorage)
.github/workflows/ci.yml, daily-job.yml
Dockerfile (multi-stage), render.yaml, requirements.txt, requirements-ml.txt
```

### Data model (Alembic `0001_initial`, fresh schema; old demo data is discarded)

- `users`: id, email (lowercased, unique), hashed_password, full_name, role {citizen, officer, admin}, city, ward, language, is_active, deleted_at, created_at
- `hazards`: id, type {pothole, open_manhole, waterlogging, debris, other}, severity {low, medium, high}, status, lat, lng, address, road, ward, city, priority_score, report_count, ai_verified bool, cover_image_key, official_ticket_id, official_portal, first_reported_at, last_reported_at, fixed_at; indexes on (lat, lng) and (city, status)
- `reports`: id, hazard_id, user_id (nullable after deletion), image_key, public_image_key, thumb_key, ai_label, ai_confidence, ai_boxes JSON, location_source {gps, manual}, gps_accuracy_m, note, idempotency_key (unique per user), created_at
- `uploads`: id uuid, user_id, image keys, ai JSON, lat, lng, accuracy, place JSON, created_at, consumed_at (purged after 24 h)
- `status_events`: id, hazard_id, from_status, to_status, actor_id, note, image_key, created_at
- `verifications`: id, hazard_id, user_id, is_fixed, image_key, created_at; unique (hazard_id, user_id, round), where round = number of times the hazard has entered `fixed_unverified`
- `follows`: (user_id, hazard_id) PK; created automatically for reporters
- `flags`: id, hazard_id, user_id, reason, resolved, created_at
- `push_subscriptions`: id, user_id, endpoint unique, p256dh, auth, created_at
- `geocode_cache`: key "lat4,lng4" PK, payload JSON, created_at

Roads and wards are **derived** (group by `road`/`ward` on hazards). There is no `road_segments` table and no stored health score. Road health = `max(0, 100 − Σ severity_weight)` over its active hazards (low 5 / medium 10 / high 20).

---

## Milestone 0: Baseline and cleanup

### Task 0.1: Save current work and branch
- [ ] Create a branch `v1-launch` (superpowers:using-git-worktrees is optional). Commit the current uncommitted WIP as is: "wip: snapshot before v1 rework".
- [ ] Commit `mobile/` to a branch `archive/expo-mobile` (so it can be recovered), then remove `mobile/` and `frontend/` from `v1-launch`.
- [ ] Delete `backend/service/inference.py` (Streamlit), `test_db.py`, `test_e2e.py`, `citypulse.db`, `static/` demo files, and root `yolov8m.pt`, `yolov8n.pt`, `yolo11n.pt`.
- [ ] Stop tracking ML artifacts with `git rm --cached -r data/ runs/ dataset.zip` plus every `.pt` (the files stay on disk). Keep **only** `runs_citypulse/yolov8n_pothole_vbest2/weights/best.onnx`, copied to `backend/ml/models/pothole-v1.onnx` (tracked; ~12 MB).
- [ ] `.gitignore`: add `.env`, `*.db`, `static/`, `media/`, `data/`, `runs*/`, `*.pt`, `*.zip`, `node_modules/`, `app/dist/`, `.agents/`, `.claude/`, `skills-lock.json`, `.pytest_cache/`.
- [ ] **Rotate secrets:** the local `.env` holds real Gemini/SendGrid/Twilio keys. Tell the user to revoke the SendGrid, Twilio and Gemini keys (no longer used). The `.env` was never committed (verified: `git log --all -- .env` is empty).
- [ ] Commit: "chore: retire mobile and vanilla frontends, untrack ML artifacts".

## Milestone 1: Backend foundation

### Task 1.1: Settings, DB, Alembic, app factory
**Files:** create `backend/settings.py`, `backend/db.py`, `alembic.ini`, `backend/alembic/`; rewrite `backend/main.py` as `create_app(settings) -> FastAPI`; delete `backend/database.py`.
- `Settings` fields: `app_env`, `app_url`, `secret_key`, `database_url`, `storage_backend {local, r2}`, `r2_*`, `media_dir`, `vapid_public_key`, `vapid_private_key`, `vapid_subject`, `smtp_*`, `job_token`, `sentry_dsn`, `locationiq_key`, `enable_demo` (default False).
- Tests `tests/test_settings.py`: `test_prod_rejects_default_secret`, `test_prod_rejects_sqlite`, and `test_dev_allows_sqlite`, each asserting that `Settings(...).validate_for_env()` raises or passes.
- `GET /health` returns `{status, db: "ok", model: "ok"}`, or 503 if either fails. Test: `test_health_reports_db_and_model`.
- `tests/conftest.py`: an `app` fixture with a temp-file SQLite DB, `create_all`, `FakeDetector`, and `LocalStorage(tmp_path)`; a `client` fixture; and `citizen`, `officer` and `admin` fixtures that log in through the API.

### Task 1.2: Models and initial migration
**Files:** rewrite `backend/models.py` per the data model; `alembic revision --autogenerate -m initial`; `backend/domain/status.py`.
- `status.py` produces `HazardStatus` enum, `ACTIVE_STATUSES`, `can_transition(from_, to, role, via) -> bool`, and `apply_transition(db, hazard, to, actor, note=None, image_key=None) -> StatusEvent` (raises `InvalidTransition`).
- Tests `tests/test_status.py`: a parametrized case for every row of the status table (allowed), plus `test_citizen_cannot_acknowledge`, `test_rejected_is_terminal` and `test_transition_writes_event`.
- CI also runs `alembic upgrade head` against Postgres (Task 6.2).

### Task 1.3: Auth hardening
**Files:** rewrite `backend/routes/auth.py`, `backend/security.py` (switch to PyJWT), `backend/deps.py`, `backend/ratelimit.py`, `backend/services/mailer.py`, `backend/cli.py`.
- Endpoints: `POST /api/auth/register {email,password,full_name}`, `POST /api/auth/login {email,password}`, `POST /api/auth/logout`, `GET/PATCH/DELETE /api/auth/me`, `POST /api/auth/forgot-password`, `POST /api/auth/reset-password {token,new_password}`. The session is a JWT in the cookie `cp_session` (30-day expiry).
- Reset token: JWT `{sub, purpose:"reset", pwd_fp: hashed_password[-10:]}`, 30 min, single-use because the fingerprint changes once the password does.
- `demo-login` exists only when `enable_demo=True`.
- Tests: `test_register_ignores_role_field` (POST with `role:"admin"` → `me.role == "citizen"`), `test_password_min_8`, `test_email_case_insensitive`, `test_login_sets_httponly_cookie`, `test_unsafe_request_without_csrf_header_403`, `test_reset_token_single_use`, `test_forgot_password_always_200`, `test_delete_account_anonymizes_reports`, `test_auth_rate_limited` (11th login/min → 429), `test_cli_create_admin`.

### Task 1.4: ONNX detector
**Files:** `backend/ml/detector.py`, `tests/test_detector.py`, `tests/fixtures/pothole.jpg` (copy of `data/pothole.jpeg`), `tests/fixtures/pothole_expected.json`.
- Produces: `@dataclass Detection(label: str, confidence: float, box: tuple[int,int,int,int])`, `class Detector(Protocol): def detect(self, img_bgr: np.ndarray) -> list[Detection]`, `OnnxPotholeDetector(model_path, imgsz=512, conf=0.25, iou=0.5)`, and `FakeDetector(detections)`.
- Approach: letterbox to 512, BGR→RGB, /255, NCHW; the output `(1, 4+6, N)` is transposed, filtered by class score, run through `cv2.dnn.NMSBoxes` and scaled back. Labels come from the class list in `data/merged_yolo_dataset/data.yaml`.
- `pothole_expected.json` is generated **once** with ultralytics `best.pt` on the fixture (a dev script `ml_scripts/make_detector_fixture.py`).
- Tests: `test_onnx_matches_ultralytics_reference` (same labels, IoU ≥ 0.9, confidence ±0.05), `test_blank_image_no_detections`, `test_output_shape_contract`.
- Helper `suggest_severity(dets, img_shape) -> Literal["low","medium","high"]`: largest pothole box area ratio <2% → low, 2–8% → medium, >8% → high. Tests for each boundary.

### Task 1.5: Images, storage, geocoding
**Files:** `backend/services/images.py`, `backend/services/storage.py`, `backend/services/geocode.py`, `backend/cities.py`.
- `process_upload(raw: bytes) -> ProcessedImage(full_jpeg, width, height)` raises `UnsupportedImage` and `ImageTooLarge`. `make_public_crop(full_jpeg, boxes, margin=0.35, min_side=320) -> bytes` falls back to a centre crop of 60% when there are no boxes. `make_thumb(jpeg, 256) -> bytes`.
- `Storage` protocol: `put(key, data, content_type, public: bool)`, `url(key, public: bool) -> str` (an R2 public-bucket URL, or a presigned URL valid for 1 h for private objects), `delete(key)`. `LocalStorage` serves from `/media/...`, and private files require a session cookie.
- `reverse_geocode(db, lat, lng) -> Place(road, ward, city, display)`, cached under key `round(lat,4),round(lng,4)`; ward = `suburb` or `city_district`. On provider failure it returns `Place(display="lat, lng")` and doesn't raise.
- `cities.py`: `CityConfig(name, bbox, grievance_portal_url, grievance_email, escalation_email, helpline)` with Pune filled in (**the implementer verifies the current PMC grievance portal URL and email on the official site before committing**) and `generic()` as a fallback.
- Tests: `test_rejects_non_image_bytes`, `test_strips_exif_gps`, `test_heic_named_jpeg_rejected_cleanly`, `test_crop_contains_box`, `test_local_storage_roundtrip`, `test_geocode_cached` (provider called once for two nearby calls, mocked with `httpx.MockTransport`), `test_geocode_failure_falls_back`.

## Milestone 2: Hazard core API

### Task 2.1: Analyze step
**Files:** `backend/routes/reports.py`, `backend/domain/geo.py` (move `haversine_distance_meters` from `backend/service/spatial_service.py`).
- `POST /api/reports/analyze` (multipart `file`, `lat`, `lng`, `accuracy_m`, `location_source`) → `{upload_id, ai:{is_pothole, confidence, severity_suggestion, boxes}, place, nearby:[{id,type,status,distance_m,report_count,thumb_url}]}`. Nearby = active or pending hazards within 30 m (we display this; auto-merging uses 20 m).
- Tests: `test_analyze_returns_ai_and_place`, `test_analyze_lists_nearby_within_30m`, `test_analyze_requires_auth`, `test_analyze_rate_limited`, `test_lat_lng_out_of_range_422`.

### Task 2.2: Submit report (create or +1)
- `POST /api/reports {upload_id, hazard_type, severity, note?, attach_to_hazard_id?}`, header `Idempotency-Key` → `{hazard_id, created_new, report_id}`.
- Logic: an explicit attach must be ≤50 m away, otherwise 422. Without one, auto-merge with the nearest same-type hazard within 20 m. Otherwise create a new hazard with status `open` (pothole + AI ≥ 0.25) or `pending_review`. Store full, public-crop and thumbnail images; geocode; compute `domain/priority.compute_priority(hazard, now)` (ported from `calculate_priority_score`, with type weights pothole 25 / open_manhole 40 / waterlogging 20 / debris 15 / other 10); auto-follow.
- Tests: `test_creates_hazard_open_when_ai_confident`, `test_low_confidence_goes_to_pending_review`, `test_manual_type_goes_to_pending_review`, `test_auto_merge_within_20m_same_type`, `test_no_merge_different_type`, `test_same_user_plus_one_counted_once`, `test_report_idempotent_retry`, `test_upload_single_use`, `test_attach_too_far_422`.

### Task 2.3: Public hazard reads and OG pages
**Files:** `backend/routes/hazards.py`, `backend/routes/scorecard.py`, plus SPA serving in `main.py`.
- `GET /api/hazards?bbox=&status=active|fixed|all&type=` (≤500, lightweight fields). `GET /api/hazards/{id}` (public crop URL, timeline, counts, days_open, ticket, `is_following`, `my_vote`). `GET /api/scorecard?city=` → `{wards:[{ward, active, median_days_to_fix_90d, pct_fixed_30d}], worst_roads:[{road, ward, active, health}]}`.
- `GET /api/meta/statuses` → `[{value, active: bool, public: bool}]` from `HazardStatus` (consumed by the frontend contract test in Task 5.1).
- `GET /h/{id}` serves `index.html` with `og:title`, `og:description` and `og:image` filled in. Any other non-API path serves `index.html`.
- Tests: `test_bbox_filters`, `test_pending_and_rejected_hidden`, `test_public_payload_has_no_reporter_pii`, `test_full_image_url_only_for_reporter_and_officer`, `test_scorecard_median_days`, `test_og_tags_injected_and_escaped` (a hazard address containing `<script>` comes back escaped).

### Task 2.4: Citizen actions
- `POST/DELETE /api/hazards/{id}/follow`, `POST /api/hazards/{id}/filed {portal, ticket_id}`, `POST /api/hazards/{id}/verify` (multipart `is_fixed`, `photo?`), `POST /api/hazards/{id}/flag {reason}`, `GET /api/me/reports`, `GET /api/me/following`, `GET /api/me/impact`.
- Tests: `test_filed_sets_ticket_and_status`, `test_fixed_vote_moves_to_fixed_unverified`, `test_two_distinct_fixed_votes_confirm`, `test_same_user_cannot_vote_twice_per_round`, `test_two_not_fixed_votes_reopen_and_boost_priority`, `test_flag_rate_limited`.

### Task 2.5: Complaint pack
**Files:** `backend/services/complaint.py`; adapt `backend/service/government/pdf_generator.py` → `backend/services/complaint_pdf.py` (keep its ReportLab layout, change the title per Global Constraints, add the hazard page URL and a QR code via `reportlab.graphics.barcode.qr`); delete `government/base.py` and `email_pdf_adapter.py`.
- `GET /api/hazards/{id}/complaint?lang=` → `{subject, body, whatsapp_text, email_to, portal_url, pdf_url, escalation?}`. `GET /api/hazards/{id}/complaint.pdf` (public, cached by `hazard.updated_at`).
- Tests: `test_complaint_uses_city_config`, `test_generic_city_fallback`, `test_pdf_has_no_reporter_pii`, `test_complaint_text_translated_hi`.

### Task 2.6: Admin API
**Files:** rewrite `backend/routes/admin.py` (this also fixes the `IncidentStatus.notified` crash by replacing it).
- `GET /api/admin/queue?ward&status&sort=priority|age`, `POST /api/admin/hazards/{id}/status {to, note, photo?}`, `GET /api/admin/moderation`, `POST /api/admin/hazards/{id}/approve`, `POST .../reject {reason}`, `POST .../merge {into_id}` (moves reports, follows and votes, then rejects as `duplicate`), `GET /api/admin/stats` (counts per status, a 30-day trend, a heatmap of points), `GET /api/admin/users`, `PATCH /api/admin/users/{id}/role` (admin only).
- Tests: `test_citizen_forbidden_admin_routes`, `test_officer_cannot_change_roles`, `test_merge_moves_reports_and_recounts`, `test_officer_fixed_requires_photo`, `test_stats_counts`.

## Milestone 3: Notifications and scheduled jobs

### Task 3.1: Web Push
**Files:** `backend/services/push.py`, `backend/routes/push.py`, plus hooks in `domain/status.apply_transition`.
- `GET /api/push/vapid-public-key`, `POST /api/push/subscribe`, `DELETE /api/push/subscribe`. `notify_followers(db, hazard, event, exclude_user_id)` sends a localized payload `{title, body, url:"/h/{id}"}` and deletes subscriptions that answer 404/410. Sending happens in FastAPI `BackgroundTasks`, so requests never block on push.
- Tests (pywebpush mocked): `test_status_change_pushes_followers_not_actor`, `test_gone_subscription_deleted`, `test_fixed_unverified_asks_followers_to_confirm`.

### Task 3.2: Daily job
**Files:** `backend/services/jobs.py`, `backend/routes/internal.py`, `.github/workflows/daily-job.yml` (cron `30 0 * * *` UTC = 06:00 IST; `curl -X POST $APP_URL/api/internal/jobs/daily -H "X-Job-Token: $JOB_TOKEN"` with retry, since it also wakes Render).
- `run_daily(db, now) -> JobReport`: recompute priorities; push a "file it" reminder at day 7 (unfiled); push an "escalate" reminder 21 days after filing with no acknowledgement; auto-confirm `fixed_unverified` older than 14 days with no dispute; purge unconsumed uploads older than 24 h, plus their storage objects. The job is idempotent: each reminder is sent once, tracked through `status_events` with `note="reminder:<kind>"`.
- Tests: `test_daily_wrong_token_401`, `test_reminder_sent_once`, `test_auto_confirm_after_14d`, `test_purges_stale_uploads`.

## Milestone 4: Citizen PWA (`app/`)

Patterns for all frontend tasks: data goes through TanStack Query hooks in `app/src/api/*.ts`. `api/client.ts` replaces `utils/api.js`: `credentials:'include'`, the CSRF header, retry with backoff on network errors or 502/503 (shows a "Server is waking up…" toast after 3 s), and errors mapped to i18n keys. Admin routes are loaded with `React.lazy`. Map pages use `MapView` (react-leaflet + `react-leaflet-cluster`, CARTO tiles, attribution).

### Task 4.1: App shell, PWA, i18n, auth pages
**Files:** `app/vite.config.js` (vite-plugin-pwa: manifest "CityPulse", theme color, icons 192/512/maskable, Workbox runtime cache for `/api/hazards*` network-first and static assets precached), `app/src/sw.ts` (push and notificationclick handlers), `app/src/i18n/{en,hi,mr}.json`, `app/src/hooks/useAuth.tsx` (cookie session via `/api/auth/me`), pages `Login`, `Register`, `ForgotPassword`, `ResetPassword`, `Settings`, `Privacy`, `Terms`, and `PublicLayout` (bottom nav: Home · Map · Report · My reports · Emergency).
- Tests (Vitest): `i18n.keys.test.ts` (hi and mr have every en key), `client.test.ts` (sends the CSRF header; retries 503 three times), `useAuth.test.tsx`.

### Task 4.2: Home and Map
**Files:** `pages/Home.tsx` (Report CTA, Emergency button, "N active hazards within 1 km" when location permission is already granted, my active reports strip), `pages/Map.tsx`, `components/HazardSheet.tsx`, `pages/Hazard.tsx` (`/h/:id`: crop photo, status badge, days open, timeline, Follow, Verify buttons when `fixed_unverified`, Share, Complaint sheet).
- Tests: `Hazard.test.tsx` (verify buttons appear only in `fixed_unverified`; no vote buttons after the user has voted).

### Task 4.3: Report flow
**Files:** `pages/report/ReportFlow.tsx` with steps `PhotoStep`, `LocationStep` (draggable pin; required when accuracy >100 m or GPS is denied), `ReviewStep` (AI result, type/severity pickers, note, nearby "same one?" list), `DoneStep`; `lib/compressImage.ts` (browser-image-compression, maxWidthOrHeight 1600, JPEG 0.8, orientation fixed).
- Tests: `compressImage.test.ts`; Playwright `report_happy_path` (fixture image, mocked geolocation → AI result → submit → Done), `report_without_gps_requires_pin` (geolocation denied → Next disabled until the pin is placed), `report_nearby_plus_one`.

### Task 4.4: Complaint share sheet and "I filed it"
**Files:** `components/ComplaintSheet.tsx`, `lib/share.ts` (`navigator.canShare({files})` → share the PDF + text; else download + `mailto:` + `https://wa.me/?text=` + copy), `components/FiledForm.tsx`.
- Tests: `share.test.ts` (both branches with a mocked navigator).

### Task 4.5: My reports, Impact, Scorecard, Emergency, push opt-in
**Files:** `pages/MyReports.tsx` (tabs: Reported / Following), `pages/City.tsx`, `pages/Emergency.tsx` (content from `i18n` keys `emergency.*`, ported from `ACCIDENT_GUIDANCE` in `backend/service/traffic_safety_service.py`, rewritten to standard first-aid guidance with 112 as the primary number and a disclaimer; **the user reviews this text before launch**), `components/PushPrompt.tsx` (asked after the first successful report, not on first load).
- Tests: `Emergency.test.tsx` (renders with no network; the `tel:112` link is present).

### Task 4.6: Offline queue
**Files:** `lib/offlineQueue.ts` (idb-keyval: `{id, idempotencyKey, blob, lat, lng, accuracy, type, severity, note, createdAt}`; `flush()` runs on `online` events and on app start, handling analyze→submit in sequence; it keeps items on 5xx/network errors and drops them on 4xx with a user-visible notice).
- Tests: `offlineQueue.test.ts` (enqueue on network error; flush sends the same Idempotency-Key; a 422 drops the item and surfaces an error).

## Milestone 5: Officer and admin console

### Task 5.1: Rework admin pages onto hazards
**Files:** rewrite `app/src/pages/admin/{Dashboard,Incidents→Queue,IncidentMap→AdminMap,Analytics}.tsx`, add `Moderation.tsx` and `Users.tsx`; update `AdminLayout`; delete `utils/severity.js` and replace it with `lib/status.ts` (labels and colours per `HazardStatus`, generated from the same list as the backend: `GET /api/meta/statuses` is used once at build time in a test to assert they match).
- Tests: `status.contract.test.ts` (frontend status list equals the backend's), Playwright `officer_marks_fixed_with_photo`, `admin_rejects_pending`.

## Milestone 6: Ship it

### Task 6.1: Container and deploy
**Files:** `Dockerfile` (stage 1 `node:22-alpine` runs `npm ci && npm run build` in `app/`; stage 2 `python:3.11-slim` installs `requirements.txt` and copies `backend/` + `app/dist`; runs `alembic upgrade head && uvicorn backend.main:app --host 0.0.0.0 --port $PORT --proxy-headers`), `render.yaml` (web service, health check `/health`, env vars listed), `requirements.txt` (exact pins), `requirements-ml.txt`, `.env.example` rewritten, `README.md` rewritten (what it is, local dev, deploy steps, env table).
- Verification: `docker build -t citypulse . && docker run --env-file .env.docker -p 8000:8000 citypulse`, then `curl localhost:8000/health` → 200 with `db: ok, model: ok`. The image should be under 700 MB and memory under 400 MB after 20 analyze calls (`docker stats`), which keeps it inside Render's free 512 MB.

### Task 6.2: CI and observability
**Files:** `.github/workflows/ci.yml` (jobs: backend = ruff + pytest on SQLite plus `alembic upgrade head` and pytest against a `postgres:16` service; frontend = eslint + vitest + build; e2e = Playwright against `docker compose`-free `uvicorn` with `APP_ENV=test` and the FakeDetector toggled by env), Sentry in `main.py` and `app/src/main.tsx` (only when a DSN is set), JSON logs with a request-id middleware.
- Verification: CI is green on the PR.

### Task 6.3: Launch checklist (run with the user)
- [ ] Neon project created → `DATABASE_URL`. R2 buckets `citypulse-public` (public) and `citypulse-private` created → keys. VAPID keys generated (`vapid --gen`). Brevo SMTP sender verified. Sentry DSNs. `JOB_TOKEN` and `SECRET_KEY` generated. GitHub secrets set.
- [ ] Render service created from `render.yaml`; first deploy is green; `python -m backend.cli create-admin`.
- [ ] Run the `security-review` skill on the branch and fix the findings.
- [ ] Phone smoke test (Android Chrome + iOS Safari "Add to Home Screen"): install, report, share to WhatsApp, receive push, verify a fix, use emergency mode offline (airplane mode).
- [ ] The user reviews the emergency text, the Privacy/Terms text, and the Pune portal URL and email.
- [ ] Merge `v1-launch` → `main` (superpowers:finishing-a-development-branch). Tag `v1.0.0`.

## Milestone 7 (gated, optional): Model check
- [ ] Measure pothole-class precision and recall on the val split with `ml_scripts/` (ultralytics, dev only). **Only if pothole mAP50 < 0.60:** retrain YOLO11n on the pothole-only subset at imgsz 640, export to ONNX, swap in `pothole-v2.onnx` behind a `MODEL_PATH` setting, and regenerate the detector fixture. Otherwise skip, since user confirmation already covers model errors.

---

## Reused from the existing code
- `haversine_distance_meters`, `calculate_priority_score`: `backend/service/spatial_service.py` → `domain/geo.py`, `domain/priority.py`
- `generate_grievance_pdf` layout: `backend/service/government/pdf_generator.py` → `services/complaint_pdf.py`
- `ACCIDENT_GUIDANCE` content: `backend/service/traffic_safety_service.py` → `i18n emergency.*`
- `save_upload_with_size_limit` streaming guard: `backend/main.py` → `services/images.py`
- React: `PublicLayout`/`AdminLayout`, CSS design tokens in `app/src/index.css`, `utils/geo.js` (`getCurrentPosition`), the Leaflet setup in `pages/admin/IncidentMap.jsx`, the Recharts setup in `Analytics.jsx`, and the i18n setup in `app/src/i18n/index.js`
- Everything else in `backend/service/` is deleted once its pieces are ported.

## End-to-end verification
1. `pytest -q` passes (SQLite) and `DATABASE_URL=postgres://… pytest -q` passes. `cd app && npm run lint && npx vitest run && npm run build` passes.
2. `npx playwright test` covers report happy path, no-GPS pin, nearby +1, officer fixed, and admin reject; all pass.
3. Local full stack: `uvicorn backend.main:app` with `APP_ENV=dev` and the real ONNX model. Use the `run` skill to: register → report `tests/fixtures/pothole.jpg` → see "pothole" → submit → open `/h/1` → generate the complaint PDF → log in as officer → mark fixed with a photo → as a second citizen vote "fixed" → status `fixed`, with a push received in the browser.
4. Docker: size and memory limits as in Task 6.1. On Render, `/health` is 200, the phone smoke checklist passes, and the daily-job workflow run returns 200 with a `JobReport`.

## Execution
- The first execution step saves this plan as `docs/superpowers/plans/2026-11-21-citypulse-v1.md` plus a short spec at `docs/superpowers/specs/2026-11-21-citypulse-v1-design.md`, committed.
- **Recommended: Native** (I implement milestone by milestone in this session, run `requesting-code-review` after each milestone, and a final whole-branch review). The tasks share many interfaces (status machine, storage, detector), so one implementer holding the context is cheaper and more consistent than a fresh subagent per task.
- Rough size: M0 + M1 ≈ 1 day, M2 ≈ 1.5 days, M3 ≈ 0.5, M4 ≈ 2, M5 ≈ 0.5, M6 ≈ 0.5 of agent time; the launch checklist needs about an hour of the user's time for the accounts.
