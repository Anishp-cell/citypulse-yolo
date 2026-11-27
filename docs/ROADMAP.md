# CityPulse: what's built and what's next

Last updated: 2026-11-26. This is the shared plan for everyone working on the repo. Update it in the same
pull request that changes the plan.

- [1. What CityPulse does today](#1-what-citypulse-does-today)
- [2. Current state](#2-current-state)
- [3. Roadmap](#3-roadmap)
- [4. The projects](#4-the-projects)
- [5. Where projects overlap, and the rules that keep them apart](#5-where-projects-overlap-and-the-rules-that-keep-them-apart)
- [6. Contributor setup](#6-contributor-setup)

## 1. What CityPulse does today

CityPulse lets citizens report road hazards (potholes, open manholes, waterlogging, debris) from their
phone. The app checks the photo with an AI model, groups duplicate reports, and helps citizens push the
city to fix the hazard and confirm the fix.

### How a report flows through the system

1. **Photo and location.** The phone shrinks the photo to 1600 px and about 1 MB
   (`app/src/lib/compressImage.js`) and sends it with the GPS position to `POST /api/reports/analyze`.
2. **Validation.** The server checks the file by its content: 10 MB limit, JPEG/PNG/WebP only, and a
   guard against oversized "image bombs". It fixes the orientation, strips EXIF metadata (including the
   GPS tags) and resizes to at most 2048 px (`backend/services/images.py`).
3. **AI check.** A YOLOv8 model in ONNX format runs on the CPU and looks for potholes
   (`backend/ml/detector.py`). At most 2 photos are analysed at once. The size of the largest pothole
   in the frame gives a suggested severity.
4. **Privacy crop.** The public photo is cropped around the detected pothole so bystanders and number
   plates are mostly left out. The full photo is stored privately; the crop and a thumbnail are public.
5. **Address.** The GPS point is turned into a road, ward and city (Nominatim or LocationIQ, cached in
   the database). The response also lists existing hazards within 30 m.
6. **Submit.** `POST /api/reports` with the type, severity and note. If a hazard of the same type
   already exists within 20 m, the report joins it. Otherwise a new hazard is created. An
   AI-confirmed pothole goes live as `open`; anything else waits in `pending_review` for a moderator.
   An idempotency key makes retries safe. The reporter automatically follows the hazard.
7. **Priority.** Each hazard gets a score from 0 to 100 from its type, severity, number of reports,
   days open and disputes (`backend/domain/priority.py`).
8. **No signal.** Reports made offline wait in IndexedDB on the phone and are sent in order when the
   app is next open (`app/src/lib/offlineQueue.js`).

### Hazard lifecycle

`pending_review → open → filed → acknowledged → in_progress → fixed_unverified → fixed` (or `rejected`).
The rules for who may make each change live in one place: `backend/domain/status.py`.

- **Filed:** the citizen gets a ready-made complaint (text, WhatsApp message or PDF, in English, Hindi
  or Marathi), sends it to the city themselves, then records the city's ticket number.
- **Fixed:** when someone marks a hazard fixed, other citizens vote. Two votes confirm or reopen it; one
  vote is enough if an officer marked it fixed.
- **Flags:** anyone can flag a report for abuse; moderators review flags.

### Everything else

| Area | What exists |
|---|---|
| Accounts | Email and password (bcrypt), 30-day httpOnly session cookie, CSRF header check, password reset by email, account deletion |
| Notifications | Web push to followers on every status change. Email is used only for password resets. |
| Officer and admin console | Queue sorted by priority, moderation of pending and flagged reports, merge duplicates, remove photos, user roles, stats, map |
| Daily job | GitHub Actions calls `/api/internal/jobs/daily` at 06:00 IST: refresh priorities, reminders at 7 days (file) and 21 days (escalate), auto-confirm fixes after 14 days, delete unused uploads after 24 h |
| Public pages | Map, a page per hazard, city scorecard, share links |
| Languages | English, Hindi, Marathi (react-i18next), language switcher, CI check for missing translation keys |
| Operations | `/health`, request IDs, JSON logs, optional Sentry, CI (lint, tests on SQLite and PostgreSQL, migrations, end-to-end, Docker) |

### Architecture

```
Phone (React PWA, service worker, IndexedDB queue)
   │  HTTPS, session cookie
   ▼
One container: FastAPI + built PWA  (Dockerfile; runs `alembic upgrade head`, then 1 uvicorn worker)
   ├─ ONNX Runtime pothole detector (CPU)
   ├─ PostgreSQL (Neon)       ← SQLite locally
   ├─ Object storage (R2 or GCS, S3 API)  ← local disk locally
   ├─ Geocoder (Nominatim / LocationIQ)
   └─ Web push (VAPID), SMTP for password reset
GitHub Actions: CI on every PR, daily maintenance cron
```

## 2. Current state

| Item | State |
|---|---|
| Production site | Reported as up. The deploy target is Render (`render.yaml`); a move to Cloud Run (`citypulse-66027`) is under discussion. |
| CI on `main` | Red since the v1 merge. Fixed by PR #3 (Git LFS checkout + one test mock), which is waiting to be merged. |
| Daily job | Failing every day since 2026-11-22: the `APP_URL` and `JOB_TOKEN` repository secrets are empty. |
| Model | YOLOv8n, mAP@50 about 0.57, pothole class only. |
| Old training data | Not on `main`. The old datasets, runs and weights are on the `CPYC` branch (restore with `git checkout origin/CPYC -- data runs_citypulse`). Don't delete that branch. |

## 3. Roadmap

### Phase 0: make the base safe (now)

| # | Task | Owner | Status |
|---|---|---|---|
| 0.1 | Get CI green (PR #3) and require green CI on `main` (branch protection) | Maintainer | PR open |
| 0.2 | Set the `APP_URL` and `JOB_TOKEN` secrets so the daily job runs | Maintainer | To do |
| 0.3 | Security headers: CSP, HSTS, X-Frame-Options, Referrer-Policy, Permissions-Policy | Code | In progress |
| 0.4 | Logins that can be cancelled: logging out everywhere and changing the password end all sessions | Code | In progress |
| 0.5 | Set `LOCATIONIQ_KEY` in production (the public Nominatim server isn't allowed for production traffic) | Maintainer | To do |
| 0.6 | Check backups: Neon restore window, versioning on the photo bucket, one test restore | Maintainer | To do |
| 0.7 | Decide Render or Cloud Run (projects 2 and 7 depend on it) | Maintainer | To decide |

Moved out of Phase 0: email verification at sign-up goes with project 3 (spam), and running migrations
as a separate release step goes with project 7 (deploy pipeline).

### Phase 1: shared foundations (before the parallel projects)

Small pull requests, merged in this order:

1. **Analysis pipeline refactor.** Move the body of `analyze()` (`backend/routes/reports.py`) into a
   pipeline module (for example `backend/services/analysis.py`) built from stages: decode → detect →
   blur → crop → hash → store. No behaviour change. Projects 1, 3 and 4 then add or swap a stage
   instead of editing the same function.
2. **Shared rate limiter.** Replace the in-memory limiter (`backend/ratelimit.py`) with one backed by
   Postgres (or Redis), keeping the same `rate_limit()` interface. Needed before running more than one
   instance, and reused by projects 3 and 8.
3. **Async upload contract.** Agree the response of `/api/reports/analyze` once analysis runs in the
   background: a job id, a status to poll, and the `upload_id` when done. Projects 2 and 5 both build
   on it.

### Phase 2: parallel projects

| Track | Order | Good for |
|---|---|---|
| ML | 1 → 4 → 3 | ML experience |
| Backend and DevOps | 2 → 8, with 7 alongside | Backend and DevOps experience |
| Frontend | 5 + 6 | Frontend experience |

Project 2 (background worker) has to land before projects 3 and 4 add models: the instance has 512 MB
of memory, and more models inside the web request would run it out.

## 4. The projects

Each project lists what already exists, so work starts from the gap.

### 1. Model quality: a real ML pipeline
- **Goal:** mAP@50 above 0.70 on a fixed held-out set, with a model card in `docs/`.
- **Exists:** training script (`ml_scripts/train_yolo.py`), `requirements-ml.txt`, ONNX inference that
  reads class names from the model file, and `ml_scripts/make_detector_fixture.py` to regenerate the
  test reference.
- **To do:** more data (RDD2022, Roboflow Universe, confirmed app reports), augmentation, an ablation
  study (image size, classes, augmentation), per-class precision and recall, ONNX export, quantization,
  latency benchmark. Then a PR that replaces `backend/ml/models/pothole-v1.onnx`.
- **Watch out:** adding classes also means updating `HAZARD_TYPES` and the `POTHOLE`-only checks in
  `backend/routes/reports.py` (today only potholes can be AI-verified). Use Python 3.11 or Colab:
  `torch==2.3.1` doesn't install on 3.13.

### 2. Async inference and scale
- **Goal:** uploads return quickly; a Locust load test (50 concurrent uploads) shows p95 latency before
  and after.
- **Exists:** on-phone compression; a cap of 2 concurrent analyses.
- **To do:** `analysis_jobs` table, worker entry point (Cloud Tasks or Cloud Run Jobs once 0.7 is
  decided), status polling in the app, load test write-up.

### 3. Duplicate and spam detection
- **Exists:** 20 m same-type merging and per-user rate limits.
- **To do:** perceptual photo hashing (imagehash), embedding similarity, per-device limits, a trust
  score per user, email verification at sign-up, irrelevant or NSFW image filter.
- **Watch out:** PostGIS queries would break the SQLite dev and test setup. Either keep the existing
  bounding-box query and add a `(lat, lng)` index, or drop SQLite from the tests first. One embedding
  model can do both duplicate matching and irrelevant-image filtering.

### 4. Privacy by design: automatic blurring
- **Exists:** EXIF stripping, the privacy crop, account deletion (`DELETE /api/auth/me`), deletion of
  unused uploads after 24 h.
- **To do:** face and number-plate blurring (ONNX, reusing the `Detector` interface) as a pipeline
  stage before the public crop, with a measured miss rate on a labelled sample. Personal data export.
  Retention rules in the daily job.

### 5. Offline-first PWA
- **Exists:** IndexedDB queue with idempotency keys, offline app shell, cached map tiles and photos.
- **To do:** background sync so the queue sends while the app is closed, resumable uploads, Playwright
  offline tests with throttling, a Lighthouse PWA score of 90 or more.

### 6. Internationalisation (fold into project 5)
- **Exists:** react-i18next, English, Hindi and Marathi, a language switcher and the CI key check.
- **To do:** right-to-left layout (Urdu is the natural candidate), km/mi distances.

### 7. Delivery pipeline and observability
- **Exists:** CI, Sentry hook, JSON logs, request IDs.
- **To do:** deploy on merge with Workload Identity Federation (no keys in GitHub), a staging service,
  a Neon branch per pull request, migrations as a release step before deploy, OpenTelemetry tracing, an
  uptime SLO dashboard, a runbook in `docs/`.
- **Watch out:** setting up Workload Identity Federation needs an IAM admin to create the pool and the
  service account; "Cloud Run Developer" alone isn't enough. A Neon branch per PR needs a Neon API key
  as a GitHub secret.

### 8. Public API and open data
- **To do:** versioned REST API with OpenAPI docs, API keys, pagination, quotas, GeoJSON and CSV
  export, an embeddable map widget, webhooks, contract tests.
- **Watch out:** quotas reuse the Phase 1 rate limiter, and webhooks reuse project 2's job queue.
  Public exports must ship after project 4's blurring, or open data leaks faces and plates.

## 5. Where projects overlap, and the rules that keep them apart

| Shared code | Projects | Rule |
|---|---|---|
| `analyze()` in `backend/routes/reports.py` | 1, 2, 3, 4 | Phase 1 pipeline refactor first; each project then adds a stage |
| Memory on the instance | 2, 3, 4 | Project 2 lands before new models are added |
| `backend/ratelimit.py` | 2, 3, 7, 8 | One shared limiter (Phase 1); per-device limits and API quotas build on it |
| Alembic migrations | 2, 3, 4, 8 | One migration per PR; rebase and renumber before merge so there is a single head |
| Upload API and the offline queue | 2, 5 | Agree the async response shape (Phase 1) before either starts |
| ONNX models and `Detector` | 1, 3, 4 | New models use the `Detector` interface |
| Deploy target | 2, 7 | Decide Render or Cloud Run first (0.7) |
| Public photos | 4, 8 | Open data ships after blurring |

## 6. Contributor setup

You need Git LFS, Python 3.11 (what CI uses; 3.12 also works) and Node 22. No API keys are needed:
locally the app uses SQLite and stores photos on disk.

```bash
git lfs install
git clone https://github.com/Anishp-cell/citypulse-yolo.git
cd citypulse-yolo
python3.11 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                  # Windows: copy .env.example .env  (keep APP_ENV=dev)
pytest -q && ruff check backend tests
cd app && npm ci && npm run lint && npm test && npm run build && cd ..
uvicorn backend.main:create_app --factory --reload   # http://localhost:8000
```

- To try the app without signing up, start the server with `ENABLE_DEMO=true` and use demo login.
- To see the officer console, run `python -m backend.cli create-admin --email you@example.com`.
- For frontend hot reload, run `npm run dev` in `app/` (http://localhost:5173).

**Working rules:** one branch per change (`git checkout -b feat/<name>`), a pull request into `main`,
and CI must be green before merging. Database changes go through an Alembic migration (one per PR).
