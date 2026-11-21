# CityPulse

**Report road hazards. Get them fixed.**

CityPulse is an installable web app (PWA) for citizens. Snap a photo of a pothole, open manhole or
waterlogged road. CityPulse checks the photo with an on-server AI model, tags the exact spot, merges
duplicate reports, writes a ready-to-send complaint for your city, and keeps everyone who follows the
hazard updated until neighbours confirm it has been fixed.

It works in English, हिन्दी and मराठी. Pune is the pilot city.

## What people can do

**Citizens**
- **Report in under a minute.** Take a photo, which is compressed on the phone. GPS tags the spot, and
  placing a pin on the map is required when GPS is off or imprecise. The AI suggests "pothole" and a
  severity, and the person confirms. Reports of the same hazard within 20 m are merged as a "+1".
- **Send it to the city.** One tap shares the complaint text and PDF on WhatsApp or by email, or
  opens the city's grievance portal (PMC CARE for Pune). The citizen can then record the official
  complaint number. Citizens send complaints themselves; CityPulse never emails officials on anyone's
  behalf.
- **Track it.** Each hazard has a public page with its full history, and push notifications arrive
  when its status changes. After 7 days the app reminds the reporter to file it, and after 21 days
  without action it suggests escalating (Aaple Sarkar / CPGRAMS).
- **Confirm repairs.** When someone says a hazard is fixed, followers are asked to confirm.
  2 "fixed" votes close it; 2 "still there" votes reopen it with higher priority.
- **Map and city scorecard.** A public map of active and fixed hazards, plus ward-level stats
  (active count, median days to fix, % fixed within 30 days) and the worst roads.
- **Emergency mode.** One-tap 112 / 108 / 100 calls, a "share my location" button and plain first-aid
  steps. It works offline and uses no AI.
- **Works on bad networks.** Reports made offline are stored on the phone and sent automatically
  later, without creating duplicates.

**Officers and admins** (`/admin`): a work queue sorted by priority, status updates (a repair photo is
required for "fixed"), moderation of unverified or flagged reports (approve, reject, merge duplicates),
a dashboard, a heatmap, and user roles.

**Privacy:** the public only ever sees a crop around the hazard, with photo metadata stripped.
Reporter identities are never shown. Users can delete their account at any time; their reports stay,
anonymised.

## Architecture

```
Browser (PWA: React + Vite, service worker, IndexedDB offline queue)
        │  same origin, cookie session + CSRF header
        ▼
FastAPI (one container on Render)
  ├─ ONNX Runtime pothole detector (YOLOv8n, CPU)
  ├─ PostgreSQL (Neon) via SQLAlchemy + Alembic
  ├─ Cloudflare R2: public crops/thumbnails + private full photos (signed URLs)
  ├─ Web Push (VAPID) · SMTP for password resets only
  └─ /api/internal/jobs/daily ← GitHub Actions cron (reminders, auto-confirm, cleanup)
```

| Path | What's there |
|---|---|
| `backend/` | API (`routes/`), domain rules (`domain/status.py` is the hazard lifecycle), services, `ml/detector.py` |
| `backend/alembic/` | database migrations |
| `app/` | the PWA (citizen app + officer console), `e2e/` Playwright tests |
| `tests/` | backend tests (pytest) |
| `ml_scripts/` | training/evaluation scripts (need `requirements-ml.txt`) |
| `docs/` | design, implementation plan, archived early docs |

## Run it locally

Requirements: Python 3.11 and Node 22.

```bash
python -m venv venv && venv/Scripts/activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                                 # defaults are fine for local use (SQLite, local media)

cd app && npm ci && npm run build && cd ..
alembic upgrade head
python -m backend.cli create-admin --email you@example.com
uvicorn backend.main:create_app --factory --reload   # http://localhost:8000
```

For frontend work with hot reload, run `npm run dev` in `app/` (http://localhost:5173). It proxies API
calls to port 8000.

Camera, GPS and push only work on `localhost` or HTTPS. To try it on a phone, use a tunnel such as
`cloudflared tunnel --url http://localhost:8000`.

## Tests

```bash
pytest -q                          # backend: ~190 tests (SQLite)
ruff check backend tests
cd app
npm test                           # unit/component tests (Vitest)
npm run lint
npm run build && npx playwright test   # end-to-end: real backend, real model, temporary DB
```

CI (`.github/workflows/ci.yml`) runs all of these, applies migrations to PostgreSQL 16, and checks
that the Docker image starts and passes its health check.

## Deploy (free / cheap tier)

1. **Database:** create a [Neon](https://neon.tech) project and copy its connection string.
2. **Media:** create two [Cloudflare R2](https://developers.cloudflare.com/r2/) buckets,
   `citypulse-public` (enable public access, e.g. its r2.dev URL) and `citypulse-private` (keep it
   private). Then create an API token with read/write access to both.
3. **Push keys:** run `npx web-push generate-vapid-keys`.
4. **Email (password reset):** any SMTP provider. Brevo's free tier works with a single verified sender.
5. **Render:** New → Blueprint → pick this repo (`render.yaml`), then fill in the variables marked
   `sync: false`. The container runs `alembic upgrade head` on every start. The free plan sleeps after
   15 minutes idle (the first request then takes about a minute, and the app shows "connecting to the
   server…"). Starter at $7/month stays awake.
6. **First admin:** in the Render shell, run `python -m backend.cli create-admin --email you@example.com`.
7. **Daily job:** in GitHub → Settings → Secrets → Actions, add `APP_URL` and `JOB_TOKEN` (the same
   value as on Render). `.github/workflows/daily-job.yml` runs at 06:00 IST.
8. **Monitoring (optional):** set `SENTRY_DSN`, and point an uptime monitor at `/health`.

In production (`APP_ENV=production`) the app refuses to start with a default secret key, a non-Postgres
database, or local media storage.

### Configuration

All settings are environment variables. See `.env.example` for the full list.

| Variable | Purpose |
|---|---|
| `APP_ENV` | `dev`, `test` or `production` |
| `APP_URL` | public URL, used in share links, emails and push |
| `SECRET_KEY` | signs sessions and reset links (≥32 random chars) |
| `DATABASE_URL` | PostgreSQL URL (`postgres://` is accepted) |
| `STORAGE_BACKEND`, `R2_*` | media storage |
| `VAPID_*` | web push; push is disabled when unset |
| `SMTP_*` | password-reset email; links are logged when unset |
| `JOB_TOKEN` | protects the daily job endpoint |
| `GEOCODER`, `LOCATIONIQ_KEY` | reverse geocoding (Nominatim by default, LocationIQ if a key is set) |
| `SENTRY_DSN`, `LOG_JSON` | observability |

### City contact details

Where complaints go is configured in `backend/cities.py`. The Pune contacts (PMC CARE portal, email,
WhatsApp, helpline, Aaple Sarkar escalation) came from public sources and are marked `verified=False`,
so the app shows a "please double-check" note. Confirm them on the official PMC website and set
`verified=True`. Adding a city means adding one `CityConfig` entry.

## The model

`backend/ml/models/pothole-v1.onnx` is a YOLOv8n trained on about 6,400 road images (6 classes; only
`pothole` is used). Overall mAP@50 is 0.57. That's why the AI only *suggests*: people confirm the type
and severity, reports the model doesn't recognise go to a moderator instead of being rejected, and
neighbours verify repairs. Retraining scripts are in `ml_scripts/`. After exporting a new ONNX file,
regenerate the reference fixture with `ml_scripts/make_detector_fixture.py`.
