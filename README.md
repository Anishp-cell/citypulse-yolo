# CityPulse — AI-Powered Road Safety Platform

> Real-time accident & pothole detection · Severity-based emergency guidance · Automated authority alerts

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=flat-square)](https://ultralytics.com)
[![Gemini AI](https://img.shields.io/badge/Gemini-2.0_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)

---

## The Problem

Road accidents kill **1.35 million people every year** (WHO) and injure up to 50 million more. Infrastructure failures — potholes, deteriorating roads — cause thousands of additional preventable incidents. Emergency response is slow because:

- **No automated detection** — accidents are reported manually, minutes after they happen
- **No bystander guidance** — most people do not know what to do at the scene
- **No systematic infrastructure tracking** — road defect data is fragmented and unactionable

CityPulse addresses all three with a single AI-powered platform, deployable by any city government worldwide.

---

## What CityPulse Does

Upload an image or video of any road scene. In under 2 seconds:

1. **Detects** accidents and potholes using a custom-trained YOLOv8 model (6 severity classes)
2. **Generates** actionable first-aid guidance via Gemini 2.0 Flash AI
3. **Alerts** emergency services and municipal departments via real SMS (Twilio) and email (SendGrid)
4. **Logs** every incident with severity, location, and annotated imagery to a persistent database

---

## Architecture

```
User / Field Officer
       |  Upload image or video
       v
  FastAPI Backend  -->  YOLOv8n Model  -->  Annotated image + detections
       |                                          |
       v                                          v
  PostgreSQL DB                          Gemini 2.0 Flash
  (Incidents, Users,                     (Medical guidance)
   Notifications)
       |
       v
  Notification Service
  |-- SendGrid  ->  Email to authorities
  `-- Twilio    ->  SMS to emergency contact
```

### Detection Classes

| Class | Emergency Level | Response |
|---|---|---|
| `no_accident` | 0 | Safe — no action |
| `minor_accident` | 1 | Document, exchange info, monitor for symptoms |
| `moderate_accident` | 2 | Call emergency services, stabilise victim |
| `severe_accident` | 3 | Immediate dispatch, CPR, spinal care |
| `totaled_vehicle` | 4 | Multi-casualty protocol, fire check |
| `pothole` | — | Municipal repair request |

### Model Performance

| Metric | Value |
|---|---|
| Precision | 0.63 |
| Recall | 0.63 |
| mAP@50 | 0.57 |
| mAP@50-95 | 0.33 |
| Inference Speed (GPU) | ~25 FPS |
| Training Images | 6,422 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8n (Ultralytics) + PyTorch |
| AI Guidance | Google Gemini 2.0 Flash Lite |
| Backend API | FastAPI + SQLAlchemy + PostgreSQL |
| Authentication | JWT (python-jose) + bcrypt |
| Notifications | SendGrid (email) + Twilio (SMS) |
| Frontend | Vanilla HTML/CSS/JS (single-page app) |
| Deployment | Docker + Google Cloud Run |

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/register` | None | Create account |
| `POST` | `/auth/login` | None | Get JWT token |
| `GET` | `/auth/me` | JWT | Current user info |
| `POST` | `/api/analyze` | JWT | Analyze image |
| `POST` | `/api/analyze-video` | JWT | Analyze video |
| `POST` | `/api/notify` | JWT | Send real SMS + email alert |
| `GET` | `/api/incidents` | JWT | List incidents (RBAC) |
| `GET` | `/api/incidents/{id}` | JWT | Get incident detail |
| `PATCH` | `/api/incidents/{id}/status` | Officer/Admin | Update status |
| `GET` | `/health` | None | Health check |
| `GET` | `/docs` | None | Swagger UI |

---

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with PostGIS extension
- (Optional) CUDA GPU for faster inference

### 1. Clone & install

```bash
git clone https://github.com/Anishp-cell/citypulse-yolo.git
cd citypulse-yolo
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your values — see .env.example for all variables
```

Minimum required:

```env
SECRET_KEY=<generate: python -c "import secrets; print(secrets.token_hex(32))">
DATABASE_URL=postgresql://user:password@localhost:5432/citypulse
GEMINI_API_KEY=your-gemini-key
```

For real notifications, also set:

```env
SENDGRID_API_KEY=SG.xxxxx
SENDGRID_FROM_EMAIL=noreply@your-domain.com
ALERT_EMAIL=emergency@your-city.gov

TWILIO_ACCOUNT_SID=ACxxxxx
TWILIO_AUTH_TOKEN=xxxxx
TWILIO_FROM_PHONE=+1XXXXXXXXXX
ALERT_PHONE=+1XXXXXXXXXX
```

### 3. Run

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — you will be prompted to register or sign in.

### 4. Docker

```bash
docker build -t citypulse .
docker run -p 8000:8000 --env-file .env citypulse
```

---

## Security

- JWT authentication enforced on all `/api/*` endpoints
- Role-based access control: `citizen` / `officer` / `admin`
- CORS restricted to `ALLOWED_ORIGINS` env var (defaults to `localhost:8000`)
- File upload validation: type checking + 50 MB image / 500 MB video size limits
- All secrets via environment variables — never hardcoded
- Startup warning emitted if default `SECRET_KEY` is still in use

---

## Business Model

CityPulse is sold as a SaaS platform to city governments and smart city operators worldwide.

| Tier | Target | Pricing |
|---|---|---|
| City License — Standard | Cities under 1M population | $20,000–40,000/year |
| City License — Metro | Cities over 1M population | $80,000–150,000/year |
| Per-Camera License | CCTV network integrations | $100–200/camera/year |
| On-Premise Deployment | Government data centre requirements | Custom |

Citizen access is always free — funded by the government contract.

---

## Roadmap

- [ ] Mobile app (React Native) for citizen reporting and field officers
- [ ] Live CCTV / RTSP stream ingestion
- [ ] Multi-language support (auto-detect locale, translate guidance)
- [ ] Model retraining pipeline from live incident data (target mAP@50 > 0.75)
- [ ] Government API integrations (emergency dispatch, road management systems)
- [ ] Admin analytics dashboard with incident heatmaps

---

Built for safer cities everywhere.
