# CityPulse — Product Implementation Roadmap
## From Prototype → Government-Grade Smart City Platform

> **Goal**: Transform CityPulse into a pitchable, deployable product for the Indian Government (MoRTH, Smart Cities Mission, NCRB). Estimated timeline: **6 months** in 4 phases.

---

## Recommended Full Tech Stack

### Frontend (Web & Mobile)
| Component | Technology | Reason |
|---|---|---|
| Web App Framework | **Next.js 14** (React) | SSR, API routes, dashboard support |
| Mobile App | **React Native + Expo** | Single codebase for iOS + Android |
| UI Library | **shadcn/ui + Tailwind CSS** | Fast, professional UI components |
| Maps | **Google Maps SDK / Mapbox GL** | Heatmaps, pin clusters, offline tiles |
| Charts | **Recharts / Victory Native** | Analytics dashboards |
| State Management | **Zustand** | Lightweight, works across web + mobile |
| Offline Storage (Mobile) | **MMKV + SQLite (via Expo-SQLite)** | Fast local caching |

### Backend
| Component | Technology | Reason |
|---|---|---|
| API Framework | **FastAPI** (existing — keep) | Async, high performance, OpenAPI |
| Auth | **Supabase Auth / Auth0** | JWT, OAuth2, RBAC out of the box |
| Database | **PostgreSQL** (via Supabase or managed RDS) | Relational, audit logs, geospatial (PostGIS) |
| ORM | **SQLAlchemy + Alembic** | Migrations, type-safe queries |
| Cache | **Redis** | API response caching, rate limiting |
| Task Queue | **Celery + Redis** | Async video processing, notification dispatch |
| File Storage | **Google Cloud Storage / S3** | Store annotated images, videos, reports |
| Notifications | **Firebase Cloud Messaging (push) + MSG91 (SMS) + SendGrid (email)** | Real delivery |
| Real-time | **WebSockets (FastAPI) + Socket.io** | Live camera feed results, dashboard live updates |

### ML / AI
| Component | Technology | Reason |
|---|---|---|
| Detection Model | **YOLOv8m / YOLOv8l** (upgrade from n) | Higher accuracy; still real-time capable |
| On-Device Inference | **YOLO11n → ONNX + Core ML / TFLite** | Mobile offline detection |
| AI Guidance | **Gemini 2.0 Flash / Gemini Pro** (existing) | Keep, expand prompts |
| Model Registry | **MLflow** | Track experiments, version models |
| Training Infra | **Google Vertex AI / Colab Pro+** | GPU-accelerated retraining |
| Video Streaming | **FFmpeg + OpenCV** (existing), add **GStreamer for RTSP** | Live camera ingestion |
| Language | **IndicTrans2 / Google Translate API** | Multilingual guidance |

### DevOps / Infrastructure
| Component | Technology | Reason |
|---|---|---|
| Container | **Docker** (existing) | Keep |
| Orchestration | **Google Kubernetes Engine (GKE)** | Auto-scaling for city-scale load |
| CI/CD | **GitHub Actions** | Automated test + deploy pipeline |
| Monitoring | **Prometheus + Grafana + Google Cloud Monitoring** | Uptime, inference latency, error rates |
| Secrets | **Google Secret Manager** | API keys, credentials |
| IaC | **Terraform** | Reproducible cloud infrastructure |

---

## Phase 1 — Foundation & Security (Weeks 1–4)

> Build the security, auth, and data persistence layer. No new features, just hardening what exists.

### 1.1 — Database Setup
- Set up **PostgreSQL** (Supabase free tier for dev, managed Cloud SQL for prod)
- Create schema:
  ```sql
  -- incidents table
  id, image_url, video_url, severity, lat, lng, address_text,
  detection_results (jsonb), guidance (jsonb), created_at, reported_by, status

  -- users table  
  id, email, phone, role (ENUM: citizen, officer, admin), district, created_at

  -- notifications table
  id, incident_id, channel, recipient, status, sent_at
  ```
- Integrate **SQLAlchemy** into FastAPI; replace all in-memory processing with DB writes
- Add **PostGIS** extension for geospatial queries

### 1.2 — Authentication & RBAC
- Integrate **Supabase Auth** (email + phone OTP for India)
- Define roles:
  - `citizen` — can submit reports via mobile app
  - `field_officer` — traffic police, can update incident status
  - `admin` — city/district administrator, full dashboard access
- Protect all API endpoints with JWT middleware
- Implement API rate limiting via **Redis** (SlowAPI for FastAPI)

### 1.3 — Security Hardening
- Remove `allow_origins=["*"]` → whitelist specific domains
- Add file upload validation (size limits, magic byte check, not just extension)
- Implement input sanitization on all endpoints
- Add request logging with correlation IDs

### 1.4 — Real Notification System
- Integrate **MSG91** (India's leading SMS gateway, officially used by government apps like DigiLocker)
  ```python
  # SMS via MSG91
  POST https://api.msg91.com/api/v5/flow/
  {"flow_id": "...", "mobiles": "91XXXXXXXXXX", "var1": "incident_type", "var2": "location"}
  ```
- Integrate **Firebase Cloud Messaging** for push notifications (for mobile app)
- Integrate **SendGrid / AWS SES** for email (with annotated image attachment)
- Wire `/api/notify` to actually deliver alerts

**Deliverable**: Secure, data-persisting API with real alerting. Same frontend, but now production-safe.

---

## Phase 2 — Web Product (Weeks 5–10)

> Rebuild the frontend as a proper Next.js application with an admin dashboard.

### 2.1 — Rewrite Frontend (Next.js)
- Migrate `frontend/` from plain HTML to **Next.js 14 App Router**
- Pages:
  - `/` — Landing + Hero (citizen-facing)
  - `/analyze` — Image/video upload (existing feature, improved UX)
  - `/map` — Incident heatmap (Google Maps)
  - `/dashboard` — Admin operations center
  - `/reports` — Incident history, filters, export
  - `/login` — Auth (Supabase)

### 2.2 — Incident Map
- Display all incidents as pins on a **Google Maps** view
- Color-coded by severity (green → red)
- Cluster pins at country/district level
- Click pin → view incident details, annotated image, actions taken
- **Heatmap mode** for pothole density (key for municipal pitch)
- Implementation: `@react-google-maps/api` or `react-map-gl` (Mapbox)

### 2.3 — Admin Dashboard
- Real-time incident count by severity (WebSocket updates)
- Charts: incidents over time, by district, by type (potholes vs accidents)
- Incident table: filter by date, severity, district, status
- Assign incidents to field officers
- One-click escalation to National Highway Authority, State PWD, etc.

### 2.4 — Incident Management Workflow
- State machine for each incident: `detected → notified → acknowledged → resolved → closed`
- Field officers update status from mobile app
- Auto-close after 30 days of no action (configurable)
- SLA tracking: "Pothole reported 5 days ago — no action"

### 2.5 — PDF Report Generation
- Generate incident PDF reports using **WeasyPrint / ReportLab**
- Include: annotated image, timestamp, GPS, severity, guidance given, actions taken
- Exportable for government records and court submissions

**Deliverable**: Full web application with admin dashboard, map view, and incident lifecycle management.

---

## Phase 3 — Mobile Application (Weeks 8–16, overlapping)

> Build iOS + Android app using React Native + Expo. This is the highest-priority feature for field use.

### 3.1 — Citizen App Features
- **Camera capture** → instant AI analysis (image sent to API)
- **GPS auto-tag** — capture latitude/longitude on photo
- **One-tap report** — submit detected incident with location
- **Status tracking** — "Your pothole report: Under Review"
- **Emergency guidance** — displayed immediately after detection
- **Push notifications** — receive updates on submitted reports

### 3.2 — Field Officer App Features
- View assigned incidents on map
- Update status in-field (phone camera re-analysis option)
- Offline mode: queue reports if no network, sync when connected
- Contact emergency services directly from within the app

### 3.3 — On-Device ML (Offline Mode)
- Export YOLOv8n to **ONNX → TFLite (Android) / Core ML (iOS)**
- Run inference locally on-device for areas with poor connectivity (rural India)
- Sync results to server when connection is restored
- Key for credibility in government pitch — doesn't rely on cloud connectivity

```bash
# Export to ONNX for mobile
yolo export model=best.pt format=onnx simplify=True
# Then convert: onnx → tflite (tf2onnx), onnx → coreml (coremltools)
```

### 3.4 — Multi-Language Support
- Implement **i18n** in both web (next-i18next) and mobile (i18n-js)
- Priority languages: **Hindi, Tamil, Telugu, Bengali, Marathi, Kannada**
- Use **IndicTrans2** (AI4Bharat open-source) for emergency guidance translation
- Emergency guidance text translated and stored in DB by language code

**Deliverable**: Published app on Play Store + App Store (internal testing track for pitch demo).

---

## Phase 4 — Smart City Integration (Weeks 14–24)

> Add the features that differentiate CityPulse as a city-scale infrastructure product.

### 4.1 — Live CCTV / Traffic Camera Integration
- Build an ingestion service that connects to IP cameras via **RTSP streams**
  ```python
  # Stream processor
  cap = cv2.VideoCapture("rtsp://camera_ip:554/stream")
  # Run YOLO on frames, push detections to WebSocket endpoint
  ```
- Frame sampling at 2–5 FPS for efficiency
- Auto-alert if high-severity incident detected with no upload needed
- **WebSocket endpoint** in FastAPI to push live results to dashboard

### 4.2 — Improved ML Model
- Expand dataset with **Indian road condition images**:
  - Sources: iNaturalist India, Roboflow community, state highway CCTV partnerships
  - Add classes: `road_debris`, `waterlogging`, `stray_animal`, `construction_hazard`
- Upgrade from **YOLOv8n → YOLOv8m or YOLOv8l** for better accuracy
- Target: **mAP@50 > 0.75**, precision > 0.80
- Train with Indian road augmentation: monsoon lighting, dust haze, night conditions
- **MLflow** integration for experiment tracking and model registry

### 4.3 — Government API Integrations
- **VAAHAN API** (Ministry of Road Transport) — vehicle registration lookup from plate number
- **RTMS (Road Traffic Management System)** — feed detections into national traffic database
- **DigiLocker / e-Seva** — authentication via government digital ID
- **NIC Cloud / MeghRaj** — option for on-premise government cloud deployment

### 4.4 — Analytics & BI Layer
- **City Risk Score**: Score each district 0–100 based on incident density, severity, resolution speed
- **Predictive Hotspot Mapping**: ML model predicting highest-risk zones by time of day and season
- **Pothole Repair ROI Calculator**: Estimate money saved in vehicle damage by fixing potholes
- Export to **Power BI / Google Looker Studio** for government reporting

### 4.5 — Scalability & Reliability
- Deploy on **GKE with HPA** (Horizontal Pod Autoscaler) — scale ML pods on CPU/GPU usage
- **Multi-region**: Primary in Mumbai, failover in Delhi (NIC MeghRaj zones)
- **SLA targets**: 99.9% uptime, <3s detection latency, <1s map load
- Load testing with **Locust** before pitch

---

## Implementation Priority Matrix

| Feature | Impact | Effort | Priority |
|---|---|---|---|
| Real Notifications (SMS + Email) | 🔴 Critical | Low | ⭐ P0 |
| Authentication + RBAC | 🔴 Critical | Medium | ⭐ P0 |
| Database + Incident Persistence | 🔴 Critical | Medium | ⭐ P0 |
| Mobile App (Citizen) | 🔴 Critical | High | ⭐ P0 |
| Incident Map (Web) | 🟠 High | Medium | P1 |
| Admin Dashboard | 🟠 High | High | P1 |
| On-Device ML (Offline) | 🟠 High | Medium | P1 |
| Multi-language (Hindi min.) | 🟠 High | Medium | P1 |
| Live CCTV Integration | 🟡 Medium | High | P2 |
| Model Improvement | 🟡 Medium | High | P2 |
| Government API Integration | 🟡 Medium | High | P2 |
| Predictive Analytics | 🟢 Nice to have | Very High | P3 |

---

## 6-Month Execution Timeline

```
Month 1:   Phase 1 — Security, Auth, DB, Real Notifications
Month 2:   Phase 2 (start) — Next.js frontend, Map view
Month 3:   Phase 2 (finish) — Admin dashboard, Reports
           Phase 3 (start) — Mobile app core
Month 4:   Phase 3 (finish) — Mobile app, Multilingual, On-device ML
Month 5:   Phase 4 (start) — CCTV integration, Model improvement
Month 6:   Phase 4 (finish) — Gov't API integrations, Load testing, Pitch prep
```

---

## Minimum Viable Pitch (MVP for 3 Months)

If 6 months is too long, here's what to build in **3 months** to have a credible government demo:

1. ✅ PostgreSQL database + incident storage
2. ✅ Real SMS/email notifications (MSG91 + SendGrid)
3. ✅ Basic auth (Supabase, email+phone OTP)
4. ✅ Next.js web app with incident map (Google Maps)
5. ✅ Admin dashboard (incident table + stats cards)
6. ✅ React Native mobile app (citizen report flow, camera, GPS, offline queue)
7. ✅ Hindi language support in mobile app
8. ✅ PDF report generation

> [!IMPORTANT]
> Even the 3-month MVP must have **real notifications, geolocation, and a mobile app** to be taken seriously. These are non-negotiable for any government technology pitch in India.

---

## Cost Estimate (Monthly, at Pitch Scale)

| Service | Provider | Est. Cost/Month |
|---|---|---|
| Backend (GKE 2-node) | Google Cloud | ~$150 |
| Database (Cloud SQL Postgres) | Google Cloud | ~$50 |
| Storage (Images/Videos) | Google Cloud Storage | ~$20 |
| SMS Notifications | MSG91 | ~₹1/SMS, budget ₹5,000 |
| Gemini AI (guidance) | Google AI | ~$30 (estimate) |
| Maps API | Google | ~$50 |
| **Total** | | **~$330/month + ₹5,000** |

For government deployment at scale (citywide), estimated **₹15–25 Lakhs/year** for 1M+ citizens across a metro.

---

*CityPulse — Building Safer Cities with AI*
