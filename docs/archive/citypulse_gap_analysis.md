# CityPulse — Gap Analysis Report
## Current State vs. Government-Grade Product

> **As of:** March 2026 | **Audience:** Internal / Pitch Preparation

---

## 1. What Exists Today (Current State)

| Layer | What's Built |
|---|---|
| **ML Model** | Custom YOLOv8n trained on merged pothole + accident datasets; 6-class detection (no_accident, minor, moderate, severe, totaled_vehicle, pothole); mAP@50 = 0.57 |
| **Backend** | FastAPI app with `/api/analyze` (image), `/api/analyze-video` (video), `/api/notify` (simulated); Gemini 2.0 Flash Lite for AI guidance; Dockerized for Cloud Run |
| **Frontend** | Single-page HTML/CSS/JS web app; drag-and-drop upload; severity banner; incident timeline; emergency guidance rendering |
| **Notifications** | Simulated email stub — only prints to console; no actual delivery |
| **Deployment** | Dockerfile + Cloud Run config; no CI/CD pipeline |
| **Data** | ~6,400 training images, zipped dataset; model weights stored in repo |

---

## 2. Critical Gaps (Blockers for Government Pitch)

### 2.1 — No Mobile Application
- **Gap**: There is **no iOS or Android app**. Every field use-case (traffic police, ambulance crew, city inspectors, citizens) requires a mobile-first experience.
- **Impact**: ❌ Without mobile, the product cannot be used at the point of incident. It is not a real product — it's a demo tool.

### 2.2 — No Real-Time CCTV / Traffic Camera Integration
- **Gap**: The system only accepts manually uploaded images/videos. There is **no live stream ingestion** from traffic cameras (RTSP/HLS feeds), drones, or dashcams.
- **Impact**: ❌ The core value proposition for smart city infrastructure — autonomous monitoring — does not exist.

### 2.3 — Notification System is Simulated / Non-Functional
- **Gap**: [send_notification()](file:///d:/python/citypulse/backend/main.py#148-159) only prints to console. No actual emails, SMS, or calls to real emergency numbers (108, 100, civic helplines).
- **Impact**: ❌ Alerting emergency services is a core feature and is completely non-functional.

### 2.4 — No User Authentication or Role Management
- **Gap**: The web app has no login, no user roles, no session management, and `allow_origins=["*"]` (open CORS). Anyone can call any API endpoint.
- **Impact**: ❌ Unacceptable for a government system. Must have role-based access (Admin, Traffic Police, Field Officer, Citizen) and audit trails.

### 2.5 — No Persistent Database / Incident Management
- **Gap**: All analysis results are ephemeral — no data is saved anywhere. Uploaded files are stored temporarily with no cleanup or tracking.
- **Impact**: ❌ There is no incident history, no analytics dashboard, no accountability layer, no way to track if a pothole was reported and repaired.

### 2.6 — No Admin / Operations Dashboard
- **Gap**: There is no dashboard for city administrators or traffic authorities to monitor incident trends, review alerts, assign remediation tasks, or generate reports.
- **Impact**: ❌ Government bodies need an operational command center, not just a detection page.

### 2.7 — No GPS / Location Intelligence
- **Gap**: Location is a free-text field with no map integration, geocoding, or coordinate tracking. There is no map view of where incidents cluster.
- **Impact**: ❌ Road infrastructure management requires precise geolocation. Manual text location is insufficient.

### 2.8 — Model Quality is Prototype-Grade
- **Gap**: mAP@50 of 0.57 and precision/recall of 0.63 is below production standards. The model has not been tested on Indian road conditions specifically. No model versioning or A/B comparison pipeline exists.
- **Impact**: ⚠️ A model with 37%+ miss rate cannot be pitched to the government as a reliable life-safety product. Needs improvement and Indian-specific dataset enrichment.

### 2.9 — No Multi-Language / Accessibility Support
- **Gap**: The entire UI is in English only. Emergency guidance is in English. India has 22 official languages.
- **Impact**: ❌ For government adoption and field use, the app must support Hindi and regional languages like Tamil, Telugu, Bengali, Marathi, etc.

### 2.10 — No CI/CD, Testing, or SLA Infrastructure
- **Gap**: No automated tests (unit, integration, load), no CI/CD pipeline, no health monitoring, no rate limiting, no SLAs/SLOs defined.
- **Impact**: ❌ Government procurement requires documented reliability, test coverage, and operational run-books.

### 2.11 — No Data Privacy / Compliance Framework
- **Gap**: No data handling policy, no PDPB (India's Personal Data Protection Bill) compliance, no data retention strategy, no encrypted storage for incident media, no anonymization of vehicle/face data in captured images.
- **Impact**: ❌ Government systems must comply with Indian IT Act, PDPB, and potentially CERT-In guidelines.

### 2.12 — No API Security
- **Gap**: CORS is open (`*`), no API key management, no rate limiting, no input validation beyond file type, no authentication middleware.
- **Impact**: ❌ API is exploitable. Any deployment would be immediately vulnerable.

---

## 3. Major Feature Gaps (Required for a Full Product)

| Feature | Current State | Required |
|---|---|---|
| Mobile App | ❌ None | iOS + Android with camera, GPS, offline mode |
| Live Camera Feed | ❌ None | RTSP/WebRTC ingestion from traffic cameras |
| Real Notifications | ❌ Simulated | SMS (Twilio/MSG91), Email, Push Notification via 108/100 API |
| Auth & Roles | ❌ None | JWT auth, RBAC (Citizen, Officer, Admin) |
| Database | ❌ None | PostgreSQL/MySQL for incidents, users, reports |
| Admin Dashboard | ❌ None | React/Next.js ops dashboard with analytics |
| Map View | ❌ None | Google Maps / Mapbox heatmaps of incidents |
| Multi-language | ❌ None | Hindi + 5 regional language support |
| Report Generation | ❌ None | PDF/Excel export of incident reports |
| Model Improvement | ⚠️ mAP 0.57 | Target mAP@50 > 0.75, Indian road-specific dataset |
| Offline Mobile | ❌ None | On-device inference for poor network areas |
| Video Streaming | ❌ None | Live CCTV/drone feed processing |
| Audit/Logging | ❌ None | Full audit trail for all actions and detections |
| Data Compliance | ❌ None | PDPB, IT Act, face/plate anonymization |
| Model Versioning | ❌ None | MLflow/DVC tracking, A/B model comparison |

---

## 4. Operational Gaps

| Area | Gap |
|---|---|
| **Infrastructure** | No Kubernetes/auto-scaling for city-scale traffic load |
| **Monitoring** | No Prometheus/Grafana, no error alerting (PagerDuty/OpsGenie) |
| **Cost Model** | No pricing model for government licensing or per-API-call costing |
| **Data Pipeline** | No automated retraining loop from new incident data |
| **Disaster Recovery** | No backup strategy, no failover |
| **Documentation** | No API docs beyond `/docs`, no operational runbook, no data dictionary |

---

## 5. Pitch Readiness Summary

| Dimension | Status | Score |
|---|---|---|
| Core AI/ML Technology | ✅ Functional prototype | 4/10 |
| Web Application | ⚠️ Demo quality | 3/10 |
| Mobile Application | ❌ Non-existent | 0/10 |
| Real-time Monitoring | ❌ Non-existent | 0/10 |
| Notifications & Alerting | ❌ Simulated only | 1/10 |
| Security & Auth | ❌ Non-existent | 0/10 |
| Data & Analytics | ❌ Non-existent | 0/10 |
| Compliance & Privacy | ❌ Non-existent | 0/10 |
| Scalability | ⚠️ Docker only | 2/10 |
| **Overall Product Readiness** | **Proof of Concept** | **~1.5/10** |

> [!CAUTION]
> In its current state, CityPulse is a **technical proof of concept**, not a product. Presenting it to the Indian government without the missing components would likely result in immediate rejection. The core ML innovation is real and promising, but the surrounding product infrastructure does not exist.

---

*Generated for internal roadmap planning.*
