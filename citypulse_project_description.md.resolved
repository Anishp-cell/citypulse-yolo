# CityPulse — Comprehensive Project Description
## AI-Powered Road Safety & Urban Infrastructure Intelligence Platform

> **Prepared for:** Indian Government — Ministry of Road Transport & Highways (MoRTH), Smart Cities Mission, Municipal Corporations
> **Version:** 1.0 | March 2026

---

## 1. Executive Summary

CityPulse is an **AI-powered road safety and infrastructure monitoring platform** that uses computer vision and generative AI to:

1. **Detect road accidents and potholes** in real-time from images, videos, and live camera feeds
2. **Classify incident severity** across 6 distinct categories — from "no accident" to "totaled vehicle"
3. **Generate life-saving emergency guidance** — actionable medical first-aid steps that any bystander can follow
4. **Alert authorities automatically** — emergency services for accidents, municipal departments for potholes
5. **Track, manage, and analyze incidents** — geospatial heatmaps, trend analytics, and repair accountability

> **The Problem**: India records **4.6 lakh road accidents** and **1.68 lakh fatalities annually** (MoRTH 2023). Potholes alone cause **3,500+ deaths per year**. Emergency response delay is the #1 cause of preventable fatalities — CityPulse directly reduces that delay.

---

## 2. How CityPulse Works — End-to-End Technical Architecture

### 2.1 — System Overview

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌───────────────┐
│   INPUT      │ →  │  DETECTION   │ →  │  AI GUIDANCE   │ →  │   OUTPUT      │
│              │    │              │    │                │    │               │
│ • Phone Cam  │    │ YOLOv8 Model │    │ Gemini 2.0     │    │ • Annotated   │
│ • Dashcam    │    │ 6-class      │    │ Flash Lite     │    │   Image       │
│ • CCTV Feed  │    │ detection    │    │ Emergency      │    │ • Severity    │
│ • Uploaded   │    │ <1s latency  │    │ medical advice │    │   Rating      │
│   Image/     │    │              │    │ Structured     │    │ • Guidance    │
│   Video      │    │              │    │ fallback KB    │    │ • SMS/Email   │
│              │    │              │    │                │    │   Alert       │
└─────────────┘    └──────────────┘    └────────────────┘    └───────────────┘
```

### 2.2 — Input Layer

CityPulse accepts road scene media from multiple sources:

| Source | Format | How It Works |
|---|---|---|
| **Phone Camera** | JPG, PNG | Citizen captures photo → uploads via mobile app or web |
| **Dashcam Footage** | MP4, AVI, MOV, MKV, WebM | Upload video → system samples frames at 1 FPS, analyzes each |
| **Traffic CCTV** | RTSP live stream | System connects to IP cameras, continuously processes frames |
| **Drone Footage** | Video file | Upload or live stream from municipal survey drones |

- **Drag-and-drop upload** via web interface
- **Camera capture + GPS tagging** via mobile app
- **RTSP stream ingestion** for 24/7 automated monitoring (planned)

### 2.3 — Detection Engine (YOLOv8 Computer Vision)

The core of CityPulse is a **custom-trained YOLOv8 object detection model** that localizes and classifies road incidents in real-time.

#### Model Architecture
- **Base**: YOLOv8n (Ultralytics) — optimized for speed, suitable for CPU + mobile inference
- **Framework**: PyTorch 2.3 + Ultralytics SDK
- **Training**: Transfer learning from COCO pre-trained weights, fine-tuned on merged road datasets

#### 6-Class Detection System

| Class ID | Detection Class | Severity Level | Emergency Level | What It Detects |
|---|---|---|---|---|
| 0 | `no_accident` | None | 0 | Normal traffic — no action needed |
| 1 | `minor_accident` | Minor | 1 | Fender benders, small collisions, single-vehicle bumps |
| 2 | `moderate_accident` | Moderate | 2 | Multi-vehicle collisions with visible damage, possible injuries |
| 3 | `severe_accident` | Severe | 3 | Major crashes with significant damage, likely serious injuries |
| 4 | `totaled_vehicle` | Critical | 4 | Catastrophic crashes, vehicle destruction, multi-casualty events |
| 5 | `pothole` | Infrastructure Hazard | 0 | Road surface defects requiring municipal repair |

#### Detection Process (Image)
1. Image is loaded via OpenCV
2. YOLOv8 runs inference with **confidence threshold = 0.25**, **IoU threshold = 0.50**
3. For each detected object: bounding box coordinates, class ID, and confidence score are extracted
4. Bounding boxes are drawn on the image with **color-coded severity** (green → yellow → orange → red → cyan for potholes)
5. The highest-severity detection determines the overall incident classification

#### Detection Process (Video)
1. Video is opened via OpenCV `VideoCapture`
2. Frames are sampled at **1 FPS** (configurable) — this avoids processing 30 identical frames per second
3. Each sampled frame is processed independently through the same detection pipeline
4. An **incident timeline** is generated: timestamp → severity → detection count per frame
5. The **worst-severity keyframe** (frame with highest emergency level) is saved as the representative image
6. All detections across all frames are aggregated for the final report

#### Model Performance

| Metric | Value | Notes |
|---|---|---|
| Precision | 0.63 | Across all 6 classes |
| Recall | 0.63 | Across all 6 classes |
| mAP@50 | 0.57 | At 50% IoU threshold |
| mAP@50-95 | 0.33 | Strict multi-threshold average |
| Inference Speed | ~25 FPS (GPU), ~5 FPS (CPU) | YOLOv8n on RTX 3060 / i7 CPU |

#### Training Dataset
- **6,422 images** — merged from car accident datasets + pothole datasets
- **Annotation formats**: Mixed XML (Pascal VOC) and YOLO format; preprocessing pipeline normalizes all to YOLO [.txt](file:///d:/python/citypulse/requirements.txt)
- **Augmentation**: Horizontal flip, mosaic, color jitter, scale variation
- **Training**: 58 epochs with early stopping, mixed-precision training (FP16), class-balanced sampling

### 2.4 — AI Guidance Engine (Gemini + Structured Knowledge Base)

After detection, CityPulse generates **emergency medical guidance** tailored to the incident severity. This uses a **dual-layer approach**:

#### Layer 1: Structured Knowledge Base
A hardcoded medical guidance dictionary maps each severity level to:
- **Immediate actions** (e.g., "Call 108 immediately", "Do not move victims")
- **Warning signs** to monitor (e.g., "Loss of consciousness", "Severe bleeding")
- **What NOT to do** (e.g., "Do not remove helmet from motorcyclist", "Do not give food/water")
- **Recommendations** (e.g., "Start CPR if no pulse", "Apply pressure to bleeding")

This is always available, even without internet — it is the safety-critical fallback.

#### Layer 2: Gemini AI Enhancement
If a **Google Gemini API key** is configured, CityPulse calls **Gemini 2.0 Flash Lite** with a prompt like:

```
You are an emergency medical AI assistant. A road incident
has been detected by computer vision.

INCIDENT TYPE: Severe Accident
SEVERITY: Severe
DETECTIONS: 2 objects detected: severe accident (87%), totaled vehicle (72%)

Provide a brief, actionable 3-4 sentence summary covering:
1. The single most critical immediate action to take
2. Key danger signs to watch for
3. Whether/when to call emergency services (108/911)
```

**Parameters**: Temperature 0.3 (low randomness), max 200 tokens (concise output), < 2 second response time.

If Gemini is unavailable, the structured knowledge base provides equivalent guidance — the system **never fails silently**.

#### Severity-to-Guidance Mapping Examples

**Minor Accident (Level 1)**:
> ✓ Turn on hazard lights → ✓ Check for injuries → ✓ Move vehicles if safe → ✓ Exchange info → ✓ Document scene. Visit doctor within 24–48 hours. Do NOT leave scene or admit fault.

**Severe Accident (Level 3)**:
> 🚨🚨 CALL 108 IMMEDIATELY → ⚠️ Scene safety → DO NOT MOVE VICTIMS → Check Airway-Breathing-Circulation → Start CPR if needed → Spinal stabilization. Watch for: no pulse, uncontrolled bleeding, unresponsiveness.

**Pothole (Infrastructure)**:
> Report location → Drive slowly → Warn other drivers. Submit photo to municipal app. Check tire/suspension damage.

### 2.5 — Notification & Alerting System

CityPulse routes incident alerts to the correct authority:

| Incident Type | Authority | Contact Method |
|---|---|---|
| Any Accident | Emergency Services (108/100) | SMS + Push Notification + Email |
| Pothole | Roads & Infrastructure Dept | Email + Portal Submission |

Each notification includes:
- **Incident type and severity**
- **Annotated image** with bounding boxes
- **GPS coordinates** and address
- **Timestamp** with timezone
- **AI-generated incident summary**

### 2.6 — Backend API Architecture

Built on **FastAPI** (Python async web framework), the backend exposes:

| Endpoint | Method | Function |
|---|---|---|
| `GET /` | — | Serves the web frontend |
| `GET /health` | — | Health check — returns model load status |
| `POST /api/analyze` | Image upload | Runs YOLOv8 on image, returns detections + guidance |
| `POST /api/analyze-video` | Video upload | Processes video frames, returns timeline + keyframe |
| `POST /api/notify` | JSON body | Sends alert to appropriate authority |
| `GET /docs` | — | Interactive API documentation (Swagger UI) |

**Key design decisions**:
- **Model cached in memory** — loaded once at startup, shared across all requests
- **Device auto-detection** — CUDA GPU if available, MPS (Apple Silicon), or CPU fallback
- **Static file serving** — annotated images served from `/static/processed/`
- **CORS enabled** — for cross-origin frontend access

### 2.7 — Frontend Web Application

A single-page web application built with vanilla HTML/CSS/JavaScript:

- **Hero section** — project overview with animated gradient text
- **Upload zone** — drag-and-drop with auto-detection of image vs. video
- **Mode toggle** — switch between image and video analysis
- **Results display**:
  - Severity banner (color-coded)
  - Annotated image with bounding boxes
  - Detection list with confidence percentages
  - Video stats (duration, frames analyzed, incidents found)
  - Incident timeline (video only)
  - Full emergency guidance panel
- **Notification panel** — enter location, send alert to authorities
- **How It Works** section — 4-step pipeline visualization
- **Detection classes** — visual guide to all 6 severity levels
- **Tech stack** display

### 2.8 — Deployment Architecture

| Component | Technology | Configuration |
|---|---|---|
| Container | Docker (Python 3.11 slim) | Multi-stage build with ffmpeg |
| Cloud Platform | Google Cloud Run | Serverless, auto-scaling |
| Model Weights | Bundled in Docker image | `runs_citypulse/.../best.pt` |
| Static Assets | Served from FastAPI | `/static/` and `/frontend/` dirs |

---

## 3. Current Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | **PyTorch 2.3** + **Ultralytics (YOLOv8)** |
| Computer Vision | **OpenCV 4.10** |
| Generative AI | **Google Gemini 2.0 Flash Lite** |
| Backend | **FastAPI** + **Uvicorn** |
| Frontend | **HTML5 / CSS3 / JavaScript** |
| Inference Runtime | **ONNX Runtime 1.18** |
| Containerization | **Docker** |
| Cloud Deployment | **Google Cloud Run** |
| Data Processing | **NumPy**, **Pandas** |

---

## 4. Government Implementation Plan

### 4.1 — Target Government Bodies

| Body | Relevance | CityPulse Module |
|---|---|---|
| **MoRTH** (Ministry of Road Transport & Highways) | National road safety policy | Accident detection + analytics |
| **Smart Cities Mission** (MoHUA) | Urban infrastructure monitoring | CCTV integration + pothole tracking |
| **NCRB** (National Crime Records Bureau) | Accident data collection | Automated incident logging |
| **State PWDs** (Public Works Departments) | Road maintenance | Pothole detection + repair tracking |
| **Traffic Police** (State-level) | Accident response | Real-time alerts + field officer app |
| **Municipal Corporations** | Local road repair | Citizen reporting + accountability |
| **NHAI** (National Highways Authority) | Highway monitoring | CCTV integration + analytics |

### 4.2 — Deployment Model for India

```
┌─────────────────────────────────────────────────────────────────┐
│                   NATIONAL LEVEL (MoRTH / NIC)                  │
│        Aggregated analytics, policy dashboards, NCRB feed       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│  STATE 1     │  │  STATE 2     │  │  STATE N     │
│  (e.g. MH)   │  │  (e.g. KA)   │  │              │
│  State PWD   │  │  State PWD   │  │              │
│  State       │  │  State       │  │              │
│  Traffic     │  │  Traffic     │  │              │
│  Police      │  │  Police      │  │              │
└──────┬───────┘  └──────┬───────┘  └──────────────┘
       │                 │
  ┌────┼────┐       ┌────┼────┐
  │    │    │       │    │    │
┌─▼──┐▼──┐▼──┐  ┌─▼──┐▼──┐▼──┐
│City│City│City│  │City│City│City│
│ 1  │ 2  │ 3 │  │ 1  │ 2  │ 3 │
└────┘└───┘└──┘  └────┘└───┘└──┘
```

#### City-Level Deployment
Each city/district gets:
- **CityPulse API server** (deployed on NIC MeghRaj / state data center)
- **CCTV integration** — connected to existing traffic camera network
- **Citizen mobile app** — published on Play Store, city-branded
- **Admin dashboard** — accessible to traffic police HQ + municipal corporation
- **Local database** — incident records stay within state jurisdiction (data sovereignty)

#### State-Level Aggregation
- State traffic police sees all incidents across cities
- State PWD tracks pothole density and repair backlog across districts
- State-level analytics: accident hotspots, seasonal patterns, response time benchmarks

#### National-Level Dashboard
- MoRTH sees macro trends across all states
- NCRB receives structured accident data for the national database
- Smart Cities Mission tracks infrastructure health scores

### 4.3 — Integration with Existing Government Systems

| System | Integration Method | Purpose |
|---|---|---|
| **VAAHAN** (Vehicle Registration DB) | API lookup | Identify vehicles involved in accidents from plates |
| **Sarathi** (Driving License DB) | API lookup | Link incidents to driver records |
| **NCRB ADSI** (Accidental Deaths & Suicides in India) | Data feed | Automated accident statistical reporting |
| **eSeva / DigiLocker** | OAuth | Citizen identity verification |
| **112 (ERSS)** | Alert integration | Route severe accident alerts to the national emergency number |
| **Smart City CC** (Command & Control Centers) | Dashboard embed / API | Feed into existing smart city operations rooms |

### 4.4 — Rollout Strategy

#### Pilot Phase (3 months, 1 city)
- Deploy in **1 smart city** (suggestion: Pune, Hyderabad, or Bhopal — cities with existing CCTV + smart city infra)
- Connect to **50 traffic cameras** at high-accident intersections
- Onboard **500 citizens** on mobile app beta
- Assign **20 traffic police officers** with field officer app access
- Measure: detection accuracy, alert response time, false positive rate, citizen engagement

#### Expansion Phase (6 months, 5 cities)
- Roll out across **5 smart cities** within the same state
- Integrate with state traffic police command center
- Train model on Indian road conditions using pilot data
- Target: **10,000+ active citizen users**, **200+ cameras connected**

#### Scale Phase (12+ months, multi-state)
- Multi-state deployment with NIC MeghRaj cloud infrastructure
- National dashboard for MoRTH
- NCRB data feed integration
- Target: **1 million+ citizens**, **5,000+ cameras**, **all Smart Cities Mission cities**

### 4.5 — Revenue / Licensing Model for Government

| Model | Description | Price Range |
|---|---|---|
| **SaaS Per-City License** | Annual subscription per city | ₹15–25 Lakhs/year (Tier-2 city), ₹50–75 Lakhs/year (Metro) |
| **Per-Camera License** | Per connected CCTV camera/year | ₹5,000–10,000/camera/year |
| **Citizen App** | Free for citizens (government-subsidized) | ₹0 |
| **Custom Deployment** | On-premise in government data center | ₹1–2 Crore (one-time setup + AMC) |

### 4.6 — Compliance & Data Governance

| Requirement | CityPulse Approach |
|---|---|
| **DPDP Act, 2023** (Digital Personal Data Protection) | Consent-based citizen data collection; anonymization of faces/plates in stored images; data localization within India |
| **IT Act, 2000** | Encrypted data at rest (AES-256) and in transit (TLS 1.3); audit logging |
| **CERT-In Guidelines** | Incident response SOP; vulnerability disclosure program |
| **Data Localization** | All data stored on Indian servers (NIC MeghRaj / Google Cloud Mumbai region) |
| **Data Retention** | Configurable per government policy — default 2 years for incidents, 30 days for raw video |
| **Right to Erasure** | Citizen can delete their own reports and account data |

### 4.7 — Impact Projections

Based on pilot city deployment (~10 lakh population):

| Metric | Projected Impact |
|---|---|
| **Emergency response time** | Reduced by **40–60%** (automated alert vs. bystander phone call) |
| **Pothole repair backlog** | Reduced by **30%** in first year (accountability through tracking) |
| **Accident fatality rate** | Reduced by **10–15%** due to faster first-responder arrival |
| **Citizen reports** | **5,000+ monthly** reports (incentivized reporting) |
| **Infrastructure cost savings** | **₹2–5 Crore/year** in reduced vehicle damage claims from potholes |
| **Data visibility** | **100%** — every incident geo-tagged, time-stamped, severity-classified |

---

## 5. Competitive Advantage

| Feature | CityPulse | Typical Gov. Tools |
|---|---|---|
| Real-time AI detection | ✅ YOLOv8, <1s | ❌ Manual reporting |
| Severity classification | ✅ 6 levels, automated | ❌ Human judgment |
| AI emergency guidance | ✅ Gemini-powered | ❌ None |
| Mobile-first citizen reporting | ✅ Camera + GPS + offline | ⚠️ Web portal only |
| Incident lifecycle tracking | ✅ Detected → Resolved | ❌ Static records |
| Geospatial analytics | ✅ Heatmap + hotspot | ❌ Tabular data |
| On-device inference (offline) | ✅ ONNX/TFLite | ❌ Cloud-only |

---

## 6. Team & Contact

| Role | Responsibility |
|---|---|
| **AI/ML Engineer** | YOLO model training, Gemini integration, inference pipeline |
| **Full-Stack Developer** | FastAPI backend, Next.js frontend, mobile app |
| **DevOps Engineer** | Docker, GKE, CI/CD, monitoring |
| **Product Manager** | Government liaison, pilot management, roadmap |

---

> *CityPulse — Every second counts. Let AI watch the roads so humans can save lives.*
