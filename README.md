# CityPulse — AI-Powered Road Safety and Infrastructure Intelligence Platform

Professional-grade computer vision system for real-time detection of road accidents and infrastructure defects, paired with automated emergency response and municipal accountability infrastructure.

---

## Overview

CityPulse is a government-deployable AI platform designed for the Indian Ministry of Road Transport & Highways, Smart Cities Mission, and municipal administrations. It combines advanced computer vision (YOLOv8) with generative AI (Google Gemini) to detect traffic accidents and road defects in real-time, generate actionable emergency guidance for bystanders, and trigger automated alerts to the appropriate authorities.

**Core Problem**: India records 4.6 lakh road accidents and 1.68 lakh fatalities annually. Emergency response delay is the primary cause of preventable deaths. Road infrastructure defects (potholes) cause an estimated 3,500+ deaths per year. CityPulse directly addresses this by automating detection and reducing notification latency.

---

## System Function: Input and Output

### Input

The system accepts road scene data from multiple sources:

| Source | Format | Method |
|---|---|---|
| **Mobile Camera** | JPEG, PNG | Drag-and-drop web upload or mobile app capture |
| **Dashcam Footage** | MP4, AVI, MOV, MKV, WebM | Video file upload; system samples at 1 FPS |
| **Traffic CCTV** | RTSP/HLS Live Stream | Direct stream ingestion (planned for Phase 2) |
| **Drone Footage** | Video file or live stream | Upload or real-time integration |

Size limits: 50 MB for images, 500 MB for videos.

### Processing Pipeline

1. **Object Detection**: YOLOv8n inference across 6 incident classes with confidence and IoU thresholds
2. **Severity Classification**: Automatic assignment of emergency level (0-4) based on detected class
3. **Content Generation**: Gemini 2.0 Flash AI produces context-aware emergency medical guidance, with structured knowledge base fallback for offline operation
4. **Artifact Annotation**: Bounding boxes, severity labels, and confidence scores overlaid on output image
5. **Validation**: Confidence scoring and IoU filtering ensure only high-probability detections are reported

### Output

For each incident analysis:

```json
{
  "timestamp": "2026-06-04T10:23:45Z",
  "incident_type": "severe_accident | moderate_accident | minor_accident | pothole | no_accident",
  "severity_level": 0-4,
  "emergency_level": 0-4,
  "detections": [
    {
      "class": "severe_accident",
      "confidence": 0.87,
      "bounding_box": [x1, y1, x2, y2]
    }
  ],
  "annotated_image_url": "/static/processed/incident_uuid.jpg",
  "emergency_guidance": {
    "immediate_actions": ["Call 108 immediately", "Ensure scene safety", ...],
    "warning_signs": ["Loss of consciousness", ...],
    "do_not": ["Move victim unconditionally", ...],
    "recommendations": [...]
  },
  "alerts_sent": {
    "emergency_services": true,
    "municipality": false
  }
}
```

---

## Language Architecture: Python Only

CityPulse is built entirely in **Python** with no C++ components. All critical sections are written in high-level Python or native Python libraries:

| Component | Language | Library |
|---|---|---|
| **ML Inference Engine** | Python | PyTorch 2.3 + Ultralytics YOLOv8n |
| **Image Processing** | Python | OpenCV 4.10 (C++ core, Python bindings) |
| **Web Backend** | Python | FastAPI + Uvicorn (async) |
| **Video Processing** | Python | OpenCV (ffmpeg integration) |
| **Generative AI** | Python | google-generativeai (Gemini API client) |
| **Database ORM** | Python | SQLAlchemy + Psycopg2 |
| **Notifications** | Python | SendGrid + Twilio SDKs |

**Why Python?**: Prioritizes time-to-market for government deployment. Production inference uses ONNX Runtime for CPU-optimized execution. For ultra-low-latency edge deployment (future), model can be converted to TensorRT (NVIDIA) or Core ML (Apple), but core platform remains Python.

---

## Content Generation: Emergency Medical Guidance

"Content generation" refers to the automatic production of **actionable emergency medical first-aid instructions** tailored to the detected incident severity.

### Dual-Layer Approach

**Layer 1: Structured Knowledge Base** (Always Available)
A hardcoded medical guidance dictionary (see [backend/service/traffic_safety_service.py](backend/service/traffic_safety_service.py#L46-L102)) maps each severity level to:
- Immediate actions (e.g., "Call 108 immediately")
- Warning signs to monitor (e.g., "Loss of consciousness")
- Critical "Do Not" actions (e.g., "Do not remove helmet from motorcyclist", "Do not move victim unconditionally")
- Recommendations (e.g., "Apply pressure to bleeding", "Start CPR if no pulse")

This guidance is ALWAYS available, even without internet—it is the safety-critical fallback.

**Layer 2: Gemini 2.0 Flash AI Enhancement** (When Connected)
If a Google Gemini API key is configured, the system calls `gemini-2.0-flash-lite` with a structured prompt:

```
You are an emergency medical AI assistant. A road incident has been detected.

INCIDENT TYPE: Severe Accident
DETECTED SEVERITY: Severe (Emergency Level 3)
DETECTIONS: 2 vehicles involved, visible damage

Provide a concise 3-4 sentence summary of:
1. The single most critical immediate action
2. Key danger signs to watch
3. When to call emergency services
```

Parameters:
- Temperature: 0.3 (low randomness for medical accuracy)
- Max tokens: 200 (concise, actionable output)
- Timeout: < 2 seconds

If Gemini is unreachable, the structured knowledge base guidance is returned immediately—the system **never fails silently on safety-critical content**.

### Severity-to-Guidance Examples

**No Accident (Level 0)**:
> No accident detected. Drive safely!

**Minor Accident (Level 1)**:
> Turn on hazard lights → Check for injuries → Move vehicles if safe → Call police for report → Exchange information → Document scene. Visit doctor within 24-48 hours. Monitor for whiplash symptoms. Do NOT leave the scene or admit fault.

**Severe Accident (Level 3)**:
> CALL 108 IMMEDIATELY → Ensure scene safety → DO NOT MOVE VICTIMS → Check Airway-Breathing-Circulation → Start CPR if needed → Spinal stabilization. Watch for: no pulse, uncontrolled bleeding, unresponsiveness. Do NOT remove impaled objects or move trapped victims without tools.

**Pothole (Infrastructure Hazard)**:
> Report location to municipal department → Drive slowly to avoid further damage → Warn other drivers if possible. Submit annotated photo to municipal app. Check vehicle suspension and tire damage.

---

## Validation: Detection Criteria and Pass/Fail

### Image-Level Validation

**Input Validation** (Pre-detection):
- File type: JPEG, PNG, WebP only (magic byte verification)
- File size: Max 50 MB
- Dimensions: Minimum 64x64 pixels (OpenCV requirement)

**Output Validation** (Post-detection):
The detection pipeline applies two thresholds to filter false positives:

1. **Confidence Threshold**: 0.25
   - Each detected object must have confidence ≥ 0.25 (25% probability)
   - Objects below this threshold are discarded as unreliable

2. **IoU (Intersection over Union) Threshold**: 0.50
   - Overlapping bounding boxes with IoU ≥ 0.50 are merged (Non-Max Suppression)
   - Prevents duplicate detections of the same incident

3. **Severity Classification**:
   - All 6 classes are detected simultaneously
   - The **highest-severity class** determines the incident classification
   - Severity ranking: `totaled_vehicle (4) > severe_accident (3) > moderate_accident (2) > minor_accident (1) > pothole (0) > no_accident (0)`

**Pass/Fail Logic**:
- **PASS**: Confidence > 0.25 AND detection aligns with expected road geometry
- **FAIL**: Confidence < 0.25 OR image too degraded → classified as `no_accident` (fallback safe state)

### Video-Level Validation

For video uploads:
1. Frames are sampled at **1 FPS** (configurable; prevents processing identical frames 30 times/second)
2. Each frame is independently validated using the above image pipeline
3. **Worst-severity keyframe** (highest emergency level) is saved as the representative annotated image
4. **Frame count threshold**: If fewer than 5 frames of the video contain detections, the overall incident is downgraded to `no_accident` (to filter camera artifacts)

### Model Quality Metrics

The YOLOv8n model was trained on 6,422 labeled images:

| Metric | Value | Interpretation |
|---|---|---|
| Precision (all classes) | 0.63 | 63% of predicted detections are correct |
| Recall (all classes) | 0.63 | 63% of ground-truth incidents are detected |
| mAP@50 (IoU=0.50) | 0.57 | Average precision at lenient IoU threshold |
| mAP@50-95 (strict) | 0.33 | Average precision across strict IoU thresholds |
| Inference Speed | 25 FPS (GPU) / 5 FPS (CPU) | Real-time capable on modern hardware |

**Status**: Model meets MVP performance for government pilot. Production deployment requires: (a) Indian road-specific dataset enrichment, (b) targeted retraining on underrepresented classes (totaled vehicles), (c) target mAP > 0.75.

---

## Agentic / RAG Component: How AI Guidance Works

The "agentic" aspect of CityPulse refers to the **conditional, stateful generation of emergency guidance** based on incident detection.

### Architecture

CityPulse uses a **retrieval-augmented generation (RAG) pattern** with structured fallback:

```
Detected Incident
       |
       v
Severity Classifier
       |
       +--- Severity Found in Knowledge Base? ----> YES ---> Return Structured Guidance
       |                                                      (Always available)
       +--- NO ---> Query Gemini 2.0 Flash
                     |
                     +--- Success? ---> Return AI-enhanced guidance
                     |
                     +--- Timeout/Error? ---> Fall back to KB guidance
```

### Knowledge Base (Structured RAG)

The knowledge base is a Python dictionary in memory ([ACCIDENT_GUIDANCE](backend/service/traffic_safety_service.py#L46-L102)):

```python
ACCIDENT_GUIDANCE = {
    'minor_accident': {
        'severity': 'Minor',
        'immediate_actions': [...],
        'warning_signs': [...],
        'do_not_do': [...]
    },
    # ... 5 more classes
}
```

This is the **retrieval** layer: given a detected incident class, the system retrieves pre-authored medical guidance.

### Generative Layer (Gemini)

If Gemini API is available, the system optionally calls it to **augment** the structured guidance with contextual details:

- Takes the structured guidance as context
- Adds detected object confidence scores and bounding box positions
- Requests a concise (3-4 sentence) summary of immediate actions and danger signs
- Merges AI output with structured guidance for final response

### Fallback Behavior

If Gemini is unavailable or times out:
- Structured knowledge base is returned immediately
- No delay in emergency guidance
- User is notified: "Using local knowledge base (Generative AI unavailable)"

### Why This Pattern?

For life-safety systems in regions with poor connectivity (India's rural areas), the structured KB ensures guidance is **always available, latency-free**. The Gemini layer adds personalization and context awareness when possible, but the system never depends on it.

---

## Docker and GCP Deployment

### Containerization (Docker)

The application is packaged as a **Python 3.11 slim Docker image** with system dependencies for OpenCV and ffmpeg:

```dockerfile
FROM python:3.11-slim
RUN apt-get install libgl1 libglib2.0-0 ffmpeg  # System dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ frontend/ model_weights/
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "$PORT"]
```

**Build & Run**:
```bash
docker build -t citypulse-api .
docker run -p 8080:8080 --env-file .env citypulse-api
```

**Size**: ~2.8 GB (PyTorch + model weights); optimized via multi-stage builds if needed.

### Google Cloud Run Deployment

CityPulse is designed for **Google Cloud Run** (serverless container platform):

**Configuration**:
- **Memory**: 4 GB (required for PyTorch inference)
- **CPU**: 2 CPUs (concurrent request handling)
- **Timeout**: 600 seconds (10 minutes for video processing)
- **Concurrency**: 1 request per instance (ML model not thread-safe)

**Deployment**:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/citypulse
gcloud run deploy citypulse \
  --image gcr.io/PROJECT_ID/citypulse \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --set-env-vars GEMINI_API_KEY=xxx,SENDGRID_API_KEY=yyy
```

**Auto-scaling**: Cloud Run automatically scales from 0 to N instances based on traffic. Each instance handles one request concurrently.

**Networking**:
- API endpoint: `https://citypulse-XXXX.a.run.app`
- Private VPC connector option available for integration with government data centers
- Firewall rules can restrict to government IP ranges

### Alternative Deployments

**On-Premise (Government Data Center)**:
- Docker image can be deployed on NIC MeghRaj (National Informatics Centre cloud)
- Requires Kubernetes for multi-city orchestration (GKE equivalent: On-Prem K8s cluster)
- Database: PostgreSQL on-premise or via Cloud SQL with private VPC

**Hybrid (RECOMMENDED for India)**:
- Lightweight Python API runs on each city's data center (Docker-based)
- Centralized Gemini API calls routed through national gateway (for data sovereignty)
- Incident data replicated to state-level aggregation servers

---

## API Endpoints

All endpoints require Authentication (JWT bearer token) except `/health`.

### Public Health Check
```http
GET /health
```
Response: `{ "status": "ok", "model_loaded": true }`

### Image Analysis
```http
POST /api/analyze
Content-Type: multipart/form-data

image: <file.jpg>
latitude: <float>
longitude: <float>
```
Response: Annotated image URL + detections + guidance (< 2 seconds)

### Video Analysis
```http
POST /api/analyze-video
Content-Type: multipart/form-data

video: <file.mp4>
latitude: <float>
longitude: <float>
```
Response: Keyframe annotated image + timeline + guidance (< 30 seconds for typical video)

### Send Alert
```http
POST /api/notify
Content-Type: application/json
Authorization: Bearer <token>

{
  "incident_id": "uuid",
  "severity": "severe_accident",
  "location": "MG Road, Bangalore",
  "contact_authorities": ["108", "police"]
}
```

### Interactive Documentation
```
GET /docs          → Swagger UI
GET /openapi.json  → OpenAPI schema
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL (for production; SQLite for development)
- 4+ GB RAM
- NVIDIA GPU optional (10x faster inference)

### Installation

```bash
# Clone and navigate
git clone <repo>
cd citypulse-yolo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your-key-here"
export DATABASE_URL="postgresql://user:pass@localhost/citypulse"
# (or leave unset for SQLite fallback)

# Run development server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### Testing

```bash
# Upload test image
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@test_image.jpg" \
  -F "latitude=12.9716" \
  -F "longitude=77.5946"

# View results
curl http://localhost:8000/static/processed/last_result.json
```

---

## Project Structure

```
citypulse-yolo/
├── backend/
│   ├── main.py                          # FastAPI app entry point
│   ├── models.py                        # SQLAlchemy ORM models
│   ├── database.py                      # PostgreSQL connection
│   ├── service/
│   │   └── traffic_safety_service.py   # YOLOv8 + Gemini orchestration
│   └── routes/
│       └── auth.py                      # JWT authentication
├── frontend/
│   ├── index.html                       # Web UI
│   ├── app.js                           # JS upload and results rendering
│   └── style.css                        # Styling
├── ml_scripts/
│   ├── train_yolo.py                    # Model training pipeline
│   └── test.py                          # Model evaluation
├── data/
│   └── merged_yolo_dataset/             # Training data
├── runs_citypulse/
│   └── yolov8n_pothole_vbest2/
│       └── weights/best.pt              # Trained model
├── Dockerfile                           # Container specification
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## Technical Stack

| Layer | Technology | Version |
|---|---|---|
| ML Framework | PyTorch + Ultralytics | 2.3 + 8.3.0 |
| Object Detection | YOLOv8n | Custom-trained |
| Generative AI | Google Gemini | 2.0 Flash Lite |
| Web Backend | FastAPI + Uvicorn | Latest |
| Database | PostgreSQL | 14+ |
| Notifications | SendGrid + Twilio | Latest |
| Containerization | Docker | Latest |
| Cloud Platform | Google Cloud Run | - |
| Frontend | HTML5 / CSS3 / JavaScript | ES6+ |

---

## Performance Characteristics

### Latency

| Operation | Hardware | Time |
|---|---|---|
| Image detection + annotation | GPU (RTX 3060) | 0.8 - 1.2 seconds |
| Image detection + annotation | CPU (i7) | 3 - 5 seconds |
| Video frame sampling + detection | GPU | 0.5 - 1.0 FPS |
| Gemini guidance generation | Network | 1 - 2 seconds |

### Throughput

- **Peak concurrent requests** (Cloud Run): 100+ simultaneous incidents
- **Sustained rate**: 1,000+ detections/hour (10 instances, CPU-based)
- **GPU-accelerated**: 10,000+ detections/hour (1 GPU instance)

### Storage

- **Model weights**: 45 MB (YOLOv8n best.pt)
- **Per incident** (image + metadata): 2 - 8 MB (depends on resolution)
- **Video processing** (temporary): Requires 2x video size on disk for frame buffering

---

## Security & Compliance

### Current Gaps (MVP State)

- API authentication not yet implemented (planned Phase 2)
- No data encryption at rest (required for government deployment)
- CORS is open (`*`) in development
- No audit logging (required for compliance)

### Roadmap (Production)

- JWT-based API authentication with role-based access control (RBAC)
- AES-256 encryption for incident images and metadata
- PDPB (India's Personal Data Protection Bill) compliance layer
- Audit trail logging for all API calls and data access
- Face and license plate anonymization for privacy
- Integration with government cybersecurity standards (CERT-In, DSCI)

---

## Known Limitations

1. **Model Accuracy**: mAP@50 of 0.57 is suitable for MVP; production requires > 0.75
2. **Regional Data Bias**: Model trained on mixed datasets; needs Indian road-specific fine-tuning
3. **No Live Streaming**: CCTV integration planned for Phase 2
4. **No Mobile App**: Current deployment is web-only; native iOS/Android planned
5. **Notification Simulation**: Email/SMS currently mock-only; real Twilio/SendGrid integration partial
6. **Single-Language**: UI and guidance in English only; regional language support planned

---

## Roadmap

**Phase 1 (Current)**: MVP detection and guidance engine
**Phase 2** (Q3 2026): Mobile app, real CCTV integration, production database
**Phase 3** (Q4 2026): Multi-language support, dashboard analytics, government integrations (VAAHAN, NCRB)
**Phase 4** (2027): Kubernetes orchestration, on-premise deployments, advanced reporting

---

## Contributing

This is a government-pilot project. Contributions are accepted via GitHub Issues and Pull Requests. All code must comply with:
- PEP 8 Python style guide
- Unit test coverage > 80%
- Security review before production deployment

---

## Support and Contact

For government procurement, deployment, or technical questions:
- Email: [citypulse-support@example.com](mailto:citypulse-support@example.com)
- Documentation: [citypulse_project_description.md](citypulse_project_description.md.resolved)
- Gap Analysis: [citypulse_gap_analysis.md](citypulse_gap_analysis.md)
- Implementation Roadmap: [citypulse_implementation_roadmap.md](citypulse_implementation_roadmap.md)

---

## License

Government of India — Restricted Distribution

---

**Version**: 1.0 | **Last Updated**: June 2026
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
| `GET` | `/api/stats` | JWT | Aggregated stats (totals, severity, status, daily 7-day) |
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

### 4. Seed demo data (optional)

Populate the database with 40 realistic incidents across major world cities for demos and testing:

```bash
python scripts/seed_demo.py
```

This creates a demo officer account (`demo@citypulse.ai` / `DemoPass123!`) and inserts incidents with varied severities, statuses, and timestamps spread over the last 30 days. Override defaults via env vars:

```env
SEED_USER_EMAIL=your@email.com
SEED_USER_PASSWORD=YourPass!
SEED_COUNT=60
```

### 5. Docker

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

## Admin Dashboard

After logging in, click **Dashboard** in the nav bar to access the live analytics view:

- **Stat tiles** — Total incidents, Open, Resolved, Critical (severe + totaled) counts
- **Severity chart** — Doughnut breakdown across all 6 detection classes
- **Status pipeline chart** — Doughnut showing detected → notified → acknowledged → resolved → closed
- **7-day activity bar chart** — Daily incident volume over the past week
- **Live incident map** — Leaflet map with color-coded circle markers per severity; click any pin for details

All charts and the map update on each Dashboard visit by calling `GET /api/stats` and `GET /api/incidents`.

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
- [x] Admin analytics dashboard with Chart.js charts and Leaflet incident map

---

Built for safer cities everywhere.
