# Use Python 3.11 (stable with torch/ultralytics)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libs needed by OpenCV + video codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

# Copy application code
COPY backend/ backend/
COPY frontend/ frontend/
COPY runs_citypulse/yolov8n_pothole_vbest2/weights/best.pt runs_citypulse/yolov8n_pothole_vbest2/weights/best.pt
COPY runs_citypulse/yolov8n_pothole_vbest2/args.yaml runs_citypulse/yolov8n_pothole_vbest2/args.yaml

# Expose port (Cloud Run sets $PORT; default 8080)
ENV PORT=8080
CMD ["sh", "-c", "cd /app && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
