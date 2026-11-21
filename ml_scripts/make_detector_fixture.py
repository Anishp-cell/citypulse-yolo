"""Dev-only: record ultralytics predictions for a fixture image so the ONNX detector can be checked against them.

    python ml_scripts/make_detector_fixture.py tests/fixtures/pothole.jpg
Requires requirements-ml.txt (torch + ultralytics) and the training weights on disk.
"""

import json
import sys
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path("runs_citypulse/yolov8n_pothole_vbest2/weights/best.pt")


def main(image: str) -> None:
    model = YOLO(str(WEIGHTS))
    res = model.predict(image, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    dets = [{"label": res.names[int(b.cls)], "confidence": round(float(b.conf), 4),
             "box": [round(float(v)) for v in b.xyxy[0]]} for b in res.boxes]
    out = Path(image).with_name(Path(image).stem + "_expected.json")
    out.write_text(json.dumps({"imgsz": 640, "detections": dets}, indent=2))
    print(out, dets)


if __name__ == "__main__":
    main(sys.argv[1])
