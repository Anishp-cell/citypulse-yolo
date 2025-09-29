import os
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime

# -------------------------
# Config
# -------------------------
# Point this to your best weights
WEIGHTS = r"D:\python\citypulse\runs_citypulse\yolov8n_pothole_vbest2\weights\best.pt"
# Optional: set to a video path or leave '' to be prompted
VIDEO_PATH = r"D:\python\citypulse\videoplayback (online-video-cutter.com).mp4"  
CONF_THRES = 0.25
IOU_THRES = 0.50

# Optional class filter:
# - Leave as None to show all classes.
# - Or set to a class *name* (e.g., "accident") to only draw that class.
CLASS_FILTER_NAME = None  # e.g. "accident"

# -------------------------
# Device
# -------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")

# -------------------------
# Load YOLO model
# -------------------------
if not os.path.exists(WEIGHTS):
    raise FileNotFoundError(f"YOLO weights not found at: {WEIGHTS}")

model = YOLO(WEIGHTS)
names = model.names  # dict: {class_id: class_name}
print(f"Loaded model with classes: {names}")

# Resolve class filter to ID if provided
class_filter_ids = None
if CLASS_FILTER_NAME is not None:
    # find class id by name (case-insensitive)
    target = CLASS_FILTER_NAME.strip().lower()
    found = [cid for cid, cname in names.items() if str(cname).lower() == target]
    if not found:
        raise ValueError(f"Class '{CLASS_FILTER_NAME}' not found in model classes: {list(names.values())}")
    class_filter_ids = set(found)
    print(f"Filtering to class: {CLASS_FILTER_NAME} (id(s): {sorted(class_filter_ids)})")

# -------------------------
# Get input video
# -------------------------
if not VIDEO_PATH:
    VIDEO_PATH = input("Enter path to video file (e.g., D:\\python\\citypulse\\test_videos\\car_accident.mp4): ").strip('" ')
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Failed to open video.")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps and fps > 0 else 30.0

# Output path
base, ext = os.path.splitext(VIDEO_PATH)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"{base}_yolo_pred_{stamp}{ext}"
fourcc = cv2.VideoWriter_fourcc(*"mp4v") if ext.lower() in [".mp4", ".m4v"] else cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

print(f"Reading:  {VIDEO_PATH}")
print(f"Writing:  {out_path}")
print("Press 'q' to quit early.")

# -------------------------
# Inference loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the BGR frame directly
    results = model.predict(
        source=frame,
        device=device,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False
    )

    # There is one result per image/frame
    res = results[0]

    # Draw detections
    if res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_name = names.get(cls_id, str(cls_id))

            # Filter by class if requested
            if class_filter_ids is not None and cls_id not in class_filter_ids:
                continue

            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Label text: "class conf"
            label = f"{cls_name} {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_text = max(int(y1) - 10, th + 5)
            cv2.rectangle(frame, (int(x1), y1_text - th - 6), (int(x1) + tw + 6, y1_text + 2), (0, 255, 0), -1)
            cv2.putText(frame, label, (int(x1) + 3, y1_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    writer.write(frame)
    cv2.imshow("YOLO Video - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print("Done.")
print(f"Saved annotated video to: {out_path}")
