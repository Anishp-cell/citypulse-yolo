"""Pothole detection. `Detector` is the interface the rest of the app depends on."""

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in image pixels


class Detector(Protocol):
    def detect(self, img_bgr: np.ndarray) -> list[Detection]: ...


class FakeDetector:
    """Returns a fixed list of detections; used in tests and when ML is disabled."""

    def __init__(self, detections: list[Detection] | None = None):
        self.detections = list(detections or [])

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        return list(self.detections)


DEFAULT_LABELS = ["no_accident", "minor_accident", "moderate_accident", "severe_accident",
                  "totaled_vehicle", "pothole"]
POTHOLE = "pothole"


class OnnxPotholeDetector:
    """YOLOv8 ONNX inference on CPU, matching ultralytics' predict() preprocessing and NMS."""

    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5):
        import ast

        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # small instances; keeps memory and CPU contention low
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"]) if "names" in meta else dict(enumerate(DEFAULT_LABELS))
        self.labels = [names[i] for i in sorted(names)]
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def _letterbox(self, img: np.ndarray, stride: int = 32):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_w, new_h = round(w * r), round(h * r)
        dw, dh = (self.imgsz - new_w) % stride / 2, (self.imgsz - new_h) % stride / 2
        if (new_w, new_h) != (w, h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, left, top

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        h, w = img_bgr.shape[:2]
        padded, gain, pad_x, pad_y = self._letterbox(img_bgr)
        blob = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        out = self.session.run(None, {self.input_name: blob})[0][0].T  # (anchors, 4 + classes)
        scores = out[:, 4:]
        class_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(scores)), class_ids]
        keep = confs > self.conf
        if not keep.any():
            return []
        boxes, confs, class_ids = out[keep, :4], confs[keep], class_ids[keep]
        xywh = np.column_stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]])
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), confs.tolist(), class_ids.tolist(), self.conf, self.iou)
        dets = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = xywh[i]
            x1 = int(round(np.clip((x - pad_x) / gain, 0, w)))
            y1 = int(round(np.clip((y - pad_y) / gain, 0, h)))
            x2 = int(round(np.clip((x + bw - pad_x) / gain, 0, w)))
            y2 = int(round(np.clip((y + bh - pad_y) / gain, 0, h)))
            if x2 > x1 and y2 > y1:
                dets.append(Detection(self.labels[int(class_ids[i])], float(confs[i]), (x1, y1, x2, y2)))
        return sorted(dets, key=lambda d: d.confidence, reverse=True)


def suggest_severity(dets: list[Detection], img_shape: tuple[int, ...]) -> str:
    """Severity from the largest pothole's share of the frame: <2% low, 2-8% medium, >8% high."""
    potholes = [d for d in dets if d.label == POTHOLE]
    if not potholes:
        return "medium"
    area = max((d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) for d in potholes)
    ratio = area / float(img_shape[0] * img_shape[1])
    if ratio < 0.02:
        return "low"
    if ratio <= 0.08:
        return "medium"
    return "high"
