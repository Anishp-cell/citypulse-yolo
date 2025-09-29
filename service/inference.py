# service/inference.py
import os
import cv2
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from ultralytics import YOLO

ArrayLike = Union[np.ndarray, "numpy.ndarray"]

class Detector:
    def __init__(
        self,
        weights: Union[str, Path],
        device: Optional[str] = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.50,
        per_class_conf: Optional[Dict[Union[int, str], float]] = None,
    ):
        """
        weights: .pt or .onnx (Ultralytics can run both)
        device:  'cuda:0' or 'cpu'. None => auto.
        per_class_conf: {class_name|class_id: conf_threshold}
        """
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # .onnx cannot be .to(device), but ultralytics handles backend internally
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.names: Dict[int, str] = {int(k): v for k, v in self.model.names.items()}
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.per_class_conf = self._resolve_thresholds(per_class_conf)

    # ---------- public API ----------
    def predict_image(
        self,
        img_or_path: Union[ArrayLike, str, Path],
        classes: Optional[Sequence[Union[int, str]]] = None,
        return_annotated: bool = False,
    ) -> Dict:
        """Runs detection on a single image (np array BGR or file path)."""
        src = img_or_path
        t0 = time.perf_counter()
        results = self.model.predict(
            source=src,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        res = results[0]
        dets = self._postprocess(res, classes)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        out = {
            "detections": dets,
            "image_shape": res.orig_img.shape[:2],  # (H, W)
            "time_ms": float(elapsed_ms),
        }
        if return_annotated:
            img = res.orig_img if isinstance(src, (str, Path)) else src
            out["annotated"] = self.annotate(img, dets)
        return out

    def predict_video(
        self,
        video_path: Union[str, Path],
        out_path: Optional[Union[str, Path]] = None,
        classes: Optional[Sequence[Union[int, str]]] = None,
        show: bool = False,
    ) -> Dict:
        """Reads a video from disk, writes optional annotated output, returns simple stats."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if out_path:
            out_path = str(out_path)
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_path.lower().endswith((".mp4", ".m4v")) else "XVID"))
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        times = []
        frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            out = self.predict_image(frame, classes=classes, return_annotated=True)
            times.append((time.perf_counter() - t0) * 1000.0)
            frames += 1

            ann = out["annotated"]
            if writer:
                writer.write(ann)
            if show:
                cv2.imshow("Detections (q to quit)", ann)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        return {
            "frames": frames,
            "avg_ms_per_frame": float(np.mean(times)) if times else 0.0,
            "p95_ms": float(np.percentile(times, 95)) if times else 0.0,
            "out_path": out_path,
        }

    def annotate(self, img_bgr: ArrayLike, detections: List[Dict], thickness: int = 2) -> ArrayLike:
        """Draws boxes + labels on a BGR image."""
        frame = img_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            label = f'{d["class_name"]} {d["confidence"]:.2f}'
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y = max(y1 - 10, th + 5)
            cv2.rectangle(frame, (x1, y - th - 6), (x1 + tw + 6, y + 2), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return frame

    # ---------- helpers ----------
    def _resolve_thresholds(self, per_class_conf: Optional[Dict[Union[int, str], float]]):
        if not per_class_conf:
            return None
        resolved: Dict[int, float] = {}
        for k, v in per_class_conf.items():
            if isinstance(k, int):
                resolved[k] = float(v)
            else:
                cid = self._name_to_id(k)
                if cid is not None:
                    resolved[cid] = float(v)
        return resolved or None

    def _name_to_id(self, name: str) -> Optional[int]:
        target = str(name).strip().lower()
        for cid, cname in self.names.items():
            if str(cname).lower() == target:
                return cid
        return None

    def _names_to_ids(self, items: Sequence[Union[int, str]]) -> List[int]:
        ids = []
        for it in items:
            if isinstance(it, int):
                ids.append(it)
            else:
                cid = self._name_to_id(it)
                if cid is not None:
                    ids.append(cid)
        return ids

    def _postprocess(self, res, classes: Optional[Sequence[Union[int, str]]]) -> List[Dict]:
        img = res.orig_img
        h, w = img.shape[:2]
        dets: List[Dict] = []

        if res.boxes is None or len(res.boxes) == 0:
            return dets

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int)

        allowed = None
        if classes is not None:
            allowed = set(self._names_to_ids(classes))

        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, clses):
            if allowed is not None and cid not in allowed:
                continue
            thr = (self.per_class_conf.get(cid) if self.per_class_conf else self.conf)
            if conf < thr:
                continue

            area = max(0.0, float((x2 - x1) * (y2 - y1)))
            area_norm = area / float(w * h) if w > 0 and h > 0 else 0.0

            dets.append({
                "class_id": int(cid),
                "class_name": str(self.names.get(int(cid), int(cid))),
                "confidence": float(conf),
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "xywhn": [
                    float(((x1 + x2) / 2) / w),
                    float(((y1 + y2) / 2) / h),
                    float((x2 - x1) / w),
                    float((y2 - y1) / h),
                ],
                "area_norm": float(area_norm),
            })
        return dets


# -------- CLI for quick manual tests --------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to .pt or .onnx")
    ap.add_argument("--image", help="Path to image")
    ap.add_argument("--video", help="Path to video")
    ap.add_argument("--out", help="Optional output (image/video) path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--device", default=None, help="cuda:0 or cpu")
    ap.add_argument("--classes", nargs="*", help="Filter by names or ids, e.g. pothole 3")
    ap.add_argument("--pconf", nargs="*", help="Per-class conf like pothole=0.22 severe_accident=0.35")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    pconf = None
    if args.pconf:
        pconf = {}
        for kv in args.pconf:
            k, v = kv.split("=")
            try:
                k = int(k)
            except ValueError:
                pass
            pconf[k] = float(v)

    det = Detector(
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        per_class_conf=pconf,
    )

    if args.image:
        out = det.predict_image(args.image, classes=args.classes, return_annotated=True)
        ann = out["annotated"]
        dst = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_pred.jpg"))
        cv2.imwrite(dst, ann)
        print(f"Saved: {dst} | time_ms={out['time_ms']:.1f}")
    elif args.video:
        stats = det.predict_video(args.video, out_path=args.out, classes=args.classes, show=args.show)
        print(stats)
    else:
        print("Provide --image or --video")
