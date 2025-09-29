# scripts/bench.py
import argparse, time, statistics
import cv2
from pathlib import Path
from service.inference import Detector

def bench_image(det: Detector, img_path: str, warmup=5, iters=30):
    for _ in range(warmup):
        det.predict_image(img_path)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        det.predict_image(img_path)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": statistics.median(times),
        "p95_ms": statistics.quantiles(times, n=100)[94],
        "mean_ms": sum(times)/len(times)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    args = ap.parse_args()

    det = Detector(args.weights, device=args.device, conf=args.conf, iou=args.iou)
    stats = bench_image(det, args.image)
    print({"device": det.device, **stats})

if __name__ == "__main__":
    main()
