import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.ml.detector import Detection, OnnxPotholeDetector, suggest_severity
from backend.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def detector():
    return OnnxPotholeDetector(Settings(_env_file=None).model_path)


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter)


def test_onnx_matches_ultralytics_reference(detector):
    expected = json.loads((FIXTURES / "pothole_expected.json").read_text())["detections"]
    got = detector.detect(cv2.imread(str(FIXTURES / "pothole.jpg")))
    assert [d.label for d in got] == [e["label"] for e in expected]
    for d, e in zip(got, expected, strict=True):
        assert iou(d.box, e["box"]) >= 0.9
        assert abs(d.confidence - e["confidence"]) <= 0.05


def test_blank_image_no_detections(detector):
    assert detector.detect(np.full((480, 640, 3), 127, dtype=np.uint8)) == []


def test_output_types_and_bounds(detector):
    img = cv2.imread(str(FIXTURES / "pothole.jpg"))
    h, w = img.shape[:2]
    for d in detector.detect(img):
        assert isinstance(d, Detection) and isinstance(d.confidence, float)
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def _pothole(area_ratio, shape=(1000, 1000)):
    side = int((area_ratio * shape[0] * shape[1]) ** 0.5)
    return Detection("pothole", 0.9, (0, 0, side, side))


@pytest.mark.parametrize("ratio,expected", [(0.01, "low"), (0.019, "low"), (0.021, "medium"),
                                            (0.079, "medium"), (0.081, "high"), (0.5, "high")])
def test_suggest_severity_boundaries(ratio, expected):
    assert suggest_severity([_pothole(ratio)], (1000, 1000)) == expected


def test_suggest_severity_uses_largest_pothole_and_ignores_other_labels():
    dets = [_pothole(0.01), _pothole(0.1), Detection("severe_accident", 0.9, (0, 0, 1000, 1000))]
    assert suggest_severity(dets, (1000, 1000)) == "high"


def test_suggest_severity_without_potholes_is_medium():
    assert suggest_severity([], (100, 100)) == "medium"
