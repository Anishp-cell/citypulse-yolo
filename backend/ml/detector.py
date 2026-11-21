"""Pothole detection. `Detector` is the interface the rest of the app depends on."""

from dataclasses import dataclass
from typing import Protocol

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
