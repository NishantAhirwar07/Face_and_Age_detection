"""
detector.py — Chronoscope core API

A small, UI-agnostic API for face detection + age estimation. It knows
nothing about Streamlit (or any other frontend) so it can be reused in a
script, a notebook, a batch job, or wrapped in a REST endpoint later.

    from detector import FaceAgeDetector

    detector = FaceAgeDetector().load()
    result = detector.analyze(image_bgr, confidence_threshold=0.7)
    annotated = detector.draw(image_bgr, result.detections)
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

MODEL_DIR = Path(__file__).parent / "models"

# Pretrained weights, sourced from the LearnOpenCV AgeGender project.
MODEL_FILES = {
    "opencv_face_detector.pbtxt": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/"
        "AgeGender/opencv_face_detector.pbtxt"
    ),
    "opencv_face_detector_uint8.pb": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/"
        "AgeGender/opencv_face_detector_uint8.pb"
    ),
    "age_deploy.prototxt": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/"
        "AgeGender/age_deploy.prototxt"
    ),
    "age_net.caffemodel": (
        "https://raw.githubusercontent.com/eveningglow/age-and-gender-"
        "classification/5b60d9f8a8608cdbbcdaaa39bf28f351e8d8553b/model/"
        "age_net.caffemodel"
    ),
}

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


@dataclass
class Detection:
    """A single detected face and its estimated age."""

    id: int
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
    face_confidence: float
    age_bracket: str
    age_confidence: float


@dataclass
class ScanResult:
    """Full result of analyzing one image."""

    detections: List[Detection] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)  # width, height

    @property
    def face_count(self) -> int:
        return len(self.detections)

    def to_records(self) -> List[dict]:
        """JSON/CSV-friendly representation."""
        return [
            {
                "id": d.id,
                "x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3],
                "face_confidence": round(d.face_confidence, 4),
                "age_bracket": d.age_bracket,
                "age_confidence": round(d.age_confidence, 4),
            }
            for d in self.detections
        ]


class ModelDownloadError(RuntimeError):
    """Raised when a required model file can't be fetched."""


def ensure_models(progress_callback: Optional[Callable[[str], None]] = None) -> None:
    """Download any missing model files into MODEL_DIR."""
    MODEL_DIR.mkdir(exist_ok=True)
    for filename, url in MODEL_FILES.items():
        dest = MODEL_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            continue
        if progress_callback:
            progress_callback(filename)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            raise ModelDownloadError(f"Could not download {filename}: {exc}") from exc


class FaceAgeDetector:
    """Loads the face + age models once and exposes a simple analyze() API."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._face_net = None
        self._age_net = None

    def load(self, progress_callback: Optional[Callable[[str], None]] = None) -> "FaceAgeDetector":
        ensure_models(progress_callback)
        self._face_net = cv2.dnn.readNetFromTensorflow(
            str(self.model_dir / "opencv_face_detector_uint8.pb"),
            str(self.model_dir / "opencv_face_detector.pbtxt"),
        )
        self._age_net = cv2.dnn.readNetFromCaffe(
            str(self.model_dir / "age_deploy.prototxt"),
            str(self.model_dir / "age_net.caffemodel"),
        )
        return self

    @property
    def ready(self) -> bool:
        return self._face_net is not None and self._age_net is not None

    def _detect_faces(self, frame: np.ndarray, confidence_threshold: float):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False
        )
        self._face_net.setInput(blob)
        raw = self._face_net.forward()

        boxes = []
        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            if confidence < confidence_threshold:
                continue
            x1 = max(0, int(raw[0, 0, i, 3] * w))
            y1 = max(0, int(raw[0, 0, i, 4] * h))
            x2 = min(w - 1, int(raw[0, 0, i, 5] * w))
            y2 = min(h - 1, int(raw[0, 0, i, 6] * h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2, confidence))
        return boxes

    def _predict_age(self, face_crop: np.ndarray) -> Tuple[str, float]:
        blob = cv2.dnn.blobFromImage(
            face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )
        self._age_net.setInput(blob)
        preds = self._age_net.forward()[0]
        idx = int(np.argmax(preds))
        return AGE_BUCKETS[idx], float(preds[idx])

    def analyze(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.7,
        padding: int = 20,
    ) -> ScanResult:
        """Detect every face in `frame` (BGR ndarray) and estimate its age."""
        if not self.ready:
            self.load()

        h, w = frame.shape[:2]
        raw_boxes = self._detect_faces(frame, confidence_threshold)

        detections: List[Detection] = []
        for i, (x1, y1, x2, y2, face_conf) in enumerate(raw_boxes):
            crop = frame[
                max(0, y1 - padding): min(h, y2 + padding),
                max(0, x1 - padding): min(w, x2 + padding),
            ]
            if crop.size == 0:
                continue
            age_bracket, age_conf = self._predict_age(crop)
            detections.append(
                Detection(
                    id=i + 1,
                    box=(x1, y1, x2, y2),
                    face_confidence=face_conf,
                    age_bracket=age_bracket,
                    age_confidence=age_conf,
                )
            )

        return ScanResult(detections=detections, image_size=(w, h))

    @staticmethod
    def draw(frame: np.ndarray, detections: List[Detection], color=(90, 166, 242)) -> np.ndarray:
        """Return a copy of `frame` (BGR) with boxes + labels drawn on it.

        Note: color is given in BGR (OpenCV convention), default is the
        Chronoscope amber accent (#F2A65A) expressed as BGR.
        """
        out = frame.copy()
        thickness = max(1, round(out.shape[0] / 250))
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"#{det.id}  {det.age_bracket}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                out, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2, cv2.LINE_AA,
            )
        return out
