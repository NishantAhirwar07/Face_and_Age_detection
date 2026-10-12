"""
Face and Age Detection using OpenCV and Caffe deep learning models.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Optional


class FaceAgeDetector:
    """Detects faces and predicts age groups using pre-trained Caffe models."""

    def __init__(self):
        """Initialize the detector with model paths."""
        self._face_net = None
        self._age_net = None
        self._age_list = [
            "(0-10)",
            "(10-20)",
            "(20-30)",
            "(30-40)",
            "(40-50)",
            "(50-60)",
            "(60-70)",
            "(70+)",
        ]
        self._model_mean = np.array([78.4263377603, 87.7689143744, 114.895847746])

    def load(self) -> None:
        """Load face and age detection models from Caffe format files."""
        base_dir = os.path.dirname(__file__)

        # Face detection model
        face_proto = os.path.join(
            base_dir, "models", "deploy_face.prototxt"
        )
        face_model = os.path.join(base_dir, "models", "res10_300x300_ssd_iter_140000.caffemodel")

        # Age detection model
        age_proto = os.path.join(
            base_dir, "models", "deploy_age.prototxt"
        )
        age_model = os.path.join(base_dir, "models", "age_net.caffemodel")

        # Validate model files exist
        required_files = [
            (face_proto, "Face detection prototxt"),
            (face_model, "Face detection model"),
            (age_proto, "Age detection prototxt"),
            (age_model, "Age detection model"),
        ]

        missing_files = [
            f"{name}: {path}" for path, name in required_files
            if not os.path.exists(path)
        ]

        if missing_files:
            raise FileNotFoundError(
                "Model files not found. Please place them in the 'models/' directory:\n"
                + "\n".join(missing_files)
                + "\n\nDownload from:\n"
                "- Face: https://github.com/sr6033/face-detection-opencv/raw/master/res10_300x300_ssd_iter_140000.caffemodel\n"
                "- Prototxt files: https://github.com/sr6033/face-detection-opencv/tree/master/opencv_face_detector_uint8.pb.txt\n"
            )

        try:
            self._face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
            self._age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
        except cv2.error as e:
            raise RuntimeError(
                f"Failed to load Caffe models. Ensure opencv-python-headless is installed.\n"
                f"Error: {e}"
            ) from e

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        if self._face_net is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), [104, 117, 123], False, False
        )
        self._face_net.setInput(blob)
        detections = self._face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x, y, width, height = x1, y1, x2 - x1, y2 - y1
                faces.append((x, y, width, height))

        return faces

    def predict_age(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Predict age group for a face image.

        Args:
            face_image: Cropped face image (BGR format)

        Returns:
            Tuple of (age_group_label, confidence)
        """
        if self._age_net is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        blob = cv2.dnn.blobFromImage(
            face_image, 1.0, (227, 227), self._model_mean, swapRB=False
        )
        self._age_net.setInput(blob)
        age_predictions = self._age_net.forward()
        age_idx = age_predictions[0].argmax()
        age_confidence = age_predictions[0][age_idx]

        return self._age_list[age_idx], float(age_confidence)

    def detect_and_predict(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """
        Detect faces and predict their ages in one call.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            List of tuples: ((x, y, w, h), age_label, confidence)
        """
        faces = self.detect_faces(image)
        results = []

        for x, y, w, h in faces:
            face_roi = image[y : y + h, x : x + w]
            if face_roi.size == 0:
                continue

            age_label, confidence = self.predict_age(face_roi)
            results.append(((x, y, w, h), age_label, confidence))

        return results
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """Detect faces in an image."""
    if self._face_net is None:
        raise RuntimeError("Model not loaded. Call load() first.")
    
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    self._face_net.setInput(blob)
    detections = self._face_net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:  # ← Use parameter
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x, y, width, height = x1, y1, x2 - x1, y2 - y1
            faces.append((x, y, width, height))
    
    return faces

def detect_and_predict(
    self, image: np.ndarray, confidence_threshold: float = 0.5
) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """Detect faces and predict their ages."""
    faces = self.detect_faces(image, confidence_threshold)  # ← Pass threshold
    results = []
    for x, y, w, h in faces:
        face_roi = image[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue
        age_label, confidence = self.predict_age(face_roi)
        results.append(((x, y, w, h), age_label, confidence))
    return results
