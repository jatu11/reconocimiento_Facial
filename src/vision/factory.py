from typing import Any

from src.utils.config import INSIGHTFACE_REC_THRESH
from src.vision.recognition_engine import RecognitionEngine
from src.vision.tracker import FaceTracker
from src.vision.vision_engine import VisionEngine


def get_vision_engine() -> VisionEngine:
    """Instancia el motor principal de visión (único acceso a InsightFace)."""
    return VisionEngine()


def get_face_tracker() -> FaceTracker:
    """Instancia el tracker de rostros."""
    return FaceTracker()


def get_recognition_engine(
    known_encodings: list[Any],
    known_names: list[str],
) -> RecognitionEngine:
    """Instancia el motor de reconocimiento por embeddings."""
    return RecognitionEngine(
        known_encodings=known_encodings,
        known_names=known_names,
        threshold=INSIGHTFACE_REC_THRESH,
    )
