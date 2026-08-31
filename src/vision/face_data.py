# src/vision/face_data.py
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

BoundingBox = tuple[int, int, int, int]

@dataclass(slots=True)
class DetectedFace:
    """
    Representa un rostro detectado dentro de un FrameContext.
    Esta clase es completamente independiente de InsightFace y lógica de negocio.
    """
    # ------------------------------
    # Detección
    # ------------------------------
    bbox: BoundingBox
    score: float = 0.0
    landmarks: np.ndarray | None = None
    embedding: np.ndarray | None = None

    # ------------------------------
    # Tracking
    # ------------------------------
    track_id: int | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0

    # ------------------------------
    # Reconocimiento
    # ------------------------------
    identity_uuid: str = "unknown"
    confidence: float = 0.0
    recognition_state: str = "NEW"
    # Estados: NEW, PROCESSING, RECOGNIZED, UNKNOWN

    # ------------------------------
    # Atributos extra (Futuro)
    # ------------------------------
    age: int | None = None
    gender: str | None = None

    # ------------------------------------------------
    def __post_init__(self):
        now = time.time()
        if self.first_seen == 0:
            self.first_seen = now
        if self.last_seen == 0:
            self.last_seen = now

    # ------------------------------------------------
    @property
    def top(self) -> int:
        return self.bbox[0]

    @property
    def right(self) -> int:
        return self.bbox[1]

    @property
    def bottom(self) -> int:
        return self.bbox[2]

    @property
    def left(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[int, int]:
        return (
            self.left + self.width // 2,
            self.top + self.height // 2,
        )