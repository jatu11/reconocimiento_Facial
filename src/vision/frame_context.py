# src/vision/frame_context.py
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from src.vision.face_data import DetectedFace


@dataclass(slots=True)
class FrameContext:
    """
    Contenedor del estado completo de un fotograma.
    Se crea una sola vez por frame y es compartido
    por todos los módulos del motor.
    """
    frame: np.ndarray
    faces: list[DetectedFace] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    width: int = 0
    height: int = 0

    def __post_init__(self):
        self.height, self.width = self.frame.shape[:2]

    @property
    def face_count(self) -> int:
        return len(self.faces)