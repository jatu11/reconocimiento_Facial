# src/vision/preprocessing.py

from __future__ import annotations

import cv2
import numpy as np

BoundingBox = tuple[int, int, int, int]


class FacePreprocessor:
    """
    Utilidades para preparar un rostro antes de generar el embedding.

    Este módulo NO depende de InsightFace.
    Solo manipula imágenes.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (112, 112),
        padding: float = 0.20,
    ) -> None:
        self.target_size = target_size
        self.padding = padding

    def expand_bbox(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int, int],
    ) -> BoundingBox:
        """
        Expande ligeramente el bounding box para incluir más contexto
        alrededor del rostro.
        """

        top, right, bottom, left = bbox

        h = bottom - top
        w = right - left

        pad_y = int(h * self.padding)
        pad_x = int(w * self.padding)

        top -= pad_y
        bottom += pad_y

        left -= pad_x
        right += pad_x

        height, width = frame_shape[:2]

        top = max(0, top)
        left = max(0, left)

        bottom = min(height, bottom)
        right = min(width, right)

        return top, right, bottom, left

    def crop_face(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> np.ndarray:

        top, right, bottom, left = bbox

        return frame[top:bottom, left:right]

    def resize_face(
        self,
        face: np.ndarray,
    ) -> np.ndarray:

        return cv2.resize(
            face,
            self.target_size,
            interpolation=cv2.INTER_LINEAR,
        )

    def normalize_face(
        self,
        face: np.ndarray,
    ) -> np.ndarray:

        face = face.astype(np.float32)

        face /= 255.0

        return face

    def prepare_face(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> np.ndarray:
        """
        Pipeline completo.
        """

        bbox = self.expand_bbox(
            bbox,
            frame.shape,
        )

        face = self.crop_face(
            frame,
            bbox,
        )

        if face.size == 0:
            raise ValueError("Face crop vacío.")

        face = self.resize_face(face)

        face = self.normalize_face(face)

        return face
