from abc import ABC, abstractmethod

import numpy as np


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        pass


class BaseFaceRecognizer(ABC):
    @abstractmethod
    def recognize(
        self,
        frame: np.ndarray,
        face_locations: list[tuple[int, int, int, int]],
    ) -> list[tuple[str, float]]:
        pass

    @abstractmethod
    def extract_embeddings(
        self,
        frame: np.ndarray,
        face_locations: list[tuple[int, int, int, int]],
    ) -> list[list]:
        pass
