# src/storage/file_manager.py
import logging
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FileManager:
    """Clase responsable de la gestión física de directorios y archivos de imagen."""

    @staticmethod
    def create_person_directory(base_dir: Path | str, name: str) -> Path:
        base_dir = Path(base_dir)

        normalized_name = name.strip().replace(" ", "_")
        target_path = base_dir / normalized_name

        if not target_path.exists():
            target_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Directorio creado exitosamente en: {target_path}")
        else:
            logging.info(
                f"El directorio ya existe: {target_path}. Se añadirán imágenes al conjunto actual."
            )

        return target_path

    @staticmethod
    def is_blurry(frame: np.ndarray, threshold: float) -> bool:
        """
        Calcula la varianza del Laplaciano sobre el fotograma en escala de grises.
        Una varianza baja indica una falta de bordes definidos (imagen borrosa).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        focus_measure = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return focus_measure < threshold

    @staticmethod
    def save_frame(directory: Path | str, frame: np.ndarray, photo_index: int) -> bool:
        directory = Path(directory)

        import uuid

        unique_suffix = uuid.uuid4().hex[:8]
        filename = f"face_{photo_index:03d}_{unique_suffix}.png"

        full_path = directory / filename

        success = cv2.imwrite(str(full_path), frame)
        if success:
            logging.info(f"Fotografía guardada: {full_path}")
        else:
            logging.error(f"Error crítico al intentar guardar: {full_path}")

        return success

    @staticmethod
    def get_dataset_directories(base_dir: Path | str) -> list[Path]:
        base_dir = Path(base_dir)

        if not base_dir.exists():
            return []

        return [p for p in base_dir.iterdir() if p.is_dir()]

    @staticmethod
    def save_model(data: dict[str, Any], file_path: Path | str) -> bool:
        file_path = Path(file_path)
        """Serializa el diccionario de encodings y etiquetas y lo guarda en disco."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            logging.info(f"Modelo guardado exitosamente en: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error crítico al guardar el modelo: {e}")
            return False

    @staticmethod
    def load_model(file_path: Path | str) -> dict[str, Any]:
        file_path = Path(file_path)
        """Carga en memoria el modelo de encodings previamente entrenado."""
        if not file_path.exists():
            logging.error(f"Archivo de modelo no encontrado en: {file_path}")
            return {}

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            logging.info("Modelo de reconocimiento cargado exitosamente.")
            return data
        except Exception as e:
            logging.error(f"Error crítico al leer el archivo del modelo: {e}")
            return {}
