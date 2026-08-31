# src/utils/config.py
from pathlib import Path

# --- RUTAS Y DIRECTORIOS BASE ---
# BASE_DIR apunta a la raíz del proyecto (2 niveles arriba de config.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Carpetas de almacenamiento local (Edge)
DATASET_DIR = str(BASE_DIR / "data" / "dataset")
MODEL_PATH = str(BASE_DIR / "data" / "models" / "encodings.pkl")

# --- CONFIGURACIÓN DE CÁMARAS ---
# El motor Edge solo necesita saber el ID de la cámara y su fuente.
# Las reglas de a qué curso pertenece se evalúan en el Backend en la nube.
CAMERA_SOURCES = [
    {
        "camera_id": "CAM_001",
        "nombre": "CAMARA 1",
        "src": 0, # 0 para la cámara web principal de la laptop
        "ubicacion": {
            "latitude": -2.128589, 
            "longitude": -79.931099
        }
    },
    {
        "camera_id": "CAM_002",
        "nombre": "CAMARA 2",
        "src": 1, # 1 para una segunda cámara USB, o "http://IP_DE_CAMARA/video" para cámaras web
        "ubicacion": {
            "latitude": -2.128650, 
            "longitude": -79.931200
        }
    }
]

RECONNECT_DELAY_SECONDS = 2
MAX_PHOTOS_PER_PERSON = 30
BLUR_THRESHOLD = 70.0

# --- CONFIGURACIÓN INSIGHTFACE (SCRFD + ARCFACE) ---
INSIGHTFACE_MODEL_PACK = "buffalo_l"
INSIGHTFACE_DET_THRESH = 0.5
INSIGHTFACE_INPUT_SIZE = (320, 320)
INSIGHTFACE_EMBEDDING_SIZE = 512
INSIGHTFACE_REC_THRESH = 0.45

# --- CONFIGURACIÓN DEL TRACKER (BYTETRACK) ---
TRACKER_BUFFER = 30
TRACKER_MATCH_THRESH = 0.8