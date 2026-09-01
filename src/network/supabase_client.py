import logging
import time
import os
from datetime import datetime
import requests
from dotenv import load_dotenv

# Carga las variables de entorno del archivo .env
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SupabaseClient:
    """
    Cliente REST puro para comunicarse directamente con la API de Supabase.
    """
    def __init__(self):
        # TODO: Reemplaza con tus credenciales de Supabase (Project Settings -> API)
        self.base_url = os.getenv("SUPABASE_URL")
        self.api_key = os.getenv("SUPABASE_KEY")
        
        self.headers = {
            "apikey": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        self.is_connected = True
        logging.info("[SUPABASE] Cliente REST inicializado.")

    def set_camera_status(self, camera_id: str, is_active: bool, ubicacion=None, nombre_camara=None):
        """Hace un UPSERT (Update o Insert) del estado de la cámara."""
        payload = {
            "id": camera_id,
            "activa": is_active
        }
        if nombre_camara:
            payload["nombre"] = nombre_camara
        if ubicacion:
            payload["ubicacion"] = str(ubicacion)

        url = f"{self.base_url}/camaras"
        headers = self.headers.copy()
        # Indicamos a Supabase que haga un UPSERT basado en la llave primaria
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"

        try:
            requests.post(url, headers=headers, json=payload)
            estado_str = "ENCENDIDA" if is_active else "APAGADA"
            logging.info(f"[SUPABASE] Cámara {camera_id} actualizada a: {estado_str}")
        except Exception as e:
            logging.error(f"[ERROR] No se pudo actualizar estado de cámara {camera_id}: {e}")

    def registrar_asistencia_clase(self, hora_clase: str, datos_asistencia: dict):
        """
        Descarga la relación de estudiantes y envía en lote (batch insert) la asistencia.
        Calcula estados: Presente, Falta, Fugado e Intruso.
        """
        try:
            hoy = datetime.now().strftime("%Y-%m-%d")
            
            # 1. Traer datos base y el historial de HOY para detectar fugados
            req_cursos = requests.get(f"{self.base_url}/cursos?select=id,camara_id", headers=self.headers)
            req_estudiantes = requests.get(f"{self.base_url}/estudiantes?select=cedula,curso_id", headers=self.headers)
            req_asistencia_hoy = requests.get(f"{self.base_url}/asistencia?fecha=eq.{hoy}&estado=eq.Presente", headers=self.headers)
            
            cursos = req_cursos.json() if req_cursos.status_code == 200 else []
            estudiantes = req_estudiantes.json() if req_estudiantes.status_code == 200 else []
            asistencia_hoy = req_asistencia_hoy.json() if req_asistencia_hoy.status_code == 200 else []
            
            # Mapeos en memoria para cruce ultrarrápido
            cam_to_curso = {c["camara_id"]: c["id"] for c in cursos if c.get("camara_id")}
            curso_to_estudiantes = {}
            for e in estudiantes:
                curso_to_estudiantes.setdefault(e["curso_id"], []).append(e["cedula"])
                
            # Set de cédulas que ya vinieron a clases hoy (para marcar fugados)
            presentes_previos = {registro["estudiante_cedula"] for registro in asistencia_hoy}

            asistencia_batch = []

            # 2. Lógica deductiva
            for cam_id, detected_list in datos_asistencia.items():
                curso_id = cam_to_curso.get(cam_id)
                if not curso_id:
                    continue
                
                enrolled_cedulas = curso_to_estudiantes.get(curso_id, [])
                detected_cedulas = [d["uuid"] for d in detected_list if d["uuid"] not in ["unknown", "Desconocido"]]
                
                # Evaluar matriculados (Presentes, Faltas, Fugados)
                for cedula in enrolled_cedulas:
                    if cedula in detected_cedulas:
                        estado = "Presente"
                    elif cedula in presentes_previos:
                        estado = "Fugado"
                    else:
                        estado = "Falta"
                        
                    asistencia_batch.append({
                        "estudiante_cedula": cedula,
                        "curso_id": curso_id,
                        "fecha": hoy,
                        "hora_clase": hora_clase,
                        "estado": estado
                    })
                
                # Evaluar Intrusos
                for d in detected_list:
                    cedula = d["uuid"]
                    if cedula not in enrolled_cedulas and cedula not in ["unknown", "Desconocido"]:
                        asistencia_batch.append({
                            "estudiante_cedula": cedula,
                            "curso_id": curso_id,
                            "fecha": hoy,
                            "hora_clase": hora_clase,
                            "estado": "Intruso"
                        })

            # 3. Enviar todo en una sola petición (Batch UPSERT HTTP POST)
            if asistencia_batch:
                # Añadimos on_conflict para indicar qué columnas forman la regla de duplicados
                insert_url = f"{self.base_url}/asistencia?on_conflict=estudiante_cedula,fecha,hora_clase"
                headers = self.headers.copy()
                
                # merge-duplicates actúa como un UPSERT: actualiza si ya existe, inserta si es nuevo
                headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
                
                res = requests.post(insert_url, headers=headers, json=asistencia_batch)
                if res.status_code in (201, 204):
                    logging.info(f"[FERIA] Asistencia registrada/actualizada exitosamente para {hora_clase}.")
                    return True
                else:
                    logging.error(f"[ERROR] Supabase rechazó el registro: {res.text}")
                    return False
            else:
                logging.info(f"[FERIA] No hubo datos válidos para registrar en {hora_clase}.")
                return True
                
        except Exception as e:
            logging.error(f"[FERIA] Error al registrar asistencia por clase: {e}")
            return False