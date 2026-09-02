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
        Las cámaras actúan de forma 100% independiente.
        Cada cámara procesa su lista y la envía a la base de datos por separado.
        """
        try:
            hoy = datetime.now().strftime("%Y-%m-%d")
            
            # 1. Traer datos base
            req_cursos = requests.get(f"{self.base_url}/cursos?select=id,camara_id", headers=self.headers)
            req_estudiantes = requests.get(f"{self.base_url}/estudiantes?select=cedula,curso_id", headers=self.headers)
            req_asistencia_hoy = requests.get(f"{self.base_url}/asistencia?fecha=eq.{hoy}&estado=eq.Presente", headers=self.headers)
            
            cursos = req_cursos.json() if req_cursos.status_code == 200 else []
            estudiantes = req_estudiantes.json() if req_estudiantes.status_code == 200 else []
            asistencia_hoy = req_asistencia_hoy.json() if req_asistencia_hoy.status_code == 200 else []
            
            cam_to_curso = {c["camara_id"]: c["id"] for c in cursos if c.get("camara_id")}
            curso_to_estudiantes = {}
            todas_las_cedulas = set()
            
            for e in estudiantes:
                curso_to_estudiantes.setdefault(e["curso_id"], []).append(e["cedula"])
                todas_las_cedulas.add(e["cedula"])
                
            presentes_previos = {registro["estudiante_cedula"] for registro in asistencia_hoy}

            # 2. PROCESAR Y ENVIAR POR SEPARADO CADA CÁMARA
            for cam_id, detected_list in datos_asistencia.items():
                curso_id = cam_to_curso.get(cam_id)
                if not curso_id:
                    continue
                
                asistencia_camara = [] # Lista exclusiva para esta cámara
                detected_cedulas = set(d["uuid"] for d in detected_list if d["uuid"] not in ["unknown", "Desconocido"])
                enrolled_cedulas = curso_to_estudiantes.get(curso_id, [])
                
                # A. Evaluar matriculados de ESTA cámara
                for cedula in enrolled_cedulas:
                    if cedula in detected_cedulas:
                        estado = "Presente"
                    elif cedula in presentes_previos:
                        estado = "Fugado"
                    else:
                        estado = "Falta"
                        
                    asistencia_camara.append({
                        "estudiante_cedula": cedula,
                        "curso_id": curso_id,
                        "fecha": hoy,
                        "hora_clase": hora_clase,
                        "estado": estado
                    })
                
                # B. Evaluar Intrusos de ESTA cámara
                for cedula in detected_cedulas:
                    if cedula not in todas_las_cedulas:
                        continue # Bloquear fantasmas
                        
                    if cedula not in enrolled_cedulas:
                        asistencia_camara.append({
                            "estudiante_cedula": cedula,
                            "curso_id": curso_id,
                            "fecha": hoy,
                            "hora_clase": hora_clase,
                            "estado": "Intruso"
                        })

                # C. ENVIAR INMEDIATAMENTE A SUPABASE (Solo los datos de esta cámara)
                if asistencia_camara:
                    insert_url = f"{self.base_url}/asistencia?on_conflict=estudiante_cedula,fecha,hora_clase,curso_id"
                    headers = self.headers.copy()
                    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
                    
                    res = requests.post(insert_url, headers=headers, json=asistencia_camara)
                    if res.status_code in (200, 201, 204):
                        logging.info(f"[FERIA] {cam_id} registró su asistencia en {hora_clase} exitosamente.")
                    else:
                        logging.error(f"[ERROR] Falló {cam_id}: {res.text}")

            return True 
                
        except Exception as e:
            logging.error(f"[FERIA] Error al registrar asistencia: {e}")
            return False