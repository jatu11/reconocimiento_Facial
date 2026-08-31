import logging
import os
import threading
import time
from datetime import datetime
from typing import Any

import firebase_admin
from firebase_admin import credentials, db, messaging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FirebaseClient:
    """
    Gestiona Firebase RTDB y las notificaciones Push (Entrada, Salida y Faltas).
    Incluye rutina automática de faltas y corrección a estado "Atrasado".
    """
    def __init__(self, database_url: str = "https://motor-c7e0d-default-rtdb.firebaseio.com"):
        self.database_url = database_url
        self.is_connected = False
        
        # Diccionario para controlar el estado de las notificaciones hoy.
        self.daily_notifications: dict[str, dict[str, dict[str, Any]]] = {}
        
        self.horas_minimas_para_salida = 4.0 
        
        self.horarios_entrada = {
            "MATUTINO": "07:00",
            "VESPERTINO": "13:00",
            "NOCTURNO": "18:00"
        }
        
        if not firebase_admin._apps:
            cred_path = "serviceAccountKey.json"
            if not os.path.exists(cred_path):
                logging.error(f"[FIREBASE] ERROR CRÍTICO: No se encontró el archivo {cred_path}.")
                return

            try:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': self.database_url
                })
                self.is_connected = True
                logging.info("[FIREBASE] Conexión a Realtime Database y FCM establecida con éxito.")
            except Exception as e:
                logging.error(f"[FIREBASE] Error al inicializar: {e}")
        else:
            self.is_connected = True
            
        if self.is_connected:
            self.db_ref = db.reference('SesionesCamara')
            self.estudiantes_ref = db.reference('Estudiantes')
            self.usuarios_ref = db.reference('Usuarios')

    def iniciar_rutina_faltas_automatica(self, hora_check: str = "10:00"):
        """
        Inicia un hilo en segundo plano que vigila el reloj y dispara la evaluación de inasistencias.
        """
        def _rutina():
            logging.info(f"[SISTEMA] Reloj automático de faltas iniciado. Evaluación programada a las {hora_check}.")
            while True:
                ahora = datetime.now().strftime("%H:%M")
                if ahora == hora_check:
                    self.procesar_inasistencias()
                    time.sleep(61)  # Dormir 61 segundos evita que se ejecute múltiples veces en el mismo minuto
                time.sleep(30)  # Verificar cada 30 segundos
        
        hilo = threading.Thread(target=_rutina, daemon=True)
        hilo.start()

    def send_event(self, event_data: dict[str, Any], modo_operacion: str = "ENTRADA") -> bool:
        if not self.is_connected:
            return False
            
        try:
            cam_id = event_data.get("camera_id", "CAM_DEFAULT")
            uuid = event_data.get("identity_uuid", "unknown")
            timestamp_evento = event_data.get("timestamp", time.time())
            bloque_horario = event_data.get("block", "MATUTINO")
            
            cam_ref = self.db_ref.child(cam_id)
            cam_ref.update({
                "activo": True,
                "estado": "activa",
                "camara_nombre": f"{cam_id} ({modo_operacion})"
            })
            
            es_registrado = uuid not in ["unknown", "Desconocido", "Calculando..."]
            registro = {
                "timestamp": timestamp_evento,
                "bloque_horario": bloque_horario,
                "tipo_evento": modo_operacion
            }
            
            nombre_estudiante = uuid

            if es_registrado:
                estudiante_data = self.estudiantes_ref.child(uuid).get()
                if estudiante_data:
                    nombre_estudiante = estudiante_data.get('nombre', uuid)
                    rep_uid = estudiante_data.get('representante_uid')
                    
                    if rep_uid:
                        self._evaluar_reglas_y_notificar(rep_uid, nombre_estudiante, cam_id, uuid, bloque_horario, timestamp_evento, modo_operacion)

                registro["total_presentes"] = 1
                registro["total_desconocidos"] = 0
                registro["lista_presentes"] = {
                    uuid: {"id": uuid, "nombre": nombre_estudiante, "registrado": True}
                }
            else:
                nombre_estudiante = "Rostro No Reconocido"
                registro["total_presentes"] = 0
                registro["total_desconocidos"] = 1
                registro["lista_intrusos"] = {
                    "intr_temp": {"id": "desconocido", "nombre": nombre_estudiante, "registrado": False}
                }
            
            cam_ref.child("RegistroDiario").push(registro)
            logging.info(f"[FIREBASE] Evento de {modo_operacion} para {nombre_estudiante} guardado.")
            return True
            
        except Exception as e:
            logging.error(f"[FIREBASE] Error al escribir en base de datos: {e}")
            return False

    def _evaluar_reglas_y_notificar(self, rep_uid: str, nombre_estudiante: str, cam_id: str, uuid: str, bloque_horario: str, timestamp_evento: float, modo_operacion: str):
        hoy = datetime.fromtimestamp(timestamp_evento).strftime("%Y-%m-%d")
        
        if hoy not in self.daily_notifications:
            self.daily_notifications.clear()
            self.daily_notifications[hoy] = {}

        if uuid not in self.daily_notifications[hoy]:
            self.daily_notifications[hoy][uuid] = {}

        historial = self.daily_notifications[hoy][uuid]
        tipo_notificacion = None
        estado_entrada = None

        # REGLA PARA ENTRADA
        if modo_operacion == "ENTRADA":
            # Si no ha entrado antes, procesamos la llegada
            if "entrada" not in historial:
                historial["entrada"] = timestamp_evento
                
                # Si existía una falta previa, la anulamos para corregir el estado a Atrasado
                if "falta" in historial:
                    historial.pop("falta")
                    logging.info(f"[SISTEMA] Falta anulada para {nombre_estudiante}. Actualizando a llegada con atraso.")
                
                jornada = bloque_horario.split("_")[-1] if "_" in bloque_horario else "MATUTINO"
                hora_esperada_str = self.horarios_entrada.get(jornada, "07:00")
                
                hora_esperada_dt = datetime.strptime(hora_esperada_str, "%H:%M").time()
                hora_real_dt = datetime.fromtimestamp(timestamp_evento).time()
                
                if hora_real_dt > hora_esperada_dt:
                    estado_entrada = "ATRASADO"
                else:
                    estado_entrada = "PUNTUAL"
                    
                tipo_notificacion = "ENTRADA"
                
        # REGLA PARA SALIDA DEFINITIVA
        elif modo_operacion == "SALIDA":
            if "entrada" in historial and "salida" not in historial and "falta" not in historial:
                historial["salida"] = timestamp_evento
                tipo_notificacion = "SALIDA"

        if tipo_notificacion:
            self._disparar_push(rep_uid, nombre_estudiante, tipo_notificacion, timestamp_evento, estado_entrada)

    def procesar_inasistencias(self):
        if not self.is_connected:
            return
            
        logging.info("[SISTEMA] Evaluando inasistencias del día...")
        hoy = datetime.now().strftime("%Y-%m-%d")
        timestamp_actual = time.time()
        
        if hoy not in self.daily_notifications:
            self.daily_notifications[hoy] = {}

        try:
            todos_los_estudiantes = self.estudiantes_ref.get()
            if not todos_los_estudiantes:
                return

            for uuid, data in todos_los_estudiantes.items():
                rep_uid = data.get('representante_uid')
                nombre_estudiante = data.get('nombre', uuid)
                
                if uuid not in self.daily_notifications[hoy]:
                    self.daily_notifications[hoy][uuid] = {}
                
                historial = self.daily_notifications[hoy][uuid]
                
                if "entrada" not in historial and "falta" not in historial:
                    historial["falta"] = True
                    
                    if rep_uid:
                        self._disparar_push(rep_uid, nombre_estudiante, "FALTA", timestamp_actual)
                        
            logging.info("[SISTEMA] Procesamiento de inasistencias finalizado con éxito.")
        except Exception as e:
            logging.error(f"[SISTEMA] Error al procesar inasistencias: {e}")

    def _disparar_push(self, rep_uid: str, nombre_estudiante: str, tipo: str, timestamp_evento: float, estado_entrada: str = None):
        try:
            usuario_data = self.usuarios_ref.child(rep_uid).get()
            if not usuario_data:
                return
            
            fcm_token = usuario_data.get('fcm_token')
            hora_real = datetime.fromtimestamp(timestamp_evento).strftime("%H:%M")
            
            if fcm_token:
                if tipo == "ENTRADA":
                    if estado_entrada == "ATRASADO":
                        titulo = '⚠️ Ingreso con Atraso'
                        cuerpo = f'Su representado {nombre_estudiante} llegó atrasado a la institución a las {hora_real}.'
                    else:
                        titulo = '✅ Ingreso Confirmado'
                        cuerpo = f'Su representado {nombre_estudiante} llegó a la institución a las {hora_real}.'
                elif tipo == "SALIDA":
                    titulo = '👋 Salida de la Institución'
                    cuerpo = f'Su representado {nombre_estudiante} salió de la institución a las {hora_real}.'
                elif tipo == "FALTA":
                    titulo = '❌ Inasistencia Registrada'
                    cuerpo = f'Su representado {nombre_estudiante} no asistió a la institución el día de hoy.'

                mensaje = messaging.Message(
                    notification=messaging.Notification(title=titulo, body=cuerpo),
                    token=fcm_token,
                )
                response = messaging.send(mensaje)
                logging.info(f"[FCM] Notificación de {tipo} enviada para {nombre_estudiante}. ID: {response}")
            else:
                logging.info(f"[FCM] Omitiendo {tipo} para {nombre_estudiante}: Representante sin Token configurado.")
                
        except Exception as e:
            logging.error(f"[FCM] Error al enviar notificación Push: {e}")



    def set_camera_status(self, camera_id, is_active, ubicacion=None, nombre_camara=None):
        """
        Actualiza el estado de conexión, nombre y ubicación de la cámara en Firebase.
        """
        try:
            from firebase_admin import db
            ref = db.reference(f'SesionesCamara/{camera_id}')
            
            # Datos básicos de estado
            datos = {
                'activo': is_active,
                'estado': 'Activa' if is_active else 'Desconectada',
                'status': 'online' if is_active else 'offline'
            }
            
            # Solo actualizamos el nombre y la ubicación cuando la cámara se enciende
            if is_active:
                if nombre_camara:
                    datos['camara_nombre'] = nombre_camara
                if ubicacion:
                    datos['ubicacion'] = ubicacion
                    
            ref.update(datos)
            estado_str = "ENCENDIDA" if is_active else "APAGADA"
            print(f"[FIREBASE] Estado de la cámara {camera_id} actualizado a: {estado_str}")
        except Exception as e:
            print(f"[ERROR] No se pudo actualizar el estado de la cámara {camera_id}: {e}")

    def registrar_asistencia_clase(self, hora_clase: str, datos_asistencia: dict):
        """
        Registra la asistencia en bloque por hora de clase (Modo Feria).
        Cruza los datos de las cámaras con los estudiantes de bachillerato 
        registrados en cada curso para deducir quién faltó.
        """
        if not self.is_connected:
            logging.warning("[FERIA] No hay conexión a Firebase. Omitiendo registro.")
            return
            
        try:
            hoy = datetime.now().strftime("%Y-%m-%d")
            
            # 1. Obtenemos la estructura de los cursos de Firebase
            cursos_ref = db.reference("Cursos").get() or {}
            
            # 2. Mapeamos qué cámara pertenece a qué curso y quiénes están matriculados
            cam_to_curso = {}
            for curso_id, curso_data in cursos_ref.items():
                cam_id = curso_data.get("camara_id")
                enrolled = curso_data.get("estudiantes_registrados", {})
                if cam_id:
                    cam_to_curso[cam_id] = {
                        "curso_id": curso_id,
                        "enrolled": list(enrolled.keys())
                    }
                    
            asistencia_updates = {}
            
            # 3. Procesamos los rostros enviados desde la GUI (track_cache)
            for cam_id, detected_list in datos_asistencia.items():
                info_curso = cam_to_curso.get(cam_id)
                if not info_curso:
                    continue  # Esta cámara no tiene un curso asignado en la BD
                    
                curso_id = info_curso["curso_id"]
                enrolled_uuids = info_curso["enrolled"]
                
                # Obtenemos solo los UUIDs válidos detectados por InsightFace
                detected_uuids = [d["uuid"] for d in detected_list if d["uuid"] not in ["unknown", "Desconocido"]]
                
                ruta_base = f"Asistencia/{hoy}/{curso_id}/{hora_clase}"
                
                # 4. Evaluar a los estudiantes matriculados
                for uuid in enrolled_uuids:
                    if uuid in detected_uuids:
                        estado = "Presente"
                    else:
                        # Si está en el curso pero no frente a la cámara, le marcamos falta para esa hora
                        estado = "Falta"
                        
                    asistencia_updates[f"{ruta_base}/{uuid}"] = {
                        "timestamp": time.time(),
                        "estado": estado,
                        "metodo": "Reconocimiento Facial Edge"
                    }
                
                # 5. Registrar "Visitas/Intrusos" (Estudiantes detectados que no pertenecen a ese curso)
                for d in detected_list:
                    uuid = d["uuid"]
                    if uuid not in enrolled_uuids and uuid not in ["unknown", "Desconocido"]:
                        asistencia_updates[f"{ruta_base}/{uuid}"] = {
                            "timestamp": time.time(),
                            "estado": "Visita / Fuera de Curso",
                            "metodo": "Reconocimiento Facial Edge",
                            "confianza_ia": d.get("confidence", 0.0)
                        }
            
            # 6. Enviamos todos los registros a Firebase en una sola petición rápida
            if asistencia_updates:
                db.reference().update(asistencia_updates)
                logging.info(f"[FERIA] Asistencia cruzada y registrada exitosamente para {hora_clase}.")
            else:
                logging.info(f"[FERIA] No hubo datos válidos para registrar en {hora_clase}.")
                
        except Exception as e:
            logging.error(f"[FERIA] Error crítico al registrar asistencia por clase: {e}")