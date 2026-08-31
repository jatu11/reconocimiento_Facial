import logging
import time
from datetime import datetime
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EventManager:
    """
    Gestiona la cola de eventos locales de asistencia.
    Ahora utiliza un seguimiento DIARIO para evitar registros duplicados por cambios de jornada.
    """
    def __init__(self):
        self.event_queue: list[dict[str, Any]] = []
        self.registered_today: set[str] = set()
        self.current_day: str = ""

    def _get_current_block(self) -> str:
        # Mantenemos esta función solo para enviar el dato a Firebase (para estadísticas)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        
        if 0 <= now.hour < 12:
            shift = "MATUTINO"
        elif 12 <= now.hour < 18:
            shift = "VESPERTINO"
        else:
            shift = "NOCTURNO"
            
        return f"{date_str}_{shift}"

    def _get_current_day_string(self) -> str:
        """Devuelve la fecha actual para reiniciar la memoria a la medianoche."""
        return datetime.now().strftime("%Y-%m-%d")

    def register_recognition(self, identity_uuid: str, camera_id: str) -> None:
        if identity_uuid == "Calculando...":
            return

        active_day = self._get_current_day_string()
        active_block = self._get_current_block()
        
        # Limpiamos la memoria de rostros SOLO cuando cambia el DÍA (a la medianoche)
        if active_day != self.current_day:
            self.current_day = active_day
            self.registered_today.clear()
            logging.info(f"--- [NUEVO DÍA: {self.current_day}] Memoria de detecciones reiniciada ---")

        # La clave ahora depende del DÍA, no de la franja horaria
        event_key = f"{camera_id}_{identity_uuid}_{self.current_day}"
        
        # Para desconocidos, generamos una clave temporal cada 10 segundos
        if identity_uuid in ("unknown", "Desconocido"):
            event_key = f"{camera_id}_unknown_{int(time.time() // 10)}"

        if event_key not in self.registered_today:
            event = {
                "identity_uuid": identity_uuid,
                "camera_id": camera_id,
                "timestamp": time.time(),
                "block": active_block, # Se envía a Firebase pero no afecta la memoria
                "status": "pending"
            }
            self.event_queue.append(event)
            self.registered_today.add(event_key)
            
            logging.info(f"[EVENTO EN COLA] Nueva detección física de: {identity_uuid} en {camera_id}")

    def get_pending_events(self) -> list[dict[str, Any]]:
        return self.event_queue
        
    def clear_events(self, count: int) -> None:
        self.event_queue = self.event_queue[count:]