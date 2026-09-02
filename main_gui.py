import os
import platform
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

if platform.system() == "Windows":
    import winsound

from pathlib import Path

from src.capture.camera_stream import CameraStream
from src.events.event_manager import EventManager
from src.storage.file_manager import FileManager
from src.training.trainer import ModelTrainer
from src.utils.config import (
    BLUR_THRESHOLD,
    CAMERA_SOURCES,
    DATASET_DIR,
    INSIGHTFACE_REC_THRESH,
    MAX_PHOTOS_PER_PERSON,
    MODEL_PATH,
    RECONNECT_DELAY_SECONDS,
)
from src.vision.recognition_engine import RecognitionEngine
from src.vision.tracker import FaceTracker
from src.vision.vision_engine import VisionEngine
from src.network.supabase_client import SupabaseClient

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Motor de Reconocimiento Facial Edge - Modo Feria")
        self.root.geometry("1100x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.running = True
        self.mode = "RECOGNIZE"
        self.identity_label = ""
        self.captured_photos = 0
        self.cooldown_time = 0.0
        self.current_imgtk = None

        self.active_camera_idx = 0
        self.view_mode = "SINGLE"
        self.streams = []
        
        # Variables para el temporizador de la feria
        self.last_auto_scan_time = time.time()

        # Optimización de rendimiento
        self.frame_counter = 0
        self.process_every_n_frames = 3
        self.last_faces = [] 
        self.last_reg_faces = [] 

        self.zoom_factor = tk.DoubleVar(value=1.0)
        self.pan_x = tk.DoubleVar(value=0.0)
        self.pan_y = tk.DoubleVar(value=0.0)
        
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.root.columnconfigure(0, weight=7)
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(0, weight=1)

        self.setup_ui()
        self.init_backend()
        self.update_frame()

    def setup_ui(self):
        self.video_frame = tk.Frame(self.root, bg="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True, fill="both")
        
        self.video_label.bind("<MouseWheel>", self.on_mouse_wheel)
        self.video_label.bind("<ButtonPress-1>", self.on_mouse_press)
        self.video_label.bind("<B1-Motion>", self.on_mouse_drag)

        self.control_frame = tk.Frame(self.root, bg="#2C3E50", padx=20, pady=20)
        self.control_frame.grid(row=0, column=1, sticky="nsew")

        lbl_title = tk.Label(self.control_frame, text="Panel de Control Edge", font=("Helvetica", 16, "bold"), bg="#2C3E50", fg="white")
        lbl_title.pack(pady=(0, 15))

        self.lbl_status = tk.Label(self.control_frame, text="Estado: Reconocimiento Activo", font=("Helvetica", 11), bg="#2C3E50", fg="#2ECC71")
        self.lbl_status.pack(pady=(0, 15))

        lbl_cams = tk.Label(self.control_frame, text="Selector de Cámara", font=("Helvetica", 10, "bold"), bg="#2C3E50", fg="#BDC3C7")
        lbl_cams.pack(pady=(5, 5))

        cam_options = [f"{cam.get('nombre', f'Cam {i}')} ({cam.get('camera_id', 'UNKNOWN')})" for i, cam in enumerate(CAMERA_SOURCES)]
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(self.control_frame, textvariable=self.cam_var, values=cam_options, state="readonly", font=("Helvetica", 10))
        self.cam_combo.pack(fill="x", pady=5)
        if cam_options:
            self.cam_combo.current(0)
        self.cam_combo.bind("<<ComboboxSelected>>", self.on_camera_select)
        
        btn_grid = tk.Button(self.control_frame, text="Vista General (Todas)", bg="#9B59B6", fg="white", command=self.show_grid_view)
        btn_grid.pack(fill="x", pady=(5, 15))

        # --- SECCIÓN NUEVA: CONTROLES DE DEMOSTRACIÓN ---
        marco_demo = tk.LabelFrame(self.control_frame, text=" Toma de Asistencia ", bg="#2C3E50", fg="#F1C40F", font=("Helvetica", 10, "bold"))
        marco_demo.pack(fill="x", pady=10, ipadx=5, ipady=5)

        lbl_hora = tk.Label(marco_demo, text="Hora Clase Actual", bg="#2C3E50", fg="white")
        lbl_hora.pack(pady=(5, 0))
        
        self.hora_var = tk.StringVar()
        horas_disponibles = [
            "Hora_1", "Hora_2", "Hora_3", "Hora_4", 
            "Recreo", "Hora_5", "Hora_6", "Hora_7", "Hora_8"
        ]
        self.hora_combo = ttk.Combobox(marco_demo, textvariable=self.hora_var, values=horas_disponibles, state="readonly")
        self.hora_combo.pack(fill="x", padx=10, pady=5)
        self.hora_combo.current(0)

        lbl_intervalo = tk.Label(marco_demo, text="Auto-Escaneo (Segundos)", bg="#2C3E50", fg="white")
        lbl_intervalo.pack(pady=(5, 0))

        self.intervalo_var = tk.IntVar(value=10)
        self.intervalo_combo = ttk.Combobox(marco_demo, textvariable=self.intervalo_var, values=[10, 30, 60, 300, 600], state="readonly")
        self.intervalo_combo.pack(fill="x", padx=10, pady=5)

        self.btn_tomar_lista = tk.Button(marco_demo, text="📸 Tomar Asistencia Ahora", font=("Helvetica", 11, "bold"), bg="#27AE60", fg="white", command=self.tomar_asistencia_demo)
        self.btn_tomar_lista.pack(fill="x", padx=10, pady=(10, 5), ipady=5)
        # --------------------------------------------------

        button_font = ("Helvetica", 11)
        self.btn_register = tk.Button(self.control_frame, text="Registrar Nuevo Usuario", font=button_font, bg="#3498DB", fg="white", command=self.start_registration)
        self.btn_register.pack(fill="x", pady=5, ipady=3)

        self.btn_train = tk.Button(self.control_frame, text="Actualizar Modelo", font=button_font, bg="#F39C12", fg="white", command=self.start_training)
        self.btn_train.pack(fill="x", pady=5, ipady=3)

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_factor.set(min(4.0, self.zoom_factor.get() + 0.1))
        elif event.delta < 0:
            self.zoom_factor.set(max(1.0, self.zoom_factor.get() - 0.1))

    def on_mouse_press(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_mouse_drag(self, event):
        if self.zoom_factor.get() <= 1.0:
            return
            
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        pan_speed = 0.003 / self.zoom_factor.get()
        new_pan_x = self.pan_x.get() - (dx * pan_speed)
        new_pan_y = self.pan_y.get() - (dy * pan_speed)

        self.pan_x.set(max(-1.0, min(1.0, new_pan_x)))
        self.pan_y.set(max(-1.0, min(1.0, new_pan_y)))

        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def init_backend(self):
        print("[INFO] Cargando modelo y motores de visión...")
        model = FileManager.load_model(Path(MODEL_PATH))
        known_encodings = model.get("encodings", [])
        known_names = model.get("names", [])

        self.vision_engine = VisionEngine()
        self.tracker = FaceTracker()
        self.recognition_engine = RecognitionEngine(
            known_encodings=known_encodings,
            known_names=known_names,
            threshold=INSIGHTFACE_REC_THRESH,
        )
        
        self.event_manager = EventManager()
        self.api_client = SupabaseClient() 

        # Hilo secundario para Firebase
        self.sender_thread = threading.Thread(target=self._event_sender_task, daemon=True)
        self.sender_thread.start()

        print("[INFO] Conectando a las cámaras...")
        for cam in CAMERA_SOURCES:
            cam_id = cam.get("camera_id", "CAM_DEFAULT")
            cam_nombre = cam.get("nombre", f"Cámara {cam_id}")
            cam_ubicacion = cam.get("ubicacion", {"latitude": -2.128589, "longitude": -79.931099})
            
            stream = CameraStream(
                source=cam["src"],
                camera_id=cam_id,
                reconnect_delay=RECONNECT_DELAY_SECONDS,
            )
            self.streams.append(stream)
            self.api_client.set_camera_status(cam_id, True, ubicacion=cam_ubicacion, nombre_camara=cam_nombre)

    def _event_sender_task(self):
        # Mantiene viva la cola de eventos por si se usa en el futuro,
        # aunque ahora dependemos del lote (batch) de asistencia.
        while self.running:
            time.sleep(1)


    def tomar_asistencia_demo(self, auto=False):
        hora_actual = self.hora_var.get()
        modo_str = "Automático" if auto else "Manual"
        print(f"\n[DEMO] 📸 Capturando asistencia ({modo_str}) para: {hora_actual}")
        
        # 1. Indicador visual de que está procesando
        self.update_ui_state(f"Enviando Asistencia: {hora_actual}...", "#F1C40F") # Amarillo
        
        # 2. Recolectar rostros activos
        asistencia_por_camara = {}
        for cache_key, data in self.recognition_engine.track_cache.items():
            uuid = data.get("identity_uuid", "unknown")
            if uuid not in ["unknown", "Desconocido", "Calculando..."]:
                partes = cache_key.split("_")
                if len(partes) >= 2:
                    cam_id = f"{partes[0]}_{partes[1]}"
                    if cam_id not in asistencia_por_camara:
                        asistencia_por_camara[cam_id] = []
                    asistencia_por_camara[cam_id].append({
                        "uuid": uuid,
                        "confidence": data.get("confidence", 0.0)
                    })

        # 3. Enviar a Supabase en un hilo separado para no congelar el video
        threading.Thread(target=self._enviar_asistencia_batch, args=(hora_actual, asistencia_por_camara), daemon=True).start()
        
        self.last_auto_scan_time = time.time()
        
        # Sonido de obturador/captura si es manual
        if platform.system() == "Windows" and not auto:
            winsound.Beep(1200, 150)

    def _enviar_asistencia_batch(self, hora_clase, datos_asistencia):
        if hasattr(self.api_client, 'registrar_asistencia_clase'):
            # Capturamos la respuesta (True o False)
            exito = self.api_client.registrar_asistencia_clase(hora_clase, datos_asistencia)
            
            if exito:
                # Alerta visual de éxito en Hilo Principal
                self.root.after(0, lambda: self.update_ui_state("✅ ¡Asistencia Guardada!", "#00FF00")) # Verde brillante
                if platform.system() == "Windows":
                    winsound.Beep(1500, 300) # Sonido agudo de éxito
            else:
                # Alerta visual de error en Hilo Principal
                self.root.after(0, lambda: self.update_ui_state("❌ Error de Conexión", "#FF0000")) # Rojo
                if platform.system() == "Windows":
                    winsound.Beep(500, 500) # Sonido grave de error
                
            # Restaurar el estado normal de la UI después de 3.5 segundos
            self.root.after(3500, lambda: self.update_ui_state("Estado: Reconocimiento Activo", "#2ECC71"))
        else:
            print("[ADVERTENCIA] El método de Supabase no fue encontrado.")

    def switch_camera(self, idx):
        self.active_camera_idx = idx
        self.view_mode = "SINGLE"
        if platform.system() == "Windows":
            winsound.Beep(800, 100)

    def on_camera_select(self, event):
        idx = self.cam_combo.current()
        self.switch_camera(idx)

    def show_grid_view(self):
        self.view_mode = "GRID"
        if platform.system() == "Windows":
            winsound.Beep(850, 100)

    def create_connection_lost_frame(self):
        w = self.video_label.winfo_width()
        h = self.video_label.winfo_height()
        if w < 10 or h < 10:
            w, h = 640, 480
        display_frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(display_frame, "CONEXION PERDIDA", (w // 2 - 130, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, "Intentando reconectar...", (w // 2 - 120, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return display_frame

    def update_frame(self):
        if not self.running:
            return

        display_frame = None

        if self.view_mode == "SINGLE":
            stream = self.streams[self.active_camera_idx]
            frame = stream.get_frame()

            if getattr(stream, "is_connected", True) and frame is not None:
                z = self.zoom_factor.get()
                if z > 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h / z), int(w / z)
                    max_shift_x, max_shift_y = w - new_w, h - new_h
                    pan_val_x, pan_val_y = self.pan_x.get(), self.pan_y.get()
                    x1 = max(0, min(int(max_shift_x * ((pan_val_x + 1.0) / 2.0)), max_shift_x))
                    y1 = max(0, min(int(max_shift_y * ((pan_val_y + 1.0) / 2.0)), max_shift_y))
                    cropped = frame[y1 : y1 + new_h, x1 : x1 + new_w]
                    frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

                display_frame = frame.copy()

                if self.mode == "RECOGNIZE":
                    display_frame = self.process_recognition(frame, display_frame)
                elif self.mode == "REGISTER":
                    display_frame = self.process_registration(frame, display_frame)
                elif self.mode == "TRAINING":
                    cv2.putText(display_frame, "Entrenando modelo...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                display_frame = self.create_connection_lost_frame()

        elif self.view_mode == "GRID":
            frames = []
            target_w, target_h = 320, 240

            for i, stream in enumerate(self.streams):
                f = stream.get_frame()
                if f is not None and getattr(stream, "is_connected", True):
                    proc_frame = f.copy()
                    if self.mode == "RECOGNIZE":
                        proc_frame = self.process_recognition(f, proc_frame, stream_idx=i)
                    frames.append(cv2.resize(proc_frame, (target_w, target_h)))
                else:
                    blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    cv2.putText(blank, "SIN CONEXION", (80, target_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    frames.append(blank)

            if len(frames) == 1: display_frame = frames[0]
            elif len(frames) == 2: display_frame = np.hstack((frames[0], frames[1]))
            elif len(frames) >= 3:
                top = np.hstack((frames[0], frames[1]))
                bottom = np.hstack((frames[2], np.zeros((target_h, target_w, 3), dtype=np.uint8) if len(frames)==3 else frames[3]))
                display_frame = np.vstack((top, bottom))
                
            cv2.putText(display_frame, "VISTA GENERAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- LÓGICA DEL TEMPORIZADOR AUTOMÁTICO ---
        """ if self.mode == "RECOGNIZE":
            tiempo_transcurrido = time.time() - self.last_auto_scan_time
            if tiempo_transcurrido >= self.intervalo_var.get():
                self.tomar_asistencia_demo(auto=True) """
        # ------------------------------------------

        if display_frame is not None:
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            label_w, label_h = self.video_label.winfo_width(), self.video_label.winfo_height()

            if label_w > 10 and label_h > 10:
                frame_h, frame_w = rgb_frame.shape[:2]
                target_aspect, frame_aspect = label_w / label_h, frame_w / frame_h
                if frame_aspect > target_aspect:
                    new_w = int(frame_h * target_aspect)
                    x_offset = (frame_w - new_w) // 2
                    rgb_frame = rgb_frame[:, x_offset:x_offset + new_w]
                else:
                    new_h = int(frame_w / target_aspect)
                    y_offset = (frame_h - new_h) // 2
                    rgb_frame = rgb_frame[y_offset:y_offset + new_h, :]

                rgb_frame = cv2.resize(rgb_frame, (label_w, label_h), interpolation=cv2.INTER_LINEAR)

            self.current_imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
            self.video_label.configure(image=self.current_imgtk)

        self.root.after(16, self.update_frame)

    def process_recognition(self, frame, display_frame, stream_idx=None):
        if stream_idx is None:
            stream_idx = self.active_camera_idx
            
        stream = self.streams[stream_idx]
        camera_id = stream.camera_id

        self.frame_counter = (self.frame_counter + 1) % 100000

        if self.frame_counter % self.process_every_n_frames == 0:
            context = self.vision_engine.detect(frame)
            context = self.tracker.update(context)
            context = self.recognition_engine.process(frame, context, self.vision_engine, camera_id)
            self.last_faces = context.faces 

        for face in self.last_faces:
            confidence = getattr(face, "confidence", 0.0)
            identity = getattr(face, "identity_uuid", "Calculando...")

            if identity in ["unknown", "Desconocido"]:
                color = (0, 0, 255)
            elif identity == "Calculando...":
                color = (255, 255, 0)
            else:
                color = (0, 255, 0)
                # NOTA: Se eliminó el envío continuo a 'event_manager' para 
                # dar paso a la captura manual/temporizada de la feria.

            cv2.rectangle(display_frame, (int(face.left), int(face.top)), (int(face.right), int(face.bottom)), color, 2)
            display_id = identity[:15] if len(identity) > 15 else identity
            label = f"{display_id} ({confidence:.1f}%)" if confidence > 0 else f"{display_id}"
            cv2.putText(display_frame, label, (int(face.left), int(face.top) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return display_frame

    def process_registration(self, frame, display_frame):
        self.frame_counter = (self.frame_counter + 1) % 100000
        if self.frame_counter % self.process_every_n_frames == 0:
            self.last_reg_faces = self.vision_engine.app.get(frame)

        faces = self.last_reg_faces

        if len(faces) == 1:
            box = faces[0].bbox.astype(int)
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                color = (200, 200, 200) 
                if self.frame_counter % self.process_every_n_frames == 0:
                    blur_variance = cv2.Laplacian(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    color = (0, 0, 255)
                    if blur_variance >= BLUR_THRESHOLD and (time.time() - self.cooldown_time > 0.4):
                        filename = os.path.join(self.person_dir, f"{self.identity_label}_{self.captured_photos:03d}.jpg")
                        cv2.imwrite(filename, face_crop)
                        self.captured_photos += 1
                        self.cooldown_time = time.time()
                        color = (0, 255, 0)
                        if platform.system() == "Windows": winsound.Beep(1000, 150)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"Capturas: {self.captured_photos}/{MAX_PHOTOS_PER_PERSON}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if self.captured_photos >= MAX_PHOTOS_PER_PERSON:
                if platform.system() == "Windows": winsound.Beep(1500, 400)
                messagebox.showinfo("Éxito", f"Se registró el rostro a la Cédula: {self.identity_label}.")
                self.mode = "RECOGNIZE"
                self.update_ui_state("Estado: Reconocimiento Activo", "#2ECC71")
        elif len(faces) > 1:
            cv2.putText(display_frame, "ERROR: Multiples rostros detectados", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return display_frame

    def start_registration(self):
        if self.mode == "TRAINING": return
        codigo = simpledialog.askstring("Registro", "Ingrese el número de Cédula (10 dígitos):", parent=self.root)
        if not codigo or not codigo.strip(): return
        if not (len(codigo.strip()) == 10 and codigo.strip().isdigit()):
            messagebox.showwarning("Error", "Debe contener 10 dígitos numéricos.", parent=self.root)
            return

        self.identity_label = codigo.strip()
        self.person_dir = os.path.join(DATASET_DIR, self.identity_label)
        os.makedirs(self.person_dir, exist_ok=True)
        self.captured_photos = 0
        self.cooldown_time = time.time()
        self.mode = "REGISTER"
        self.update_ui_state(f"Estado: Registrando a {self.identity_label}...", "#3498DB")

    def start_training(self):
        if self.mode == "TRAINING": return
        if messagebox.askyesno("Confirmar", "¿Iniciar el entrenamiento?"):
            self.mode = "TRAINING"
            self.update_ui_state("Estado: Entrenando Modelo...", "#F39C12")
            threading.Thread(target=self._train_task, daemon=True).start()

    def _train_task(self):
        try:
            directories = FileManager.get_dataset_directories(DATASET_DIR)
            if not directories:
                self.root.after(0, lambda: messagebox.showerror("Error", "Dataset vacío."))
                return

            model_data = ModelTrainer(detection_model="hog").train_from_directory(directories)
            if len(model_data["encodings"]) > 0:
                FileManager.save_model(model_data, MODEL_PATH)
                self.recognition_engine.known_encodings = model_data["encodings"]
                self.recognition_engine.known_names = model_data["names"]
                self.root.after(0, lambda: messagebox.showinfo("Éxito", "Modelo actualizado."))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "No se generaron embeddings."))
        except Exception as e:
            self.root.after(0, lambda msg=str(e): messagebox.showerror("Error", msg))
        finally:
            self.root.after(0, self._restore_recognition_mode)

    def _restore_recognition_mode(self):
        self.mode = "RECOGNIZE"
        self.update_ui_state("Estado: Reconocimiento Activo", "#2ECC71")

    def update_ui_state(self, text, color):
        self.lbl_status.config(text=text, fg=color)

    def on_closing(self):
        if messagebox.askokcancel("Salir", "¿Deseas cerrar el programa?"):
            self.running = False
            for stream in self.streams:
                self.api_client.set_camera_status(stream.camera_id, False)
                stream.release()
            self.root.destroy()
            sys.exit(0)

if __name__ == "__main__":
    os.makedirs("src/events", exist_ok=True)
    os.makedirs("src/network", exist_ok=True)
    if not os.path.exists("src/events/__init__.py"): open("src/events/__init__.py", 'a').close()
    if not os.path.exists("src/network/__init__.py"): open("src/network/__init__.py", 'a').close()
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()