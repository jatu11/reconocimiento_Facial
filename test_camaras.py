import cv2

def test_cameras():
    print("Buscando cámaras conectadas...")
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Cámara encontrada en el índice: {i}")
            cap.release()
    print("Búsqueda finalizada.")

if __name__ == "__main__":
    test_cameras()