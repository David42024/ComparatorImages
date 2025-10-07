import streamlit as st
import cv2
import numpy as np
from PIL import Image

def ejercicio_10():
    st.title("🧱 Realidad Aumentada – Seguimiento con Pirámide 3D")

    st.write("""
    Este ejercicio permite **seleccionar un objeto** y seguirlo en tiempo real mediante la cámara,
    superponiendo una pirámide simulada sobre él.
    """)

    run = st.checkbox("Iniciar cámara")

    # 🔧 Imagen inicial negra para evitar error
    empty_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    FRAME_WINDOW = st.image(empty_frame, caption="Vista de cámara")

    tracker = None
    roi_selected = False

    if run:
        cap = cv2.VideoCapture(0)
        st.info("Presiona 'Seleccionar ROI' para marcar el área del objeto a seguir.")
        
        if st.button("Seleccionar ROI"):
            ret, frame = cap.read()
            if ret:
                r = cv2.selectROI("Selecciona ROI", frame, fromCenter=False)
                cv2.destroyAllWindows()
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, r)
                roi_selected = True
                st.success(f"✅ ROI seleccionada: {r}")
            else:
                st.error("No se pudo capturar el frame para selección.")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ No se pudo leer el frame.")
                break

            if roi_selected and tracker:
                ok, bbox = tracker.update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in bbox]

                    # Pirámide simulada
                    pts_base = np.array([
                        [x, y + h],
                        [x + w, y + h],
                        [x + w // 2, y]
                    ], np.int32)

                    apex = (x + w // 2, y - h // 2)

                    cv2.polylines(frame, [pts_base], True, (0, 255, 0), 2)
                    for px, py in pts_base:
                        cv2.line(frame, (px, py), apex, (255, 0, 0), 2)

                    cv2.putText(frame, "Tracking...", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Perdido...", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()

    else:
        st.warning("Activa la cámara para comenzar.")
