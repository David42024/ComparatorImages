import streamlit as st
import cv2
import numpy as np
from PIL import Image

def crear_tracker_compatible():
    """Crea un tracker KCF compatible con todas las versiones de OpenCV"""
    try:
        # OpenCV >= 4.5.1
        return cv2.legacy.TrackerKCF_create()
    except AttributeError:
        # OpenCV < 4.5.1
        return cv2.TrackerKCF_create()

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

    # Usar session_state para mantener el tracker entre ejecuciones
    if 'tracker' not in st.session_state:
        st.session_state.tracker = None
    if 'roi_selected' not in st.session_state:
        st.session_state.roi_selected = False

    if run:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ No se pudo abrir la cámara. Verifica que esté conectada y no esté siendo usada por otra aplicación.")
            return
        
        st.info("Presiona 'Seleccionar ROI' para marcar el área del objeto a seguir.")
        
        if st.button("Seleccionar ROI"):
            ret, frame = cap.read()
            if ret:
                # Seleccionar ROI
                r = cv2.selectROI("Selecciona ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyAllWindows()
                
                if r[2] > 0 and r[3] > 0:  # Verificar que se seleccionó un área válida
                    # Crear tracker compatible
                    st.session_state.tracker = crear_tracker_compatible()
                    st.session_state.tracker.init(frame, r)
                    st.session_state.roi_selected = True
                    st.success(f"✅ ROI seleccionada: {r}")
                else:
                    st.warning("⚠️ No se seleccionó ninguna región válida.")
            else:
                st.error("❌ No se pudo capturar el frame para selección.")

        # Botón para resetear el tracking
        if st.button("Resetear Tracking"):
            st.session_state.tracker = None
            st.session_state.roi_selected = False
            st.info("Tracking reseteado. Puedes seleccionar una nueva ROI.")

        # Loop principal de captura
        stop_button = st.button("Detener")
        
        while run and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ No se pudo leer el frame.")
                break

            if st.session_state.roi_selected and st.session_state.tracker:
                ok, bbox = st.session_state.tracker.update(frame)
                
                if ok:
                    x, y, w, h = [int(v) for v in bbox]

                    # Asegurar que las coordenadas estén dentro del frame
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)

                    # Pirámide simulada 3D
                    pts_base = np.array([
                        [x, y + h],
                        [x + w, y + h],
                        [x + w, y],
                        [x, y]
                    ], np.int32)

                    # Ápice de la pirámide (encima del objeto)
                    apex = (x + w // 2, max(0, y - h // 2))

                    # Dibujar base de la pirámide
                    cv2.polylines(frame, [pts_base], True, (0, 255, 0), 2)
                    
                    # Dibujar aristas desde la base hacia el ápice
                    for px, py in pts_base:
                        cv2.line(frame, (px, py), apex, (255, 0, 0), 2)
                    
                    # Dibujar el ápice
                    cv2.circle(frame, apex, 5, (0, 0, 255), -1)

                    # Rectángulo del bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                    # Texto de tracking
                    cv2.putText(frame, "Tracking...", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Objeto perdido - Selecciona nueva ROI", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    st.session_state.roi_selected = False

            # Convertir a RGB para Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

        cap.release()
        cv2.destroyAllWindows()

    else:
        st.warning("✋ Activa la cámara para comenzar.")

if __name__ == "__main__":
    ejercicio_10()