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

    # Opción de entrada
    img_data = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if img_data is None:
        img_data = st.camera_input("O usa la cámara para tomar una foto")

    if img_data is not None:
        # Cargar imagen
        img = np.array(Image.open(img_data).convert("RGB"))
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        
        st.image(img, caption="Imagen Original")
        
        st.write("Selecciona la región para la pirámide:")
        
        # Sliders para seleccionar ROI
        x = st.slider("X (izquierda)", 0, w-1, w//4)
        y = st.slider("Y (arriba)", 0, h-1, h//4)
        roi_w = st.slider("Ancho", 10, w-x, w//2)
        roi_h = st.slider("Alto", 10, h-y, h//2)
        
        # Crear copia para dibujar
        output = frame.copy()
        
        # Pirámide simulada 3D
        pts_base = np.array([
            [x, y + roi_h],
            [x + roi_w, y + roi_h],
            [x + roi_w, y],
            [x, y]
        ], np.int32)

        # Ápice de la pirámide
        apex = (x + roi_w // 2, max(0, y - roi_h // 2))

        # Dibujar base de la pirámide
        cv2.polylines(output, [pts_base], True, (0, 255, 0), 3)
        
        # Dibujar aristas desde la base hacia el ápice
        for px, py in pts_base:
            cv2.line(output, (px, py), apex, (255, 0, 0), 2)
        
        # Dibujar el ápice
        cv2.circle(output, apex, 8, (0, 0, 255), -1)

        # Rectángulo del bounding box
        cv2.rectangle(output, (x, y), (x + roi_w, y + roi_h), (0, 255, 255), 2)

        # Mostrar resultado
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        st.image(output_rgb, caption="Resultado con Pirámide 3D")
    else:
        st.info("Por favor, sube una imagen o usa la cámara para comenzar.")

if __name__ == "__main__":
    ejercicio_10()