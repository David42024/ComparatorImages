import streamlit as st
import cv2
import numpy as np
from PIL import Image

def ejercicio_10():
    st.title("🧱 Realidad Aumentada – Seguimiento con Pirámide 3D")

    st.write("""
    Este ejercicio permite **seleccionar una región en una imagen** y superponer una pirámide 3D.
    """)
    
    st.warning("⚠️ El tracking en tiempo real con cámara no está disponible en Streamlit Cloud. Usa una imagen estática.")

    # Opción de entrada
    opcion = st.radio("Selecciona el modo:", ["Subir imagen", "Tomar foto"])
    
    img_data = None
    
    if opcion == "Subir imagen":
        img_data = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    else:
        img_data = st.camera_input("Toma una foto")

    if img_data is not None:
        # Cargar imagen
        img = np.array(Image.open(img_data).convert("RGB"))
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        
        st.image(img, caption="Imagen Original", use_container_width=True)
        
        st.write("### Selecciona la región para la pirámide:")
        
        # Sliders para seleccionar ROI
        col1, col2 = st.columns(2)
        with col1:
            x = st.slider("X (izquierda)", 0, w-1, w//4, key="x_roi")
            y = st.slider("Y (arriba)", 0, h-1, h//4, key="y_roi")
        with col2:
            roi_w = st.slider("Ancho", 10, w-x, w//2, key="w_roi")
            roi_h = st.slider("Alto", 10, h-y, h//2, key="h_roi")
        
        # Crear copia para dibujar
        output = frame.copy()
        
        # Pirámide simulada 3D
        pts_base = np.array([
            [x, y + roi_h],
            [x + roi_w, y + roi_h],
            [x + roi_w, y],
            [x, y]
        ], np.int32)

        # Ápice de la pirámide (encima del objeto)
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
        st.write("### Resultado con Pirámide 3D:")
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        st.image(output_rgb, use_container_width=True)
        
        # Botón de descarga
        from io import BytesIO
        result_img = Image.fromarray(output_rgb)
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            label="📥 Descargar Resultado",
            data=buf.getvalue(),
            file_name="piramide_3d.png",
            mime="image/png"
        )
    else:
        st.info("👆 Sube una imagen o toma una foto para comenzar.")

if __name__ == "__main__":
    ejercicio_10()