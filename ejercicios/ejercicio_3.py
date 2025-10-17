import streamlit as st
import cv2
import numpy as np
from PIL import Image

def ejercicio_3():
    st.title("Ejercicio 3: Efecto Negativo en Región Seleccionada")
    st.write("Sube una imagen o usa la cámara, selecciona una región y aplica el efecto negativo solo en esa zona.")

    img_data = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="ej3_img")
    if img_data is None:
        img_data = st.camera_input("O usa la cámara para tomar una foto")

    if img_data is not None:
        img = np.array(Image.open(img_data).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rows, cols = img_bgr.shape[:2]

        # Guardar dimensiones en session_state para mantener los sliders
        if 'img_cols' not in st.session_state:
            st.session_state.img_cols = cols
            st.session_state.img_rows = rows

        st.write("Selecciona la región a invertir (negativo):")
        x0 = st.slider("x0 (izquierda)", 0, cols-1, 0)
        y0 = st.slider("y0 (arriba)", 0, rows-1, 0)
        x1 = st.slider("x1 (derecha)", x0+1, cols, cols)
        y1 = st.slider("y1 (abajo)", y0+1, rows, rows)

        output = img_bgr.copy()
        output[y0:y1, x0:x1] = 255 - output[y0:y1, x0:x1]
        st.subheader("Resultados")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Imagen Original")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Imagen con Región Negativa")
    else:
        st.info("Por favor, sube una imagen o usa la cámara para comenzar.")