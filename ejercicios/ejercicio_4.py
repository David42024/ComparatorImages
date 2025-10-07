import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

def ejercicio_4():
    st.title("Ejercicio 4: Detección de Boca")
    st.write("Sube una imagen o usa la cámara. Se detectará la boca usando un clasificador Haar y se dibujará un rectángulo verde.")

    # Cargar clasificador de boca
    mouth_cascade_path = os.path.join("cascade_files", "haarcascade_mcs_mouth.xml")
    if not os.path.exists(mouth_cascade_path):
        st.error(f"No se encontró el clasificador de boca en {mouth_cascade_path}")
        return
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
    if mouth_cascade.empty():
        st.error("No se pudo cargar el clasificador de boca.")
        return

    img_data = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="ej4_img")
    if img_data is None:
        img_data = st.camera_input("O usa la cámara para tomar una foto")

    if img_data is not None:
        img = np.array(Image.open(img_data).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_out = img_bgr.copy()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mouth_rects = mouth_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=11)
        for (x, y, w, h) in mouth_rects:
            y = int(y - 0.15 * h)
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 3)
            break  # Solo la primera boca detectada
        st.subheader("Resultados")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Imagen Original")
        st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), caption="Imagen con Boca Detectada")
    else:
        st.info("Por favor, sube una imagen o usa la cámara para comenzar.")
