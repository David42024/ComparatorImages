import streamlit as st
import cv2
import numpy as np
from PIL import Image

def ejercicio_8():
    st.title("Ejercicio 8: Sustracción de Fondo (MOG2 y GMG)")
    st.write("Sube una imagen y observa el resultado de los sustractores de fondo MOG2 y GMG.")

    file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="ej8_img")
    if file is not None:
        img = np.array(Image.open(file).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        st.image(img, caption="Imagen Original")

        # MOG2
        bgSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2()
        mask_mog2 = bgSubtractorMOG2.apply(img_bgr, learningRate=0.05)
        mask_mog2_rgb = cv2.cvtColor(mask_mog2, cv2.COLOR_GRAY2RGB)
        st.image(mask_mog2_rgb & img, caption="MOG2: Objetos en movimiento")
    else:
        st.info("Por favor, sube una imagen para comenzar.")
