import streamlit as st
import cv2
import numpy as np
from PIL import Image

def ejercicio_1():
    st.title("Ejercicio 1: Traslación de Imágenes")
    st.write("Sube una imagen para aplicar traslaciones y ver el efecto de borde envolvente (WRAP).")

    file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="ej1_img")
    if file is not None:
        img = np.array(Image.open(file).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        num_rows, num_cols = img_bgr.shape[:2]

        # Primera traslación
        translation_matrix1 = np.float32([[1, 0, 70], [0, 1, 110]])
        img_translation = cv2.warpAffine(img_bgr, translation_matrix1, (num_cols + 70, num_rows + 110))

        # Segunda traslación
        translation_matrix2 = np.float32([[1, 0, -30], [0, 1, -50]])
        img_translation2 = cv2.warpAffine(img_translation, translation_matrix2, (num_cols + 70 + 30, num_rows + 110 + 50))

        # Borde envolvente (WRAP)
        translation_matrix3 = np.float32([[1, 0, 70], [0, 1, 110]])
        img_border_wrap = cv2.warpAffine(img_bgr, translation_matrix3, (num_cols, num_rows), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP, borderValue=1)

        st.subheader("Resultados")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Imagen Original")
        st.image(cv2.cvtColor(img_translation2, cv2.COLOR_BGR2RGB), caption="Traslación General")
        st.image(cv2.cvtColor(img_border_wrap, cv2.COLOR_BGR2RGB), caption="Con Borde Envolvente (WRAP)")
    else:
        st.info("Por favor, sube una imagen para comenzar.")
