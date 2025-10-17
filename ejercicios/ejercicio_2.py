import streamlit as st
import cv2
import numpy as np
from PIL import Image

def ejercicio_2():
    st.title("Ejercicio 2: Filtro Viñeta (Vignette)")
    st.write("Sube una imagen para aplicar el filtro de viñeta básico.")

    file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="ej2_img")
    if file is not None:
        img = np.array(Image.open(file).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rows, cols = img_bgr.shape[:2]
        # generar máscara de viñeta usando kernels gaussianos
        kernel_x = cv2.getGaussianKernel(cols, 100)
        kernel_y = cv2.getGaussianKernel(rows, 100)
        kernel = kernel_y * kernel_x.T
        mask = kernel / np.max(kernel)  # Normaliza para que el centro sea 1
        output = np.copy(img_bgr)
        # aplicar la máscara a cada canal
        for i in range(3):
            output[:,:,i] = output[:,:,i] * mask
        output = np.clip(output, 0, 255).astype(np.uint8)
        st.subheader("Resultados")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Imagen Original")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Imagen con Viñeta")
    else:
        st.info("Por favor, sube una imagen para comenzar.")
