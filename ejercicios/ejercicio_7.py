import streamlit as st
import cv2
import numpy as np
from PIL import Image

def ejercicio_7():
    st.title("Ejercicio 7: Segmentación de Imagen con GrabCut")
    st.write("Sube una imagen y selecciona la región de interés con los sliders. Se aplicará GrabCut para segmentar el objeto.")

    file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="ej7_img")
    if file is not None:
        img = np.array(Image.open(file).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rows, cols = img_bgr.shape[:2]
        st.image(img, caption="Imagen Original")
        st.write("Selecciona la región de interés:")
        x = st.slider("x (izquierda)", 0, cols-2, 0)
        y = st.slider("y (arriba)", 0, rows-2, 0)
        w = st.slider("ancho", 1, cols-x-1, min(100, cols-x-1))
        h = st.slider("alto", 1, rows-y-1, min(100, rows-y-1))
        if st.button("Segmentar (GrabCut)"):
            with st.spinner("Procesando GrabCut..."):
                mask = np.zeros(img_bgr.shape[:2], np.uint8)
                rect = (x, y, w, h)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
                img_result = img_bgr * mask2[:, :, np.newaxis]
                st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), caption="Segmentación con GrabCut")
    else:
        st.info("Por favor, sube una imagen para comenzar.")
