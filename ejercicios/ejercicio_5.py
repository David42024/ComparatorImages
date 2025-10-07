import streamlit as st
import cv2
import numpy as np
from PIL import Image

def ejercicio_5():
    st.title("Ejercicio 5: BRIEF (Binary Robust Independent Elementary Features)")
    st.write("Sube una imagen para detectar keypoints con FAST y extraer descriptores con BRIEF.")

    file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="ej5_img")
    if file is not None:
        img = np.array(Image.open(file).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # FAST keypoints
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray, None)

        # BRIEF descriptors
        try:
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        except AttributeError:
            st.error("No se encontró xfeatures2d. Instala opencv-contrib-python.")
            return
        keypoints, descriptors = brief.compute(gray, keypoints)

        img_kp = cv2.drawKeypoints(img_bgr, keypoints, None, color=(0,255,0))
        st.subheader("Resultados")
        st.image(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB), caption="Keypoints detectados con FAST + BRIEF")
        st.write(f"Keypoints detectados: {len(keypoints)}")
        st.write(f"Shape de los descriptores: {descriptors.shape if descriptors is not None else 'N/A'}")
    else:
        st.info("Por favor, sube una imagen para comenzar.")
