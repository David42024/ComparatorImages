import streamlit as st
import cv2
import numpy as np
from PIL import Image

class DenseDetector:
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound

    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, cols, self.initFeatureScale):
            for y in range(self.initImgBound, rows, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.initXyStep))
        return keypoints

def ejercicio_9():
    st.title("🖼️ Comparador de Imágenes (Dense vs SIFT)")

    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Sube la Imagen A", type=["jpg", "jpeg", "png"], key="img1")
    with col2:
        file2 = st.file_uploader("Sube la Imagen B", type=["jpg", "jpeg", "png"], key="img2")

    method = st.radio("Selecciona el detector de características:", ["Dense + ORB", "SIFT"])

    if file1 and file2:
        img1 = np.array(Image.open(file1).convert("RGB"))
        img2 = np.array(Image.open(file2).convert("RGB"))

        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        if method == "Dense + ORB":
            detector = DenseDetector(step_size=15, feature_scale=15, img_bound=5)
            kp1 = detector.detect(img1_bgr)
            kp2 = detector.detect(img2_bgr)

            orb = cv2.ORB_create()
            kp1, des1 = orb.compute(img1_bgr, kp1)
            kp2, des2 = orb.compute(img2_bgr, kp2)

        else:
            try:
                sift = cv2.SIFT_create()  
            except:
                sift = cv2.xfeatures2d.SIFT_create()  

            kp1, des1 = sift.detectAndCompute(img1_bgr, None)
            kp2, des2 = sift.detectAndCompute(img2_bgr, None)

        # MATCHING
        if des1 is not None and des2 is not None:
            if method == "Dense + ORB":
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Dibujar top N coincidencias
            N_MATCHES = 50
            img_matches = cv2.drawMatches(
                img1_bgr, kp1, img2_bgr, kp2, matches[:N_MATCHES], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            st.subheader("🔑 Resultados del Matching")
            st.image(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB),
             caption=f"Top {N_MATCHES} coincidencias encontradas con {method}")
            st.success(f"Se detectaron {len(matches)} coincidencias entre las imágenes")
        else:
            st.error("No se pudieron calcular descriptores en una de las imágenes.")
