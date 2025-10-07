import streamlit as st
import cv2
import numpy as np
from PIL import Image


def compute_energy_matrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy

def find_vertical_seam(img, energy):
    rows, cols = energy.shape
    seam = np.zeros(rows, dtype=np.int32)
    cost = energy.copy()
    backtrack = np.zeros_like(cost, dtype=np.int32)
    for i in range(1, rows):
        for j in range(cols):
            min_idx = j
            min_cost = cost[i-1, j]
            if j > 0 and cost[i-1, j-1] < min_cost:
                min_cost = cost[i-1, j-1]
                min_idx = j-1
            if j < cols-1 and cost[i-1, j+1] < min_cost:
                min_cost = cost[i-1, j+1]
                min_idx = j+1
            cost[i, j] += min_cost
            backtrack[i, j] = min_idx
    seam[-1] = np.argmin(cost[-1])
    for i in range(rows-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]
    return seam

def remove_vertical_seam(img, seam):
    rows, cols = img.shape[:2]
    output = np.zeros((rows, cols-1, 3), dtype=img.dtype)
    for i in range(rows):
        output[i, :, 0] = np.delete(img[i, :, 0], seam[i])
        output[i, :, 1] = np.delete(img[i, :, 1], seam[i])
        output[i, :, 2] = np.delete(img[i, :, 2], seam[i])
    return output

def add_vertical_seam(img, seam, num_iter):
    seam = seam + num_iter
    rows, cols = img.shape[:2]
    zero_col_mat = np.zeros((rows, 1, 3), dtype=img.dtype)
    img_extended = np.hstack((img, zero_col_mat))
    for row in range(rows):
        for col in range(cols, int(seam[row]), -1):
            img_extended[row, col] = img[row, col-1]
        for i in range(3):
            v1 = img_extended[row, int(seam[row])-1, i]
            v2 = img_extended[row, int(seam[row])+1, i]
            img_extended[row, int(seam[row]), i] = (int(v1) + int(v2)) // 2
    return img_extended

def ejercicio_6():
    st.title("Ejercicio 6: Expansión de Imagen con Seam Carving")
    st.write("Sube una imagen y selecciona cuántas columnas agregar usando seam carving. El proceso puede tardar dependiendo del tamaño de la imagen y el número de columnas.")

    file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="ej6_img")
    if file is not None:
        img = np.array(Image.open(file).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rows, cols = img_bgr.shape[:2]
        st.image(img, caption="Imagen Original")
        num_seams = st.slider("Número de columnas a agregar", 1, max(1, cols // 4), 10)
        if st.button("Expandir imagen (puede tardar)"):
            with st.spinner("Procesando seam carving..."):
                img_tmp = img_bgr.copy()
                img_output = img_bgr.copy()
                for i in range(num_seams):
                    energy = compute_energy_matrix(img_tmp)
                    seam = find_vertical_seam(img_tmp, energy)
                    img_tmp = remove_vertical_seam(img_tmp, seam)
                    img_output = add_vertical_seam(img_output, seam, i)
                st.image(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB), caption="Imagen expandida (seam carving)")
    else:
        st.info("Por favor, sube una imagen para comenzar.")
