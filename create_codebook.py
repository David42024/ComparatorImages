# create_codebook.py
import os
import cv2
import pickle
import numpy as np
from sklearn.cluster import KMeans

def create_codebook(image_dir="dataset", n_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []

    for label in os.listdir(image_dir):
        class_path = os.path.join(image_dir, label)
        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            if des is not None:
                descriptors_list.append(des)

    descriptors = np.vstack(descriptors_list)
    print(f"Entrenando KMeans con {len(descriptors)} descriptores...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(descriptors)
    centroids = kmeans.cluster_centers_

    os.makedirs("models", exist_ok=True)
    with open("models/codebook.pkl", "wb") as f:
        pickle.dump((kmeans, centroids), f)

    print("✅ Codebook creado y guardado en 'models/codebook.pkl'")

if __name__ == "__main__":
    create_codebook()
