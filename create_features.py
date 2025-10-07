import argparse
import os
import pickle
import cv2
import numpy as np
from sklearn.cluster import KMeans

def resize_to_size(img, size=(150, 150)):
    """Redimensiona imagen a tamaño estándar"""
    return cv2.resize(img, size)

class Quantizer(object):
    def __init__(self):
        pass
    
    def get_feature_vector(self, img, kmeans, centroids):
        """Extrae vector de características usando BOW"""
        # Detector SIFT
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        if descriptors is None:
            return np.zeros((1, len(centroids)))
        
        # Predecir cluster para cada descriptor
        labels = kmeans.predict(descriptors)
        
        # Crear histograma de palabras visuales
        feature_vector = np.zeros(len(centroids))
        for label in labels:
            feature_vector[label] += 1
        
        # Normalizar
        feature_vector = feature_vector / np.sum(feature_vector)
        return feature_vector.reshape(1, -1)

class FeatureExtractor(object):
    def __init__(self):
        self.quantizer = Quantizer()
    
    def extract_features(self, img):
        """Extrae descriptores SIFT de una imagen"""
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        return descriptors
    
    def get_feature_vector(self, img, kmeans, centroids):
        return self.quantizer.get_feature_vector(img, kmeans, centroids)

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Crea características para imágenes')
    parser.add_argument("--samples", dest="samples", action='append', 
                       nargs=2, required=True,
                       help="Etiqueta y ruta de carpeta. Ejemplo: cat images/cat/")
    parser.add_argument("--codebook-file", dest="codebook_file", required=True,
                       help="Archivo de salida para el codebook")
    parser.add_argument("--feature-map-file", dest="feature_map_file", required=True,
                       help="Archivo de salida para el mapa de características")
    parser.add_argument("--num-clusters", dest="num_clusters", type=int, default=100,
                       help="Número de clusters para k-means (default: 100)")
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    
    print("Extrayendo descriptores de imágenes...")
    all_descriptors = []
    feature_map = []
    
    extractor = FeatureExtractor()
    
    # Extraer descriptores de todas las imágenes
    for label, folder_path in args.samples:
        print(f"Procesando categoría: {label}")
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                img = resize_to_size(img)
                descriptors = extractor.extract_features(img)
                
                if descriptors is not None:
                    all_descriptors.append(descriptors)
                    feature_map.append({
                        'label': label,
                        'filename': filename,
                        'descriptors': descriptors
                    })
    
    # Concatenar todos los descriptores
    all_descriptors = np.vstack(all_descriptors)
    print(f"Total de descriptores extraídos: {all_descriptors.shape[0]}")
    
    # Aplicar k-means clustering
    print(f"Aplicando k-means con {args.num_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)
    centroids = kmeans.cluster_centers_
    
    # Guardar codebook (kmeans y centroides)
    print("Guardando codebook...")
    with open(args.codebook_file, 'wb') as f:
        pickle.dump((kmeans, centroids), f)
    
    # Crear vectores de características usando BOW
    print("Creando vectores de características...")
    for item in feature_map:
        img_path = os.path.join(
            [folder for label, folder in args.samples if label == item['label']][0],
            item['filename']
        )
        img = cv2.imread(img_path)
        img = resize_to_size(img)
        
        feature_vector = extractor.get_feature_vector(img, kmeans, centroids)
        item['feature_vector'] = feature_vector
        del item['descriptors']  # No necesitamos guardar los descriptores
    
    # Guardar feature map
    print("Guardando mapa de características...")
    with open(args.feature_map_file, 'wb') as f:
        pickle.dump(feature_map, f)
    
    print("¡Proceso completado!")
    print(f"Codebook guardado en: {args.codebook_file}")
    print(f"Feature map guardado en: {args.feature_map_file}")