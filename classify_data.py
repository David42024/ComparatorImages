import argparse
import pickle
import cv2
import numpy as np
import create_features as cf

class ImageClassifier(object):
    def __init__(self, ann_file, le_file, codebook_file):
        # Cargar red neuronal entrenada
        self.ann = cv2.ml.ANN_MLP_load(ann_file)
        
        # Cargar codificador de etiquetas
        with open(le_file, 'rb') as f:
            self.le = pickle.load(f)
        
        # Cargar codebook (kmeans y centroides)
        with open(codebook_file, 'rb') as f:
            self.kmeans, self.centroids = pickle.load(f)
    
    def classify(self, encoded_word, threshold=0.5):
        """Clasifica una palabra codificada"""
        # Para clasificación binaria, elegir la clase con mayor valor
        print(f"Vector de predicción: {encoded_word}")
        
        # Cargar las etiquetas desde el LabelEncoder
        label_words = self.le.classes_
        
        if len(label_words) == 2:
            if encoded_word[0] > encoded_word[1]:
                return label_words[0]
            else:
                return label_words[1]
        else:
            models = self.le.inverse_transform(np.asarray([encoded_word]), threshold)
            return models[0]
    
    def getImageTag(self, img):
        """Obtiene la etiqueta de clasificación para una imagen"""
        # Redimensionar imagen
        img = cf.resize_to_size(img)
        
        # Extraer vector de características
        feature_vector = cf.FeatureExtractor().get_feature_vector(
            img, self.kmeans, self.centroids
        )
        
        # Asegurar que sea float32 y tenga la forma correcta
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        feature_vector = feature_vector.astype(np.float32)
        
        print(f"Dimensiones del vector: {feature_vector.shape}")
        print(f"Tipo de dato: {feature_vector.dtype}")
        
        # Clasificar el vector de características
        retval, image_tag = self.ann.predict(feature_vector)
        
        print(f"Salida de la red: {image_tag}")
        
        # Retornar la clase predicha
        return self.classify(image_tag[0])

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Clasifica imágenes usando la red neuronal entrenada'
    )
    parser.add_argument("--input-image", dest="input_image", required=True,
                       help="Imagen de entrada a clasificar")
    parser.add_argument("--codebook-file", dest="codebook_file", required=True,
                       help="Archivo que contiene el codebook")
    parser.add_argument("--ann-file", dest="ann_file", required=True,
                       help="Archivo que contiene la ANN entrenada")
    parser.add_argument("--le-file", dest="le_file", required=True,
                       help="Archivo que contiene la clase LabelEncoder")
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    
    # Cargar imagen
    print(f"Cargando imagen: {args.input_image}")
    input_image = cv2.imread(args.input_image)
    
    if input_image is None:
        print("Error: No se pudo cargar la imagen")
        exit(1)
    
    # Clasificar imagen
    print("Clasificando imagen...")
    classifier = ImageClassifier(
        args.ann_file,
        args.le_file,
        args.codebook_file
    )
    
    tag = classifier.getImageTag(input_image)
    
    print("\n" + "="*50)
    print(f"Clase predicha: {tag}")
    print("="*50)