import argparse
import pickle
import numpy as np
import cv2
from sklearn import preprocessing
from collections import OrderedDict
import random

class ClassifierANN(object):
    def __init__(self, feature_vector_size, label_words):
        self.ann = cv2.ml.ANN_MLP_create()
        self.label_words = label_words
        
        # Tamaño de la capa de entrada (número de centroides)
        input_size = feature_vector_size
        # Tamaño de la capa de salida (siempre 2 para clasificación binaria)
        output_size = 2 if len(label_words) == 2 else len(label_words)
        # Aplicando regla de Heaton para capa oculta
        hidden_size = int((input_size * (2/3)) + output_size)
        
        nn_config = np.array([input_size, hidden_size, output_size], dtype=np.int32)
        self.ann.setLayerSizes(nn_config)
        
        print(f"Configuración de red: Input={input_size}, Hidden={hidden_size}, Output={output_size}")
        
        # Función de activación: Sigmoide simétrica
        self.ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
        
        # Configurar método de entrenamiento con mejores parámetros
        self.ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.ann.setBackpropWeightScale(0.05)  # Reducido para aprendizaje más fino
        self.ann.setBackpropMomentumScale(0.05)  # Reducido para aprendizaje más fino
        
        # Términos de criterio de parada - más iteraciones
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5000, 0.001)
        self.ann.setTermCriteria(criteria)
        
        # Codificador de etiquetas
        self.le = preprocessing.LabelBinarizer()
        self.le.fit(label_words)
    
    def train(self, training_set):
        """Entrena la red neuronal"""
        label_words = [item['label'] for item in training_set]
        dim_size = training_set[0]['feature_vector'].shape[1]
        
        train_samples = np.asarray(
            [np.reshape(x['feature_vector'], (dim_size,)) for x in training_set]
        )
        
        # Crear matriz de salida one-hot encoding manualmente
        train_response = np.zeros((len(training_set), 2), dtype=np.float32)
        
        for i, label in enumerate(label_words):
            if label == self.label_words[0]:  # Primera clase (ej: cat)
                train_response[i] = [1.0, 0.0]
            else:  # Segunda clase (ej: dog)
                train_response[i] = [0.0, 1.0]
        
        print(f"Dimensiones de entrada: {train_samples.shape}")
        print(f"Dimensiones de salida: {train_response.shape}")
        print(f"Ejemplo de salida: {train_response[:3]}")
        
        # Entrenar la red
        self.ann.train(
            train_samples.astype(np.float32),
            cv2.ml.ROW_SAMPLE,
            train_response
        )
    
    def _init_confusion_matrix(self, label_words):
        """Inicializa matriz de confusión"""
        confusion_matrix = OrderedDict()
        for label in label_words:
            confusion_matrix[label] = OrderedDict()
            for label2 in label_words:
                confusion_matrix[label][label2] = 0
        return confusion_matrix
    
    def classify(self, encoded_word, threshold=0.5):
        """Clasifica una palabra codificada"""
        # Para clasificación binaria, elegir la clase con mayor valor
        if len(self.label_words) == 2:
            if encoded_word[0] > encoded_word[1]:
                return self.label_words[0]
            else:
                return self.label_words[1]
        else:
            models = self.le.inverse_transform(np.asarray([encoded_word]), threshold)
            return models[0]
    
    def get_confusion_matrix(self, testing_set):
        """Calcula matriz de confusión"""
        label_words = [item['label'] for item in testing_set]
        dim_size = testing_set[0]['feature_vector'].shape[1]
        
        test_samples = np.asarray(
            [np.reshape(x['feature_vector'], (dim_size,)) for x in testing_set]
        )
        
        # Crear matriz de salida esperada
        expected_outputs = np.zeros((len(testing_set), 2), dtype=np.float32)
        for i, label in enumerate(label_words):
            if label == self.label_words[0]:
                expected_outputs[i] = [1.0, 0.0]
            else:
                expected_outputs[i] = [0.0, 1.0]
        
        confusion_matrix = self._init_confusion_matrix(self.label_words)
        
        retval, test_outputs = self.ann.predict(test_samples.astype(np.float32))
        
        for expected_output, test_output in zip(expected_outputs, test_outputs):
            expected_model = self.classify(expected_output)
            predicted_model = self.classify(test_output)
            confusion_matrix[expected_model][predicted_model] += 1
        
        return confusion_matrix

def split_feature_map(feature_map, training_ratio):
    """Divide el conjunto de datos en entrenamiento y prueba"""
    random.shuffle(feature_map)
    split_index = int(len(feature_map) * training_ratio)
    return feature_map[:split_index], feature_map[split_index:]

def print_accuracy(confusion_matrix):
    """Calcula y muestra la precisión"""
    acc_models = OrderedDict()
    
    for model in confusion_matrix.keys():
        acc_models[model] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    for expected_model, predicted_models in confusion_matrix.items():
        for predicted_model, value in predicted_models.items():
            if predicted_model == expected_model:
                acc_models[expected_model]['TP'] += value
                acc_models[predicted_model]['TN'] += value
            else:
                acc_models[expected_model]['FN'] += value
                acc_models[predicted_model]['FP'] += value
    
    print("\nPrecisión por clase:")
    for model, rep in acc_models.items():
        total = rep['TP'] + rep['TN'] + rep['FN'] + rep['FP']
        if total > 0:
            acc = (rep['TP'] + rep['TN']) / total
            print(f'{model:12s} \t {acc:.2%}')

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Entrena la red neuronal ANN')
    parser.add_argument("--feature-map-file", dest="feature_map_file", required=True,
                       help="Archivo pickle con el mapa de características")
    parser.add_argument("--training-set", dest="training_set", required=True,
                       type=float, help="Porcentaje para entrenamiento (ej: 0.75)")
    parser.add_argument("--ann-file", dest="ann_file", required=False,
                       help="Archivo de salida para guardar ANN")
    parser.add_argument("--le-file", dest="le_file", required=False,
                       help="Archivo de salida para guardar LabelEncoder")
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    
    # Cargar el mapa de características
    print("Cargando mapa de características...")
    with open(args.feature_map_file, 'rb') as f:
        feature_map = pickle.load(f)
    
    print(f"Total de imágenes: {len(feature_map)}")
    
    # Dividir en conjunto de entrenamiento y prueba
    training_set, testing_set = split_feature_map(feature_map, args.training_set)
    print(f"Entrenamiento: {len(training_set)} imágenes")
    print(f"Prueba: {len(testing_set)} imágenes")
    
    # Obtener etiquetas únicas
    label_words = np.unique([item['label'] for item in training_set])
    print(f"Clases: {label_words}")
    
    # Crear y entrenar clasificador
    print("\nCreando red neuronal...")
    cnn = ClassifierANN(len(feature_map[0]['feature_vector'][0]), label_words)
    
    print("Entrenando red neuronal...")
    cnn.train(training_set)
    
    # Evaluar con matriz de confusión
    print("\n===== Matriz de Confusión =====")
    confusion_matrix = cnn.get_confusion_matrix(testing_set)
    
    # Imprimir matriz de confusión
    print("\n        ", end="")
    for label in label_words:
        print(f"{label:12s}", end="")
    print()
    
    for expected_label in label_words:
        print(f"{expected_label:12s}", end="")
        for predicted_label in label_words:
            count = confusion_matrix[expected_label][predicted_label]
            print(f"{count:12d}", end="")
        print()
    
    print("\n===== Precisión de la Red =====")
    print_accuracy(confusion_matrix)
    
    # Guardar modelos
    if args.ann_file and args.le_file:
        print("\n===== Guardando modelos =====")
        cnn.ann.save(args.ann_file)
        with open(args.le_file, 'wb') as f:
            pickle.dump(cnn.le, f)
        print(f'ANN guardado en: {args.ann_file}')
        print(f'LabelEncoder guardado en: {args.le_file}')