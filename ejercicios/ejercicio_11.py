import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pickle

def resize_to_size(img, size=(150, 150)):
    """Redimensiona imagen a tamaño estándar"""
    return cv2.resize(img, size)

class Quantizer(object):
    def __init__(self):
        pass
    
    def get_feature_vector(self, img, kmeans, centroids):
        """Extrae vector de características usando BOW"""
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        if descriptors is None:
            return np.zeros((1, len(centroids)))
        
        labels = kmeans.predict(descriptors)
        feature_vector = np.zeros(len(centroids))
        for label in labels:
            feature_vector[label] += 1
        
        feature_vector = feature_vector / np.sum(feature_vector)
        return feature_vector.reshape(1, -1)

class FeatureExtractor(object):
    def __init__(self):
        self.quantizer = Quantizer()
    
    def get_feature_vector(self, img, kmeans, centroids):
        return self.quantizer.get_feature_vector(img, kmeans, centroids)

def ejercicio_11():
    st.title("Ejercicio 11 - Clasificador de Perros y Gatos 🐶🐱")
    st.markdown("### Usando Red Neuronal ANN con Bag of Visual Words")

    # Rutas de los modelos
    codebook_path = "models/codebook.pkl"
    ann_path = "models/ann.yaml"
    le_path = "models/le.pkl"
    
    # Verificar que existen los modelos
    missing_files = []
    if not os.path.exists(codebook_path):
        missing_files.append(codebook_path)
    if not os.path.exists(ann_path):
        missing_files.append(ann_path)
    if not os.path.exists(le_path):
        missing_files.append(le_path)
    
    if missing_files:
        st.error("⚠️ No se encontraron los modelos entrenados")
        st.info("📝 Archivos faltantes:")
        for file in missing_files:
            st.code(file)
        st.markdown("""
        ### Para entrenar el modelo:
        ```bash
        # Paso 1: Crear características y codebook
        python create_features.py --samples cat images/cat --samples dog images/dog --codebook-file models/codebook.pkl --feature-map-file models/feature_map.pkl --num-clusters 300
        
        # Paso 2: Entrenar la red neuronal
        python training.py --feature-map-file models/feature_map.pkl --training-set 0.75 --ann-file models/ann.yaml --le-file models/le.pkl
        ```
        """)
        return

    # Cargar modelos
    try:
        with st.spinner("Cargando modelos..."):
            # Cargar red neuronal
            ann = cv2.ml.ANN_MLP_load(ann_path)
            
            # Cargar codificador de etiquetas
            with open(le_path, 'rb') as f:
                le = pickle.load(f)
            
            # Cargar codebook
            with open(codebook_path, 'rb') as f:
                kmeans, centroids = pickle.load(f)
            
            # Crear extractor de características
            extractor = FeatureExtractor()
        
        st.success("✅ Modelos cargados correctamente")
        st.write(f"**Número de clusters (vocabulario visual):** {len(centroids)}")
        st.write(f"**Clases:** {le.classes_}")
        
    except Exception as e:
        st.error(f"❌ Error cargando los modelos: {e}")
        st.exception(e)
        return

    # --- Lógica de Clasificación ---
    
    # Subir imagen
    uploaded_file = st.file_uploader(
        "Sube una imagen para clasificar",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        try:
            # Método alternativo: guardar temporalmente
            import tempfile
            
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Cargar desde archivo temporal
            image_pil = Image.open(tmp_path).convert("RGB")
            image_cv = cv2.imread(tmp_path)
            
            # Eliminar archivo temporal
            os.unlink(tmp_path)
            
            if image_cv is None:
                st.error("❌ OpenCV no pudo leer la imagen")
                return
                
        except Exception as e:
            st.error(f"❌ No se pudo cargar la imagen: {e}")
            st.warning("⚠️ Verifica que el archivo sea una imagen válida (JPG, PNG, WEBP)")
            st.info(f"Tipo de archivo: {uploaded_file.type}")
            st.info(f"Tamaño: {uploaded_file.size} bytes")
            st.info(f"Nombre: {uploaded_file.name}")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)

        try:
            with st.spinner("Clasificando..."):
                # 1. Pre-procesamiento
                img_resized = resize_to_size(image_cv)
                
                # 2. Extraer vector de características
                feature_vector = extractor.get_feature_vector(
                    img_resized, kmeans, centroids
                )
                
                # Asegurar formato correcto
                if feature_vector.ndim == 1:
                    feature_vector = feature_vector.reshape(1, -1)
                feature_vector = feature_vector.astype(np.float32)
                
                # 3. Predicción con la red neuronal
                retval, predictions = ann.predict(feature_vector)
                
                # 4. Interpretar resultado
                # predictions es un array [prob_clase1, prob_clase2]
                pred_array = predictions[0]
                
                # Obtener las clases
                label_words = le.classes_
                
                # Determinar clase predicha
                if len(label_words) == 2:
                    if pred_array[0] > pred_array[1]:
                        clase_predicha = label_words[0]
                        confianza = float(pred_array[0])
                    else:
                        clase_predicha = label_words[1]
                        confianza = float(pred_array[1])
                else:
                    clase_idx = np.argmax(pred_array)
                    clase_predicha = label_words[clase_idx]
                    confianza = float(pred_array[clase_idx])
                
                # Normalizar confianza (valores de ANN pueden estar fuera de [0,1])
                # Convertir de [-1, 1] a [0, 1] si usa sigmoid simétrica
                confianza = (confianza + 1) / 2.0
                confianza = np.clip(confianza, 0.0, 1.0)
                
                # Determinar emoji
                emoji = "🐱" if clase_predicha == label_words[0] else "🐶"
                clase_texto = f"{emoji} {clase_predicha.capitalize()}"

            with col2:
                st.markdown("### 🧠 Resultado")
                st.markdown(f"## **{clase_texto}**")
                st.progress(float(confianza))
                st.caption(f"Confianza: {confianza*100:.2f}%")
                
                if confianza > 0.7:
                    st.success("✅ Alta confianza")
                elif confianza > 0.55:
                    st.info("⚡ Confianza moderada")
                else:
                    st.warning("⚠️ Baja confianza - La imagen podría ser ambigua")

            with st.expander("📊 Salidas de la red neuronal"):
                st.write("**Valores crudos de la red:**")
                for i, label in enumerate(label_words):
                    st.write(f"{label}: {pred_array[i]:.4f}")
                st.caption("Nota: La red usa función Sigmoide Simétrica (valores entre -1 y 1)")

        except Exception as e:
            st.error(f"❌ Error en la clasificación: {e}")
            st.exception(e)

    # Footer
    st.markdown("---")
    st.caption("💡 **Tip:** Este clasificador usa Bag of Visual Words (BOW) con descriptores SIFT")
    st.caption("Para mejores resultados, usa imágenes claras donde el animal sea el foco principal")


if __name__ == "__main__":
    ejercicio_11()