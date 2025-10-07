import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Variable global para almacenar el nombre del tensor de entrada
# Se inicializará al cargar el modelo
INPUT_TENSOR_NAME = None 

def ejercicio_11():
    st.set_page_config(page_title="Clasificador de Perros y Gatos", layout="centered")
    st.title("Ejercicio 11 - Clasificador de Perros y Gatos 🐶🐱")
    st.markdown("### Usando Red Neuronal Convolucional (CNN)")

    # Ruta del modelo exportado
    model_path = "modelo_entrenado/clasificador_gatos_perros_tf"
    
    # Inicialización de variables de estado
    model = None
    infer = None
    input_shape = None
    global INPUT_TENSOR_NAME

    # Verificar que existe el modelo
    if not os.path.exists(model_path):
        st.error("⚠️ No se encontró el modelo entrenado")
        st.info(f"📝 El modelo debe estar en la carpeta local: `{model_path}`")
        st.markdown("""
        ### Para entrenar un modelo:
        1. Descarga el dataset de Kaggle: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
        2. Entrena un modelo con TensorFlow/Keras.
        3. **Exporta el modelo** usando `model.export("modelo_entrenado/clasificador_gatos_perros_tf")` o `model.save()`.
        """)
        return

    # Cargar modelo
    try:
        with st.spinner("Cargando modelo..."):
            # Usamos tf.keras.models.load_model para cargar el SavedModel
            model = tf.keras.models.load_model(model_path)
            
            # 1. Obtener la función de inferencia (ConcreteFunction)
            infer = model.signatures["serving_default"]
            
            # 2. Obtener el diccionario de inputs (segundo elemento de la tupla)
            input_kwargs = infer.structured_input_signature[1]
            
            # 3. Extraer el nombre de la clave del input de forma dinámica
            if not input_kwargs:
                st.error("❌ La firma del modelo no tiene argumentos de entrada esperados.")
                return

            # Tomar el nombre del primer (y generalmente único) input
            INPUT_TENSOR_NAME = list(input_kwargs.keys())[0]
            
            # 4. Extraer la forma de entrada
            input_shape = input_kwargs[INPUT_TENSOR_NAME].shape

        st.success("✅ Modelo cargado correctamente")
        st.write(f"**Nombre del tensor de entrada detectado:** `{INPUT_TENSOR_NAME}`")
        st.write(f"**Forma de entrada del modelo:** {input_shape}")
    except Exception as e:
        # Esto capturará el error original del usuario y cualquier otro
        st.error(f"❌ Error cargando el modelo: {e}")
        st.caption("Verifica que el modelo fue exportado correctamente (SavedModel format).")
        return

    # Aseguramos que tenemos la forma de entrada para el pre-procesamiento
    if input_shape is None or len(input_shape) < 3:
        st.error("❌ No se pudo determinar la forma de entrada del modelo.")
        return

    # Extraer el tamaño de la imagen (asumimos que es [batch, height, width, channels])
    # Utilizamos el índice 1, asumiendo que el índice 0 es el tamaño del batch (None/1)
    img_size = input_shape[1] 

    # --- Lógica de Clasificación ---
    
    # Subir imagen
    uploaded_file = st.file_uploader(
        "Sube una imagen para clasificar",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        # Cargar y mostrar imagen
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen cargada", use_container_width=True)

        try:
            # 1. Pre-procesamiento
            # Redimensionar según la forma de entrada del modelo (ej. 150x150)
            img = image.resize((img_size, img_size))
            img_array = np.array(img).astype("float32") / 255.0
            # Añadir dimensión de lote (Batch: 1) -> (1, H, W, 3)
            img_array = np.expand_dims(img_array, axis=0) 

            # 2. Conversión a tensor y Predicción
            input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            
            # Usamos el nombre del tensor detectado para pasar el argumento:
            # {nombre_tensor: input_tensor}
            pred_result = infer(**{INPUT_TENSOR_NAME: input_tensor})
            
            # Asumimos que la salida se llama 'output_1' y es una probabilidad binaria (0-1)
            pred = pred_result["output_1"].numpy()[0][0] 

            # 3. Interpretar resultado
            es_gato = pred < 0.5
            clase = "🐱 Gato" if es_gato else "🐶 Perro"
            # La confianza es la probabilidad de la clase predicha
            confianza = (1.0 - pred) if es_gato else pred

            with col2:
                st.markdown("### 🧠 Resultado")
                st.markdown(f"## **{clase}**") 
                st.progress(confianza)
                st.caption(f"Confianza: {confianza*100:.2f}%")
                st.caption(f"Valor de predicción (Prob. Perro): {pred:.4f}")

                if confianza > 0.8:
                    st.success("✅ Alta confianza")
                elif confianza > 0.6:
                    st.info("⚡ Confianza moderada")
                else:
                    st.warning("⚠️ Baja confianza - La imagen podría ser ambigua")

            with st.expander("📊 Probabilidades detalladas"):
                st.write("**Gato 🐱:**", f"{(1-pred)*100:.2f}%")
                st.write("**Perro 🐶:**", f"{pred*100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error en la clasificación: {e}")
            st.exception(e) # Muestra el traceback completo en Streamlit

    # Footer
    st.markdown("---")
    st.caption("💡 **Tip:** Para mejores resultados, usa imágenes claras donde el animal sea el foco principal")


if __name__ == "__main__":
    ejercicio_11()
