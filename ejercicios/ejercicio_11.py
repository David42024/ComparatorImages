import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

def ejercicio_11():
    st.title("Ejercicio 11 - Clasificador de Perros y Gatos 🐶🐱")
    st.markdown("### Usando Red Neuronal Convolucional (CNN)")
    
    # Ruta del modelo
    model_path = "modelo_entrenado/clasificador_gatos_perros.h5"
    
    # Verificar que existe el modelo
    if not os.path.exists(model_path):
        st.error("⚠️ No se encontró el modelo entrenado")
        st.info(f"📝 El modelo debe estar en: `{model_path}`")
        
        st.markdown("""
        ### Para entrenar un modelo:
        1. Descarga el dataset de Kaggle: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
        2. Entrena un modelo con TensorFlow/Keras
        3. Guarda el modelo en la ruta especificada
        """)
        return
    
    # Cargar modelo
    try:
        with st.spinner("Cargando modelo..."):
            model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Modelo cargado correctamente")
        
        # Mostrar información del modelo
        with st.expander("ℹ️ Información del modelo"):
            input_shape = model.input_shape
            st.write(f"**Forma de entrada:** {input_shape}")
            st.write(f"**Número de capas:** {len(model.layers)}")
            
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {e}")
        return
    
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
        
        # Procesar imagen
        try:
            # Redimensionar según el modelo (verificar input_shape)
            img_size = 64  # Ajusta según tu modelo
            img = image.resize((img_size, img_size))
            img_array = np.array(img).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)
            
            # Predecir
            with st.spinner("Clasificando..."):
                pred = model.predict(img_array, verbose=0)[0][0]
            
            # Interpretar resultado
            # pred < 0.5 = Gato, pred >= 0.5 = Perro (según entrenamiento)
            es_gato = pred < 0.5
            clase = "🐱 Gato" if es_gato else "🐶 Perro"
            confianza = (1 - pred) if es_gato else pred
            
            with col2:
                st.markdown("### 🧠 Resultado")
                st.markdown(f"## **{clase}**")
                
                # Barra de confianza
                confianza_pct = float(confianza) * 100
                st.progress(confianza_pct / 100)
                st.caption(f"Confianza: {confianza_pct:.2f}%")
                st.caption(f"Valor de predicción: {pred:.4f}")
                
                # Interpretación
                if confianza_pct > 80:
                    st.success("✅ Alta confianza")
                elif confianza_pct > 60:
                    st.info("⚡ Confianza moderada")
                else:
                    st.warning("⚠️ Baja confianza - La imagen podría ser ambigua")
            
            # Mostrar probabilidades detalladas
            with st.expander("📊 Probabilidades detalladas"):
                prob_gato = (1 - pred) * 100
                prob_perro = pred * 100
                
                st.write("**Gato 🐱:**", f"{prob_gato:.2f}%")
                st.write("**Perro 🐶:**", f"{prob_perro:.2f}%")
                
        except Exception as e:
            st.error(f"❌ Error en la clasificación: {e}")
            st.exception(e)
    
    # Footer con información
    st.markdown("---")
    st.caption("💡 **Tip:** Para mejores resultados, usa imágenes claras donde el animal sea el foco principal")

if __name__ == "__main__":
    ejercicio_11()