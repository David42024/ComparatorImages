import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

def ejercicio_11():
    st.title("Ejercicio 11 - Clasificador de Perros y Gatos 🐶🐱")
    st.markdown("### Usando Red Neuronal Convolucional (CNN)")

    model_path = "modelo_entrenado/clasificador_gatos_perros_tf"

    if not os.path.exists(model_path):
        st.error("⚠️ No se encontró el modelo entrenado")
        st.info(f"📝 Debe estar en: `{model_path}`")
        return

    try:
        with st.spinner("Cargando modelo..."):
            model = tf.keras.models.load_model(model_path)
            infer = model.signatures["serving_default"]
            input_shape = infer.structured_input_signature[1]["input_1"].shape
        st.success("✅ Modelo cargado")
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {e}")
        return

    uploaded_file = st.file_uploader(
        "Sube una imagen para clasificar",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen cargada", use_container_width=True)

        try:
            img_size = input_shape[1]
            img = image.resize((img_size, img_size))
            img_array = np.expand_dims(np.array(img)/255.0, axis=0)
            input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            pred = infer(input_1=input_tensor)["output_1"].numpy()[0][0]

            es_gato = pred < 0.5
            clase = "🐱 Gato" if es_gato else "🐶 Perro"
            confianza = (1 - pred) if es_gato else pred

            with col2:
                st.markdown("### 🧠 Resultado")
                st.markdown(f"## **{clase}**")
                st.progress(confianza)
                st.caption(f"Confianza: {confianza*100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error en la clasificación: {e}")

if __name__ == "__main__":
    ejercicio_11()
