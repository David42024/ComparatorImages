from tensorflow.keras.models import load_model

# Cargar modelo .h5
model = load_model("modelo_entrenado/clasificador_gatos_perros.h5")

# Exportar a SavedModel
model.export("modelo_entrenado/clasificador_gatos_perros_tf")
print("✅ Modelo convertido a SavedModel correctamente")
