import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# --- CONFIGURACIÓN ---
train_dir = 'dataset/train'
test_dir = 'dataset/test'
img_height, img_width = 150, 150
batch_size = 32

# --- GENERADORES DE DATOS ---
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# --- MODELO ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# --- COMPILAR ---
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# --- ENTRENAR ---
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# --- GUARDAR MODELO ---
os.makedirs("modelo_entrenado", exist_ok=True)
model.save("modelo_entrenado/clasificador_gatos_perros.h5")

print("\n✅ Entrenamiento completo y modelo guardado en 'modelo_entrenado/clasificador_gatos_perros.h5'")
