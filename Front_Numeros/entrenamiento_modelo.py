# entrenamiento_modelo.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Cargar datos
datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)
datos_ent = datos['train'].map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))
datos_pru = datos['test'].map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(32)

# Separar datos de entrenamiento en arrays (para usar ImageDataGenerator)
x_train = []
y_train = []

for img, label in tfds.as_numpy(datos_ent):
    x_train.append(img)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Crear el modelo
modelo = tf.keras.Sequential([
    # Primera capa convolucional
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),  # 2,2 es el tamaño de la matriz

    # Segunda capa convolucional
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    # Tercera capa convolucional
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),

    # Aplanar todo
    tf.keras.layers.Flatten(),

    # Capas densas
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Crear el generador de datos aumentados
datagen = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1
)

# Ajustar al generador
datagen.fit(x_train)

# Entrenar usando datos aumentados
modelo.fit(
    datagen.flow(x_train, y_train, batch_size=32), 
    epochs=10, 
    validation_data=datos_pru
)

# Evaluar el modelo
performance = modelo.evaluate(datos_pru)
print(f"Precisión del modelo: {performance[1]*100:.2f}%")

# Guardar modelo (ejecutar solo una vez después de entrenar)
modelo.save('modelo_mnist.h5')