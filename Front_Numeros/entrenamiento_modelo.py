# entrenamiento_modelo.py
import tensorflow as tf
import tensorflow_datasets as tfds
import math

# Cargar datos
datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)
datos_ent = datos['train'].map(lambda x, y: (tf.cast(x, tf.float32)/255, y)).cache().shuffle(10000).batch(32)
datos_pru = datos['test'].map(lambda x, y: (tf.cast(x, tf.float32)/255, y)).batch(32)

# Crear el modelo
modelo = tf.keras.Sequential([
    # Definir primer capa de entrada Flatten
    tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
    # Definir la primera capa oculta
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(60, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar (si es necesario re-entrenar)
modelo.fit(datos_ent, epochs=8, validation_data=datos_pru)

# Evaluar el modelo
performance = modelo.evaluate(datos_pru)
print(f"Precisión del modelo: {performance[1]*100:.2f}%")

# Guardar modelo (ejecutar solo una vez después de entrenar)
modelo.save('modelo_mnist.h5')