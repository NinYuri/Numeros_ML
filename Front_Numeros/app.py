from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from PIL import ImageOps
import io

app = Flask(__name__)
modelo = load_model('modelo_mnist.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug 1: Verificar recepción de imagen
        print("\n--- Inicio de solicitud ---")
        
        # Obtener imagen del frontend
        file = request.files['image']
        img = Image.open(file.stream).convert('L')
        
        # Debug 2: Tamaño original recibido
        print(f"Tamaño original de la imagen: {img.size} (debe ser 280x280)")
        
        # Paso crítico 1: Redimensionar y centrar
        img_redim = ImageOps.fit(img, (28, 28), method=0, bleed=0.0, centering=(0.5, 0.5))
        print(f"Tamaño después de redimensionar: {img_redim.size} (debe ser 28x28)")
        
        # Paso crítico 2: Invertir colores (MNIST usa fondo negro)
        img_invertida = ImageOps.invert(img_redim)
        
        # Debug 3: Mostrar valores antes/después de invertir
        img_array_antes = np.array(img_redim).astype('float32') / 255.0
        img_array_despues = np.array(img_invertida).astype('float32') / 255.0
        print(f"Min/Max antes de invertir: {np.min(img_array_antes):.2f}, {np.max(img_array_antes):.2f}")
        print(f"Min/Max después de invertir: {np.min(img_array_despues):.2f}, {np.max(img_array_despues):.2f}")
        
        # Preparar array para el modelo
        img_final = img_array_despues.reshape(1, 28, 28, 1)
        
        # Debug 4: Verificar forma del array
        print(f"Forma del array de entrada: {img_final.shape}")
        
        # Predicción
        prediction = modelo.predict(img_final)
        print("Predicción cruda (probabilidades):", prediction)
        
        # Resultado
        resultado = int(np.argmax(prediction))
        print(f"Predicción final: {resultado}\n--- Fin de solicitud ---\n")
        
        return jsonify({'prediction': resultado})
        
    except Exception as e:
        print(f"\n!!! Error: {str(e)}\n", flush=True) 
        import traceback
        traceback.print_exc()  # <- Esto imprime todo el error con detalles
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Debug 5: Verificar carga del modelo
    print("Resumen del modelo cargado:")
    modelo.summary()
    app.run(debug=False)