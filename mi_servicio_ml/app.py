from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io

# 1. Inicializar la aplicación Flask
app = Flask(__name__)

# 2. Cargar el modelo preentrenado (se carga una sola vez al inicio)
model = MobileNetV2(weights='imagenet')
print("Modelo cargado exitosamente.")

def prepare_image(img_file):
    """Preprocesa una imagen para el modelo MobileNetV2."""
    img = Image.open(io.BytesIO(img_file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

# 3. Definir el endpoint de predicción
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró ningún archivo"}), 400
    file = request.files['file']
    try:
        processed_image = prepare_image(file)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        result = [{"label": label, "probability": float(prob)} for (_, label, prob) in decoded_predictions]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)