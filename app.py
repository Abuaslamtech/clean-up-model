import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# start flask
app = Flask(__name__)

# Load the model
model_path = 'model.keras'
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ['glass', 'metal', 'paper', 'plastic']  # Update this based on your model's classes

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400

        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((224, 224))  # Resize to match model input size
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0  # Normalize

        # Make the prediction
        prediction = model.predict(image_array)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Waste Classification API'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))