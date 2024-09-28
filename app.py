import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model
model_path = '/opt/render/project/src/model.keras'
model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json['data']

        # Convert the input data to a numpy array
        input_data = np.array(data)

        # Make sure the input shape matches what your model expects
        if input_data.shape != model.input_shape[1:]:
            return jsonify({'error': f'Input shape mismatch. Expected {model.input_shape[1:]}, got {input_data.shape}'}), 400

        # Make the prediction
        prediction = model.predict(np.expand_dims(input_data, axis=0))

        # Convert the prediction to a list (for JSON serialization)
        prediction_list = prediction.tolist()

        return jsonify({'prediction': prediction_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))