from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/json')
def json_response():
    return {"message": "Hello, World!"}

# Load the model
model = joblib.load('afib_model.pkl')  # Ensure this file is in the same directory

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Extract features
        heart_rate = data['HeartRate']
        spo2 = data['SpO2']
        temperature = data['Temperature']
        
        # Prepare the feature array
        features = np.array([[heart_rate, spo2, temperature]])
        
        # Make prediction
        prediction = model.predict(features)
        
        return jsonify({'prediction': int(prediction[0])})  # Convert prediction to int
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)