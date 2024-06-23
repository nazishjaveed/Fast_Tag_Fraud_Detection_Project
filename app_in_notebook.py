from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from werkzeug.serving import run_simple

# Load the trained model and preprocessor
model = joblib.load('fraud_detection_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Fastag Fraud Detection System"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert JSON to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess the input data
        processed_data = preprocessor.transform(df)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[:, 1]
        prediction = np.where(prediction_proba >= 0.5, 'Fraud', 'Not Fraud')[0]
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction, 'probability': float(prediction_proba[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True, use_evalex=True)
