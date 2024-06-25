# app.py
from venv import logger
from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
from waitress import serve
import numpy as np
import joblib
import pandas as pd
import tensorflow
import keras
import logging
import os

# intiating Flask
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# File management system 
@app.route('/')
def hello_world():
    return jsonify(message="Hello, World!")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(message='No file part'), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(message='No selected file'), 400

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        scaler = joblib.load('./model/scaler.joblib')
        df = np.array(df)
        df = df.reshape(-1, 1) 
        df = scaler.transform(df)
        
        x = []
        for i in range(50, df.shape[0]):
            x.append(df[i-50:i, 0]) 
        # logger.info(x)
        x = np.array(x) 

        # logger.info(x)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        # for filename in os.listdir("./model"):
            
            # if filename.endswith('.h5') and 'retrained' in filename:
            #     model = load_model("./model/" + filename)
            # else:
        model = load_model("./model/stock_prediction.h5")

        # logger.info("dsds",x)

        predictions = model.predict(x)
        predictions = scaler.inverse_transform(predictions)
        logger.info(predictions.shape)
        logger.info(predictions)
        
        list_data = predictions.tolist()

        return jsonify(list_data)

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    stock_ticker_symbol = data['stock_ticker_symbol']
    start_date = data['start_date']
    end_date = data['end_date']
    return jsonify(message="Hello, Train!")


if __name__ == '__main__':
    # serve(app, host="0.0.0.0", port=5002, expose_tracebacks=True)
    app.run(debug=True, host="0.0.0.0", port=5002)
