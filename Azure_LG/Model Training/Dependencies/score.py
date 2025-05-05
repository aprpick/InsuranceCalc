# C:\Users\14037\Documents\InsuranceCalc\Model Training\Dependencies\score.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def init():
    global model
    # Load the model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

def run(raw_data):
    try:
        # Convert json to dataframe
        data = pd.read_json(raw_data)
        # Make prediction
        prediction = model.predict(data)
        # Return prediction as json
        return json.dumps({"prediction": prediction.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})