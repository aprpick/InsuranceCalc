# C:\Users\14037\Documents\InsuranceCalc\Model Training\Scripts\2_Model_Train.py

import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.model import Model
import pickle

# Load configuration
config_path = r"C:\Users\14037\Documents\InsuranceCalc\Model Training\Dependencies\config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Connect to Azure ML workspace
ws = Workspace.from_config(path=config_path)

# Create or get environment
env = Environment(name="insurance_env")
env.python.user_managed_dependencies = True
env.python.interpreter_path = r"C:\Users\14037\Documents\InsuranceCalc\azureml_env_38\Scripts\python.exe"
env.register(workspace=ws)

# Define training function
def train_model():
    # Load data from specified CSV location
    csv_path = r"C:\Users\14037\Documents\InsuranceCalc\Work Product\2_Processed_Data.csv"
    data = pd.read_csv(csv_path)
    
    # Prepare features and target
    X = data.drop('charges', axis=1)
    y = data['charges']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R2 score
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Save model to specified location
    output_dir = r"C:\Users\14037\Documents\InsuranceCalc\Work Product"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "3_Trained_Model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model, r2

def main():
    # Create experiment
    experiment_name = "insurance_cost_prediction"
    experiment = Experiment(workspace=ws, name=experiment_name)
    
    # Configure the training job
    config = ScriptRunConfig(
        source_directory=r"C:\Users\14037\Documents\InsuranceCalc\Model Training\Dependencies",
        script="score.py",
        compute_target="local",
        environment=env
    )
    
    # Submit the experiment
    run = experiment.submit(config)
    run.wait_for_completion(show_output=True)
    
    # Train model locally and get R2
    model, r2_score_value = train_model()
    
    # Log metrics
    run.log("R2 Score", r2_score_value)
    
    # Register the model
    Model.register(
        workspace=ws,
        model_path=r"C:\Users\14037\Documents\InsuranceCalc\Work Product\3_Trained_Model.pkl",
        model_name="insurance_cost_model",
        description="Linear Regression model for insurance cost prediction",
        tags={"R2": str(r2_score_value)}
    )
    
    print(f"Model trained and registered. R2 Score: {r2_score_value}")

if __name__ == "__main__":
    main()