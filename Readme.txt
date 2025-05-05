######################################################

INSURANCE COST PREDICTION WORKFLOW

######################################################

OVERVIEW

This document outlines the workflow for developing, training, and deploying a machine learning model for insurance cost prediction. Each step includes a brief synopsis and the key methods used.

1️⃣  DATA PREPARATION (FEATURE ENGINEERING)
📍 Local Machine (Python Script)

Script: 1_Feature_Engineering.py

Purpose: Reads, preprocesses, and encodes the dataset.

Key Actions:

Selects CSV file from C:\Users\14037\Documents\InsuranceCalc\Work Product

Renames it to 1_Raw_Data.csv

One-hot encodes categorical variables using pd.get_dummies()

Saves processed data as 2_Processed_Data.csv

2️⃣  MODEL TRAINING (AZURE ML PIPELINE)
📍 Azure ML (Local Execution)

Script: 2_Model_Train.py

Purpose: Trains a Linear Regression model and logs metrics.

Key Actions:

Loads data from 2_Processed_Data.csv

Uses Scikit-Learn LinearRegression

Calculates R² score using r2_score()

Saves model as 3_Trained_Model.pkl

Logs experiment with Azure ML SDK

Registers model in Azure:

Model.register(workspace=ws, model_path="3_Trained_Model.pkl", model_name="insurance_cost_model")

3️⃣  MODEL DEPLOYMENT WITH FLASK (LOCAL TESTING)
📍 Dockerized Flask App (Local)

Script: app.py

Purpose: Serves predictions via an API.

Key Actions:

Loads 3_Trained_Model.pkl

Accepts user input, processes data, and predicts

Uses Flask to create a web app

Defines API route:

@app.route('/', methods=['POST'])

Serves a web UI (index.html) with form inputs

4️⃣  CREATE DOCKER CONTAINER
📍 Docker Desktop

Files: Dockerfile, requirements.txt

Purpose: Containerize Flask app for portability.

Key Actions:

Creates Dockerfile with:

FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]

Builds Docker image:

docker build -t insurance-model-app .

Runs container locally:

docker run -p 5000:5000 insurance-model-app

5️⃣  PUSH CONTAINER TO DOCKER HUB
📍 Docker Hub

Purpose: Store container image for Azure deployment.

Key Actions:

Tag image:

docker tag insurance-model-app adam56465484/insurance-model-app:latest

Push image:

docker push adam56465484/insurance-model-app:latest

6️⃣  DEPLOY ON AZURE CONTAINER APPS
📍 Azure Cloud

Purpose: Host containerized app in the cloud.

Key Actions:

Create Azure Container App

Use Docker Hub Image (adam56465484/insurance-model-app:latest)

Expose port 5000

Enable public access

Deploy & test at:

https://insurance-app.region.azurecontainerapps.io

✅ FINAL OUTPUT

Users can access a web-based Insurance Cost Prediction app via Azure Container Apps.

New model updates can be deployed by rebuilding the container and redeploying to Azure.

######################################################

END OF WORKFLOW

######################################################

