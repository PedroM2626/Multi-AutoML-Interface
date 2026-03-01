import os

def generate_consumption_code(model_type, run_id, target_column):
    """
    Generates a Python code snippet to consume the trained model.
    """
    
    base_code = f"""# Sample code to consume the trained model
# Run ID: {run_id}
# Model Type: {model_type}

import pandas as pd
import mlflow
"""

    if model_type == "autogluon":
        code = base_code + f"""
from autogluon.tabular import TabularPredictor

# 1. Download model from MLflow
print("Downloading model from MLflow...")
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")

# 2. Load model
predictor = TabularPredictor.load(local_path)

# 3. Prepare data (example)
# data = pd.read_csv("your_data.csv")

# 4. Predict
# predictions = predictor.predict(data)
# print(predictions)
"""
    elif model_type == "flaml":
        code = base_code + f"""
import pickle

# 1. Download model from MLflow
print("Downloading model from MLflow...")
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")

# 2. Find and load the .pkl file
model = None
for root, dirs, files in os.walk(local_path):
    for file in files:
        if file.endswith(".pkl"):
            with open(os.path.join(root, file), "rb") as f:
                model = pickle.load(f)
            break

if model is None:
    raise FileNotFoundError("Model file (.pkl) not found in downloaded artifacts.")

# 3. Predict
# predictions = model.predict(data)
# print(predictions)
"""
    elif model_type == "h2o":
        code = base_code + f"""
import h2o

# 1. Initialize H2O
h2o.init()

# 2. Download model from MLflow
print("Downloading model from MLflow...")
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")

# 3. Find and load the model (H2O saves as directory or zip)
model = None
for root, dirs, files in os.walk(local_path):
    for file in files:
        if file.endswith(".zip") or not "." in file: # H2O models often don't have extensions or are .zip
            model_path = os.path.join(root, file)
            model = h2o.load_model(model_path)
            break

# 4. Predict
# h2o_frame = h2o.H2OFrame(data)
# predictions = model.predict(h2o_frame)
# print(predictions.as_data_frame())
"""
    elif model_type == "tpot":
        code = base_code + f"""
import mlflow.sklearn

# 1. Load model directly from MLflow
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# 2. Predict
# predictions = model.predict(data)
# print(predictions)
"""
    else:
        code = base_code + f"""
# Code generation for {model_type} is not explicitly implemented.
# You can try downloading it via mlflow.artifacts.download_artifacts.
"""

    return code
