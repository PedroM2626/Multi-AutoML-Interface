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

def generate_api_deployment(model_type, run_id, target_column, output_dir="deploy"):
    """
    Generates a ready-to-use FastAPI deployment package for the model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate main.py (FastAPI App)
    if model_type == "autogluon":
        load_code = f"""
from autogluon.tabular import TabularPredictor
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")
model = TabularPredictor.load(local_path)

def _predict(data: pd.DataFrame):
    return model.predict(data).tolist()
"""
    elif model_type == "flaml":
        load_code = f"""
import pickle
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")
model = None
for root, dirs, files in os.walk(local_path):
    for file in files:
        if file.endswith(".pkl"):
            with open(os.path.join(root, file), "rb") as f:
                model = pickle.load(f)
            break
if model is None:
    raise FileNotFoundError("FLAML model not found.")

def _predict(data: pd.DataFrame):
    return model.predict(data).tolist()
"""
    elif model_type == "h2o":
        load_code = f"""
import h2o
h2o.init()
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")
model = None
for root, dirs, files in os.walk(local_path):
    for file in files:
        if file.endswith(".zip") or not "." in file:
            model = h2o.load_model(os.path.join(root, file))
            break

def _predict(data: pd.DataFrame):
    hf = h2o.H2OFrame(data)
    preds = model.predict(hf)
    return preds.as_data_frame()['predict'].tolist()
"""
    elif model_type == "tpot":
        load_code = f"""
import mlflow.sklearn
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

def _predict(data: pd.DataFrame):
    return model.predict(data).tolist()
"""
    else:
        load_code = f"""
# Unsupported model type for automatic API generation
model = None
def _predict(data: pd.DataFrame):
    return []
"""

    main_py_code = f"""from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import os

app = FastAPI(title="AutoML Generated API", version="1.0")

# --- Model Loading Logic ---
{load_code}
# ---------------------------

@app.get("/")
def health_check():
    return {{"status": "API is running", "model_type": "{model_type}", "run_id": "{run_id}"}}

@app.post("/predict")
def predict(payload: dict):
    try:
        # Convert payload to DataFrame (assumes a list of records or dict of lists)
        if isinstance(payload, dict):
            # Try to handle {{ "data": [...] }}
            if "data" in payload:
                df = pd.DataFrame(payload["data"])
            else:
                df = pd.DataFrame([payload])
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise ValueError("Payload must be a dictionary or list of dictionaries.")
            
        predictions = _predict(df)
        return {{"predictions": predictions}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    with open(os.path.join(output_dir, "main.py"), "w") as f:
        f.write(main_py_code)
        
    # 2. Generate requirements.txt
    reqs = f"""fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.2
pandas==2.1.4
mlflow==2.9.2
"""
    if model_type == "autogluon":
        reqs += "autogluon==1.0.0\n"
    elif model_type == "flaml":
        reqs += "flaml==2.1.2\n"
    elif model_type == "h2o":
        reqs += "h2o==3.44.0.3\n"
    elif model_type == "tpot":
        reqs += "tpot==0.12.2\nscikit-learn==1.4.2\n"

    with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
        f.write(reqs)
        
    # 3. Generate Dockerfile
    dockerfile = f"""FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (like Java for H2O if needed, or build essentials)
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libgomp1 \\
    libgl1 \\
    python3-dev \\
    default-jre \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile)
        
    # 4. Generate a simple README
    readme = f"""# API Deployment Package

This package was automatically generated for the model from run `{run_id}` ({model_type}).

## How to use locally:
1. `pip install -r requirements.txt`
2. `python main.py` or `uvicorn main:app --host 0.0.0.0 --port 8000`
3. Access `http://localhost:8000/docs` to test the API.

## How to run via Docker:
1. `docker build -t ml-api:{run_id[:8]} .`
2. `docker run -p 8000:8000 ml-api:{run_id[:8]}`

## Example Request:
POST to `http://localhost:8000/predict`
```json
{{
    "data": [
        {{
            "feature1": 1.5,
            "feature2": "value"
        }}
    ]
}}
```
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)
        
    return output_dir
