import os
import mlflow


def generate_consumption_code(model_type: str, run_id: str, target_column) -> str:
    """
    Generates a Python code snippet to load and run predictions with the trained model.
    Supports: autogluon, flaml, h2o, tpot, pycaret, lale.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        task_type = run.data.params.get("task_type", "Classification")
        data_category = run.data.params.get("data_category", "Tabular")
    except Exception:
        task_type = "Classification"
        data_category = "Tabular"

    base_code = f"""# Sample code to consume the trained model
# Run ID: {run_id}
# Model Type: {model_type}
# Task Type: {task_type}

import os
import pandas as pd
import mlflow
"""

    if model_type == "autogluon":
        if data_category == "Multimodal" or task_type.startswith("Computer Vision"):
            predictor_import = "from autogluon.multimodal import MultiModalPredictor"
            predictor_loader = "MultiModalPredictor.load(local_path)"
        elif data_category == "Tabular" and task_type == "Multi-Label Classification":
            return base_code + f"""
import glob
from autogluon.tabular import TabularPredictor

# 1. Download model from MLflow
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")

# 2. Load one TabularPredictor per target
predictors = {{}}
for target_dir in glob.glob(os.path.join(local_path, "*")):
    if os.path.isdir(target_dir):
        target_name = os.path.basename(target_dir)
        predictors[target_name] = TabularPredictor.load(target_dir)

# 3. Predict each target independently
# data = pd.read_csv("your_data.csv")
# multi_target_predictions = {{name: predictor.predict(data) for name, predictor in predictors.items()}}
# print(pd.DataFrame(multi_target_predictions))
"""
        else:
            predictor_import = "from autogluon.tabular import TabularPredictor"
            predictor_loader = "TabularPredictor.load(local_path)"

        return base_code + f"""
{predictor_import}

# 1. Download model from MLflow
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")

# 2. Load model
predictor = {predictor_loader}

# 3. Predict
# data = pd.read_csv("your_data.csv")
# predictions = predictor.predict(data)
# print(predictions)
"""

    elif model_type == "flaml":
        return base_code + f"""
import pickle

# 1. Download model from MLflow
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")

# 2. Load the .pkl file
model = None
for root, dirs, files in os.walk(local_path):
    for f in files:
        if f.endswith(".pkl"):
            with open(os.path.join(root, f), "rb") as fh:
                model = pickle.load(fh)
            break

if model is None:
    raise FileNotFoundError("Model .pkl not found in artifacts.")

# 3. Predict
# data = pd.read_csv("your_data.csv")
# predictions = model.predict(data)
# print(predictions)
"""

    elif model_type == "h2o":
        return base_code + f"""
import h2o

# 1. Initialize H2O
h2o.init()

# 2. Download model from MLflow
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")

# 3. Load the H2O model
model = None
for root, dirs, files in os.walk(local_path):
    for f in files:
        if f.endswith(".zip") or "." not in f:
            model = h2o.load_model(os.path.join(root, f))
            break

# 4. Predict
# h2o_frame = h2o.H2OFrame(pd.read_csv("your_data.csv"))
# predictions = model.predict(h2o_frame)
# print(predictions.as_data_frame())
"""

    elif model_type == "tpot":
        return base_code + f"""
import mlflow.sklearn

# 1. Load model directly from MLflow
model = mlflow.sklearn.load_model("runs:/{run_id}/model")

# 2. Predict
# data = pd.read_csv("your_data.csv")
# predictions = model.predict(data)
# print(predictions)
"""

    elif model_type == "pycaret":
        if task_type == "Regression":
            pc_module = "pycaret.regression"
        elif task_type == "Time Series Forecasting":
            pc_module = "pycaret.time_series"
        elif task_type == "Anomaly Detection":
            pc_module = "pycaret.anomaly"
        elif task_type == "Clustering":
            pc_module = "pycaret.clustering"
        else:
            pc_module = "pycaret.classification"

        return base_code + f"""
import joblib
from {pc_module} import load_model, predict_model

# 1. Download model artifact from MLflow
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")

# 2. Find and load the PyCaret .pkl file
model_path = None
for root, dirs, files in os.walk(local_path):
    for f in files:
        if f.endswith(".pkl"):
            model_path = os.path.join(root, f).replace(".pkl", "")
            break

if model_path is None:
    raise FileNotFoundError("PyCaret model .pkl not found in artifacts.")

model = load_model(model_path)

# 3. Predict
# data = pd.read_csv("your_data.csv")  # For classification/regression, must NOT contain target column
# predictions = predict_model(model, data=data)
# print(predictions)
"""

    elif model_type == "lale":
        return base_code + f"""
import joblib
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# 1. Download model artifact from MLflow
local_path = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")

# 2. Find and load the Lale joblib bundle
bundle = None
for root, dirs, files in os.walk(local_path):
    for f in files:
        if f.endswith(".pkl"):
            bundle = joblib.load(os.path.join(root, f))
            break

if bundle is None:
    raise FileNotFoundError("Lale model .pkl not found in artifacts.")

model        = bundle["model"]
col_encoders = bundle.get("col_encoders", {{}})
y_encoder    = bundle.get("y_encoder", None)

# 3. Preprocess and Predict
# data = pd.read_csv("your_data.csv")  # must NOT contain target column
#
# for col, enc in col_encoders.items():
#     data[col] = enc.transform(data[[col]]).ravel()
#
# raw_preds = model.predict(data.values)
#
# if y_encoder is not None:
#     predictions = y_encoder.inverse_transform(raw_preds)
# else:
#     predictions = raw_preds
#
# print(predictions)
"""

    else:
        return base_code + f"""
# Code generation for '{model_type}' is not explicitly implemented.
# Try loading via: mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")
"""


def _load_code_for_deploy(model_type: str, run_id: str) -> str:
    """Returns the model-loading block used in the FastAPI main.py."""
    if model_type == "autogluon":
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            task_type = run.data.params.get("task_type", "Classification")
            data_category = run.data.params.get("data_category", "Tabular")
        except Exception:
            task_type = "Classification"
            data_category = "Tabular"

        if data_category == "Multimodal" or task_type.startswith("Computer Vision"):
            import_block = "from autogluon.multimodal import MultiModalPredictor"
            load_block = "model = MultiModalPredictor.load(_local)"
        else:
            import_block = "from autogluon.tabular import TabularPredictor"
            load_block = "model = TabularPredictor.load(_local)"

        return f"""
{import_block}
import mlflow
_local = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")
{load_block}

def _predict(df):
    return model.predict(df).tolist()
"""
    elif model_type == "flaml":
        return f"""
import pickle, os, mlflow
_local = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")
model = None
for root, _, files in os.walk(_local):
    for f in files:
        if f.endswith(".pkl"):
            with open(os.path.join(root, f), "rb") as fh:
                model = pickle.load(fh)
            break
if model is None:
    raise FileNotFoundError("FLAML model not found.")

def _predict(df):
    return model.predict(df).tolist()
"""
    elif model_type == "h2o":
        return f"""
import h2o, os, mlflow
h2o.init()
_local = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")
model = None
for root, _, files in os.walk(_local):
    for f in files:
        if f.endswith(".zip") or "." not in f:
            model = h2o.load_model(os.path.join(root, f))
            break

def _predict(df):
    hf = h2o.H2OFrame(df)
    return model.predict(hf).as_data_frame()["predict"].tolist()
"""
    elif model_type == "tpot":
        return f"""
import mlflow.sklearn
model = mlflow.sklearn.load_model("runs:/{run_id}/model")

def _predict(df):
    return model.predict(df).tolist()
"""
    elif model_type == "pycaret":
        return f"""
import os, mlflow, joblib
import pandas as pd
_local = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")

try:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run("{run_id}")
    task_type = run.data.params.get("task_type", "Classification")
except Exception:
    task_type = "Classification"

if task_type == "Regression":
    from pycaret.regression import load_model, predict_model
elif task_type == "Time Series Forecasting":
    from pycaret.time_series import load_model, predict_model
elif task_type == "Anomaly Detection":
    from pycaret.anomaly import load_model, predict_model
elif task_type == "Clustering":
    from pycaret.clustering import load_model, predict_model
else:
    from pycaret.classification import load_model, predict_model

_mpath = None
for root, _, files in os.walk(_local):
    for f in files:
        if f.endswith(".pkl"):
            _mpath = os.path.join(root, f).replace(".pkl", "")
            break
if _mpath is None:
    raise FileNotFoundError("PyCaret model not found.")
model = load_model(_mpath)

def _predict(df):
    preds = predict_model(model, data=df)
    if task_type == "Anomaly Detection" and "Anomaly" in preds.columns:
        return preds["Anomaly"].tolist()
    if task_type == "Clustering" and "Cluster" in preds.columns:
        return preds["Cluster"].tolist()
    if task_type == "Classification" and "prediction_label" in preds.columns:
        return preds["prediction_label"].tolist()
    else:
        # For regression or time series, it might return 'prediction_label' or just predictions
        if "prediction_label" in preds.columns:
            return preds["prediction_label"].tolist()
        return preds.iloc[:, 0].tolist()
"""
    elif model_type == "lale":
        return f"""
import os, mlflow, joblib
import numpy as np
_local = mlflow.artifacts.download_artifacts(run_id="{run_id}", artifact_path="model")
_bundle = None
for root, _, files in os.walk(_local):
    for f in files:
        if f.endswith(".pkl"):
            _bundle = joblib.load(os.path.join(root, f))
            break
if _bundle is None:
    raise FileNotFoundError("Lale model not found.")
_model        = _bundle["model"]
_col_encoders = _bundle.get("col_encoders", {{}})
_y_encoder    = _bundle.get("y_encoder", None)

def _predict(df):
    import pandas as _pd
    df = _pd.DataFrame(df)
    for col, enc in _col_encoders.items():
        if col in df.columns:
            df[col] = enc.transform(df[[col]]).ravel()
    raw = _model.predict(df.values)
    if _y_encoder is not None:
        return _y_encoder.inverse_transform(raw).tolist()
    return raw.tolist()
"""
    else:
        return """
model = None
def _predict(df):
    return []
"""


def generate_api_deployment(model_type: str, run_id: str, target_column: str, output_dir: str = "deploy") -> str:
    """
    Generates a ready-to-use FastAPI + Docker deployment package for the model.
    Supports: autogluon, flaml, h2o, tpot, pycaret, lale.
    """
    os.makedirs(output_dir, exist_ok=True)

    load_code = _load_code_for_deploy(model_type, run_id)

    main_py = f"""from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI(title="AutoML Generated API - {model_type}", version="1.0")

# --- Model Loading ---
{load_code}
# ---------------------

@app.get("/")
def health():
    return {{"status": "running", "model": "{model_type}", "run_id": "{run_id}"}}

@app.post("/predict")
def predict(payload: dict):
    try:
        if "data" in payload:
            df = pd.DataFrame(payload["data"])
        else:
            df = pd.DataFrame([payload])
        return {{"predictions": _predict(df)}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

    with open(os.path.join(output_dir, "main.py"), "w", encoding="utf-8") as f:
        f.write(main_py)

    # requirements.txt
    base_reqs = """fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.2
pandas==2.1.4
mlflow==2.9.2
"""
    extra = {
        "autogluon": "autogluon==1.0.0\n",
        "flaml": "flaml==2.1.2\n",
        "h2o": "h2o==3.44.0.3\n",
        "tpot": "tpot==0.12.2\nscikit-learn==1.2.2\n",
        "pycaret": "pycaret==3.3.0\nscikit-learn==1.2.2\nscipy==1.11.4\n",
        "lale": "lale==0.9.1\nscikit-learn==1.2.2\njoblib\nhyperopt\n",
    }
    reqs = base_reqs + extra.get(model_type, "")
    with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(reqs)

    # Dockerfile
    dockerfile = f"""FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    build-essential libgomp1 libgl1 python3-dev default-jre curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    with open(os.path.join(output_dir, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(dockerfile)

    # README
    readme = f"""# API Deployment — {model_type} (Run: {run_id})

## Local
```bash
pip install -r requirements.txt
python main.py
```

## Docker
```bash
docker build -t ml-api:{run_id[:8]} .
docker run -p 8000:8000 ml-api:{run_id[:8]}
```

## Example request
```json
POST http://localhost:8000/predict
{{
  "data": [{{"feature1": 1.5, "feature2": "value"}}]
}}
```
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    return output_dir
