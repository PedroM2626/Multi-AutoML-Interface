import os
import pandas as pd
import mlflow
import shutil
from flaml import AutoML
import matplotlib.pyplot as plt

def train_flaml_model(train_data: pd.DataFrame, target: str, run_name: str, time_budget: int = 60, task: str = 'classification', metric: str = 'auto', estimator_list: list = 'auto'):
    """
    Trains a FLAML model and logs results to MLflow.
    """
    mlflow.set_experiment("FLAML_Experiments")
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("target", target)
        mlflow.log_param("time_budget", time_budget)
        mlflow.log_param("task", task)
        mlflow.log_param("metric", metric)
        mlflow.log_param("estimator_list", str(estimator_list))
        
        X_train = train_data.drop(columns=[target])
        y_train = train_data[target]
        
        automl = AutoML()
        settings = {
            "time_budget": time_budget,
            "metric": metric,
            "task": task,
            "estimator_list": estimator_list,
            "log_file_name": "flaml.log",
            "seed": 42,
        }
        
        # Train model
        automl.fit(X_train=X_train, y_train=y_train, **settings)
        
        # Log metrics
        mlflow.log_metric("best_loss", automl.best_loss)
        
        # Save best model
        model_path = os.path.join("models", f"flaml_{run_name}.pkl")
        os.makedirs("models", exist_ok=True)
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump(automl, f)
            
        # Log as artifact
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_param("model_type", "flaml")
        
        # Log training log as artifact
        if os.path.exists("flaml.log"):
            mlflow.log_artifact("flaml.log")
            
        return automl, run.info.run_id

def load_flaml_model(run_id: str):
    import mlflow
    import pickle
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    # Find the .pkl file in the downloaded folder
    for root, dirs, files in os.walk(local_path):
        for file in files:
            if file.endswith(".pkl"):
                with open(os.path.join(root, file), "rb") as f:
                    return pickle.load(f)
    raise FileNotFoundError("Modelo FLAML não encontrado nos artefatos.")
