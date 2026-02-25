import os
import pandas as pd
import mlflow
import shutil
from flaml import AutoML
import matplotlib.pyplot as plt
import time
from src.mlflow_utils import safe_set_experiment

def train_flaml_model(train_data: pd.DataFrame, target: str, run_name: str, time_budget: int = 60, task: str = 'classification', metric: str = 'auto', estimator_list: list = 'auto'):
    """
    Trains a FLAML model and logs results to MLflow.
    """
    safe_set_experiment("FLAML_Experiments")
    print(f"Iniciando treinamento FLAML para a run: {run_name}")
    
    with mlflow.start_run(run_name=run_name) as run:
        # Data cleaning: drop rows where target is NaN
        train_data = train_data.dropna(subset=[target])
        print(f"Dados limpos: {len(train_data)} linhas restantes.")
        
        # Log parameters
        mlflow.log_param("target", target)
        mlflow.log_param("time_budget", time_budget)
        mlflow.log_param("task", task)
        mlflow.log_param("metric", metric)
        mlflow.log_param("estimator_list", str(estimator_list))
        
        X_train = train_data.drop(columns=[target])
        y_train = train_data[target]
        
        automl = AutoML()
        
        # Determine the task-specific callback
        # FLAML's 'callbacks' in settings are for the AutoML/Tune level, not the estimator level.
        # The error happened because FLAML passed the callback down to LightGBM which expects a callable.
        
        # We will use a custom logger instead of a callback to track performance_history.csv
        # by monitoring the flaml.log file or using the 'on_trial_result' correctly.
        
        settings = {
            "time_budget": time_budget,
            "metric": metric,
            "task": task,
            "estimator_list": estimator_list,
            "log_file_name": "flaml.log",
            "seed": 42,
            "n_jobs": 1,
            "verbose": 3,
        }
        
        # Train model
        print("Executando automl.fit()...")
        try:
            automl.fit(X_train=X_train, y_train=y_train, **settings)
            print("Treinamento finalizado com sucesso.")
        except StopIteration:
            print("Treinamento interrompido por StopIteration (orçamento esgotado).")
            if not hasattr(automl, 'best_estimator') or automl.best_estimator is None:
                raise RuntimeError("FLAML parou prematuramente sem encontrar um modelo.")
        
        # Log metrics
        if hasattr(automl, 'best_loss'):
            mlflow.log_metric("best_loss", automl.best_loss)
            print(f"Melhor perda encontrada: {automl.best_loss}")
        
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
