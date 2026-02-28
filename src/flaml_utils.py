import os
import pandas as pd
import mlflow
import shutil
import logging
from flaml import AutoML
import matplotlib.pyplot as plt
import time
from src.mlflow_utils import safe_set_experiment

logger = logging.getLogger(__name__)

def train_flaml_model(train_data: pd.DataFrame, target: str, run_name: str, 
                      valid_data: pd.DataFrame = None, test_data: pd.DataFrame = None,
                      time_budget: int = 60, task: str = 'classification', metric: str = 'auto', estimator_list: list = 'auto', seed: int = 42, cv_folds: int = 0):
    """
    Trains a FLAML model and logs results to MLflow.
    """
    safe_set_experiment("FLAML_Experiments")
    logging.info(f"Iniciando treinamento FLAML para a run: {run_name}")
    
    # Ensure flaml logger is also at INFO level
    import flaml
    from flaml import AutoML
    flaml_logger = logging.getLogger('flaml')
    flaml_logger.setLevel(logging.INFO)
    
    with mlflow.start_run(run_name=run_name) as run:
        # Data cleaning: drop rows where target is NaN
        train_data = train_data.dropna(subset=[target])
        logging.info(f"Dados prontos: {len(train_data)} linhas.")
        
        # Log parameters
        mlflow.log_param("target", target)
        mlflow.log_param("time_budget", time_budget)
        mlflow.log_param("task", task)
        mlflow.log_param("metric", metric)
        mlflow.log_param("estimator_list", str(estimator_list))
        mlflow.log_param("seed", seed)
        
        X_train = train_data.drop(columns=[target])
        y_train = train_data[target]
        
        X_val, y_val = None, None
        if valid_data is not None:
            if target not in valid_data.columns:
                raise ValueError(f"A coluna alvo '{target}' não foi encontrada nos dados de Validação.")
            valid_data = valid_data.dropna(subset=[target])
            X_val = valid_data.drop(columns=[target])
            y_val = valid_data[target]
            mlflow.log_param("has_validation_data", True)
            
        if test_data is not None:
             if target not in test_data.columns:
                 raise ValueError(f"A coluna alvo '{target}' não foi encontrada nos dados de Teste.")
             mlflow.log_param("has_test_data", True)
        
        automl = AutoML()
        
        # Note: We are NOT using low_cost_partial_config because it causes 
        # TypeError in some estimators (like LGBM) when passed via automl.fit.
        # The 'No low-cost partial config given' message is just an INFO warning from FLAML.

        settings = {
            "time_budget": time_budget,
            "metric": metric,
            "task": task,
            "estimator_list": estimator_list,
            "log_file_name": "flaml.log",
            "seed": seed,
            "n_jobs": 1,
            "verbose": 0, # Reduzir verbosidade interna para evitar poluição, o progresso vai para flaml.log
        }
        
        if cv_folds > 0:
            settings["eval_method"] = "cv"
            settings["n_splits"] = cv_folds
            
        if X_val is not None:
            settings["X_val"] = X_val
            settings["y_val"] = y_val
        
        # Train model
        logging.info("Executando busca de hiperparâmetros (automl.fit)...")
        try:
            automl.fit(X_train=X_train, y_train=y_train, **settings)
            logging.info("Busca finalizada com sucesso.")
        except StopIteration:
            logging.info("Busca interrompida (limite de tempo atingido).")
            if not hasattr(automl, 'best_estimator') or automl.best_estimator is None:
                raise RuntimeError("FLAML parou sem encontrar um modelo válido.")
        
        # Log metrics
        if hasattr(automl, 'best_loss'):
            mlflow.log_metric("best_loss", automl.best_loss)
            logging.info(f"Melhor Loss final: {automl.best_loss:.4f}")
        
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
