import os
import pandas as pd
import mlflow
import shutil
import logging
from src.mlflow_utils import safe_set_experiment

logger = logging.getLogger(__name__)

def train_model(train_data: pd.DataFrame, target: str, run_name: str, 
                valid_data: pd.DataFrame = None, test_data: pd.DataFrame = None, 
                time_limit: int = 60, presets: str = 'medium_quality', seed: int = 42, cv_folds: int = 0):
    """
    Trains an AutoGluon model and logs results to MLflow using generic artifact logging.
    """
    from autogluon.tabular import TabularPredictor
    
    safe_set_experiment("AutoGluon_Experiments")
    
    with mlflow.start_run(run_name=run_name) as run:
        # Data cleaning: drop rows where target is NaN
        train_data = train_data.dropna(subset=[target])
        
        # Log parameters
        mlflow.log_param("target", target)
        mlflow.log_param("time_limit", time_limit)
        mlflow.log_param("presets", presets)
        mlflow.log_param("seed", seed)
        
        # Output directory for AutoGluon
        model_path = os.path.join("models", run_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            
        # Clean validation and test formats if present
        if valid_data is not None:
            if target not in valid_data.columns:
                raise ValueError(f"A coluna alvo '{target}' não foi encontrada nos dados de Validação. Certifique-se de que o arquivo de validação possui a mesma estrutura que o arquivo de treino.")
            valid_data = valid_data.dropna(subset=[target])
            mlflow.log_param("has_validation_data", True)
        if test_data is not None:
            if target not in test_data.columns:
                raise ValueError(f"A coluna alvo '{target}' não foi encontrada nos dados de Teste. Certifique-se de que o test set possui a variável alvo.")
            test_data = test_data.dropna(subset=[target])
            mlflow.log_param("has_test_data", True)
            
        # Train model
        fit_args = {
            "train_data": train_data,
            "time_limit": time_limit, 
            "presets": presets
        }
        if cv_folds > 0:
            fit_args["num_bag_folds"] = cv_folds
            
        if valid_data is not None and cv_folds == 0:
            fit_args["tuning_data"] = valid_data
            
        predictor = TabularPredictor(label=target, path=model_path).fit(**fit_args)
        
        # Log metrics (leaderboard)
        # Se test_data for fornecido, a leaderboard e scorage fará uso rigoroso dele,
        # senão fallback para o de treino (o autogluon usa valid internamente, mas leaderboard explicito ganha precisão)
        eval_data = test_data if test_data is not None else (valid_data if valid_data is not None else train_data)
        leaderboard = predictor.leaderboard(eval_data, silent=True)
        # Log the best model's score
        best_model_score = leaderboard.iloc[0]['score_val']
        mlflow.log_metric("best_model_score", best_model_score)
        
        # Save leaderboard as artifact
        leaderboard_path = "leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(leaderboard_path)
        if os.path.exists(leaderboard_path):
            os.remove(leaderboard_path)
        
        # Log AutoGluon model directory as a generic artifact
        # This avoids all ModuleNotFoundError issues with mlflow.autogluon
        mlflow.log_artifacts(model_path, artifact_path="model")
        mlflow.log_param("model_type", "autogluon")
        
        return predictor, run.info.run_id

def load_model_from_mlflow(run_id: str):
    """
    Loads a model from MLflow artifacts.
    """
    import mlflow
    from autogluon.tabular import TabularPredictor
    
    # Download the artifact folder
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    
    # Load the predictor from the local path
    predictor = TabularPredictor.load(local_path)
    return predictor

def get_leaderboard(predictor):
    return predictor.leaderboard(silent=True)
