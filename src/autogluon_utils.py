import os
import pandas as pd
import mlflow
import shutil

def train_model(train_data: pd.DataFrame, target: str, run_name: str, time_limit: int = 60, presets: str = 'medium_quality'):
    """
    Trains an AutoGluon model and logs results to MLflow using generic artifact logging.
    """
    from autogluon.tabular import TabularPredictor
    
    mlflow.set_experiment("AutoGluon_Experiments")
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("target", target)
        mlflow.log_param("time_limit", time_limit)
        mlflow.log_param("presets", presets)
        
        # Output directory for AutoGluon
        model_path = os.path.join("models", run_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            
        # Train model
        predictor = TabularPredictor(label=target, path=model_path).fit(
            train_data, 
            time_limit=time_limit, 
            presets=presets
        )
        
        # Log metrics (leaderboard)
        leaderboard = predictor.leaderboard(train_data, silent=True)
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
