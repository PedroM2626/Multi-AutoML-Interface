import os
import shutil
import logging
import mlflow

logger = logging.getLogger(__name__)

def heal_mlruns(mlruns_path="mlruns"):
    """
    Removes experiment directories that are missing meta.yaml to prevent MLflow crashes.
    """
    if not os.path.exists(mlruns_path):
        os.makedirs(mlruns_path, exist_ok=True)
        os.makedirs(os.path.join(mlruns_path, ".trash"), exist_ok=True)
        return

    for item in os.listdir(mlruns_path):
        item_path = os.path.join(mlruns_path, item)
        if os.path.isdir(item_path) and item.isdigit():
            meta_path = os.path.join(item_path, "meta.yaml")
            if not os.path.exists(meta_path):
                logger.warning(f"Removing malformed experiment: {item_path}")
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    logger.error(f"Error removing {item_path}: {e}")

def safe_set_experiment(experiment_name):
    """Safely set MLflow experiment"""
    try:
        import mlflow
        import os
        
        # Configure tracking URI to project directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mlruns_path = os.path.join(project_root, "mlruns")
        
        # Ensure directory and trash exist
        os.makedirs(mlruns_path, exist_ok=True)
        os.makedirs(os.path.join(mlruns_path, ".trash"), exist_ok=True)
        
        # Configure tracking URI
        normalized_path = mlruns_path.replace('\\', '/')
        tracking_uri = f"file:///{normalized_path}"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracking URI configured to: {tracking_uri}")
        logger.info(f"Experiment '{experiment_name}' configured successfully")
        
    except Exception as e:
        logger.error(f"Error configuring MLflow experiment: {e}")
        if "MissingConfigException" in str(type(e)) or "meta.yaml" in str(e):
            heal_mlruns()
            mlflow.set_experiment(experiment_name)
        else:
            raise e
