import os
import shutil
import time
import logging
import mlflow

logger = logging.getLogger(__name__)

def heal_mlruns(mlruns_path="mlruns", min_age_seconds=3600):
    """
    Moves experiment directories that are missing meta.yaml into mlruns/.trash so the
    local store can be read again. Directories touched in the last hour are left alone:
    in a multi-session deployment another worker may be creating that experiment right
    now, and deleting it would destroy a run in flight.
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
                try:
                    age = time.time() - os.path.getmtime(item_path)
                except OSError as e:
                    logger.warning(f"Cannot inspect {item_path}: {e}")
                    continue
                if age < min_age_seconds:
                    logger.info(f"Leaving {item_path} alone: written {int(age)}s ago, may belong to a running experiment")
                    continue

                trash_path = os.path.join(mlruns_path, ".trash")
                os.makedirs(trash_path, exist_ok=True)
                destination = os.path.join(trash_path, f"{item}_{int(time.time())}")
                logger.warning(f"Quarantining malformed experiment {item_path} to {destination}")
                try:
                    shutil.move(item_path, destination)
                except Exception as e:
                    logger.error(f"Error moving {item_path}: {e}")

def safe_set_experiment(experiment_name):
    """
    Point MLflow at the configured backend and select the experiment.

    An explicit MLFLOW_TRACKING_URI takes precedence: that is how a multi-session
    deployment shares one store (server or database) instead of a per-container
    directory. Without it the app keeps its local-first default of ./mlruns, which
    MLflow 3 only accepts once MLFLOW_ALLOW_FILE_STORE is set - otherwise every call
    raises and no run is ever recorded.
    """
    try:
        import mlflow
        import os

        configured_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if configured_uri:
            tracking_uri = configured_uri
        else:
            # Configure tracking URI to project directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mlruns_path = os.path.join(project_root, "mlruns")

            # Ensure directory and trash exist
            os.makedirs(mlruns_path, exist_ok=True)
            os.makedirs(os.path.join(mlruns_path, ".trash"), exist_ok=True)

            normalized_path = mlruns_path.replace('\\', '/')
            tracking_uri = f"file:///{normalized_path}"
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

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
