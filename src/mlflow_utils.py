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
        return

    for item in os.listdir(mlruns_path):
        item_path = os.path.join(mlruns_path, item)
        if os.path.isdir(item_path) and item.isdigit():
            meta_path = os.path.join(item_path, "meta.yaml")
            if not os.path.exists(meta_path):
                logger.warning(f"Removendo experimento malformado: {item_path}")
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    logger.error(f"Erro ao remover {item_path}: {e}")

def safe_set_experiment(experiment_name):
    """
    Sets the experiment, healing mlruns if it fails due to corruption.
    """
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        if "MissingConfigException" in str(type(e)) or "meta.yaml" in str(e):
            heal_mlruns()
            mlflow.set_experiment(experiment_name)
        else:
            raise e
