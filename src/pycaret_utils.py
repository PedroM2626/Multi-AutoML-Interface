import os
import logging
import traceback
import queue
import time
import pandas as pd
from typing import Dict, Any, Optional

import mlflow

from src.mlflow_utils import safe_set_experiment
from pycaret.classification import setup, compare_models, pull, tune_model, blend_models, save_model


def run_pycaret_experiment(
    train_df: pd.DataFrame,
    target_col: str,
    run_name: str,
    time_limit: Optional[int],
    log_queue: queue.Queue,
    stop_event=None,
    val_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Run PyCaret experiment for classification.
    PyCaret manages its own MLflow internally; we wrap it and pull the run ID after.
    """
    logger = logging.getLogger("pycaret")
    logger.info(f"Starting PyCaret experiment: {run_name}")
    logger.info(f"Dataset shape: {train_df.shape}, Target: {target_col}")

    # Always end any dangling MLflow run to avoid conflicts
    try:
        mlflow.end_run()
    except Exception:
        pass

    # 1. Prepare MLflow Tracking
    safe_set_experiment("Multi_AutoML_Project")

    if stop_event and stop_event.is_set():
        raise StopIteration("Experiment cancelled before setup.")

    try:
        # 2. PyCaret Setup
        # We let PyCaret use log_experiment=False and handle MLflow ourselves
        # to avoid the "Run already active" conflict.
        logger.info("Step: Setting up PyCaret environment...")

        clf_setup = setup(
            data=train_df,
            target=target_col,
            test_data=val_df,
            session_id=42,
            verbose=False,
            fold=3,
            normalize=True,
            index=False,
            feature_selection=False,
            log_experiment=False,   # disable PyCaret's internal mlflow to avoid conflicts
            system_log=False,
            memory=False,
            n_jobs=1
        )

        if stop_event and stop_event.is_set():
            raise StopIteration("Experiment cancelled after setup.")

        # 3. Start our own MLflow run AFTER PyCaret setup
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            mlflow.log_param("framework", "pycaret")
            mlflow.log_param("model_type", "pycaret")

            # 4. Model Comparison
            logger.info("Step: Comparing models...")
            include_models = ["lr", "nb", "rf", "et", "lightgbm"]
            n_select = 3
            logger.info(f"Including models: {include_models}")

            best_models = compare_models(
                n_select=n_select,
                sort="F1",
                verbose=False,
                include=include_models
            )

            comparison_df = pull()
            if not comparison_df.empty:
                top_model_name = comparison_df.iloc[0]['Model']
                logger.info(f"Best model found: {top_model_name}")

            if stop_event and stop_event.is_set():
                raise StopIteration("Experiment cancelled after model comparison.")

            # Ensure best_models is a list
            if not isinstance(best_models, list):
                best_models = [best_models]

            best_model = best_models[0]

            # 5. Tuning
            logger.info("Step: Tuning best model...")
            n_iter = 10 if time_limit is None or time_limit >= 300 else 5
            tuned_model = tune_model(
                best_model,
                optimize="F1",
                n_iter=n_iter,
                search_library="scikit-learn",
                search_algorithm="random",
                verbose=False,
                choose_better=True
            )

            if stop_event and stop_event.is_set():
                raise StopIteration("Experiment cancelled after tuning.")

            # 6. Blending (only if we have multiple models)
            if len(best_models) > 1:
                logger.info("Step: Blending top models...")
                final_model = blend_models(
                    estimator_list=best_models,
                    optimize="F1",
                    verbose=False
                )
            else:
                final_model = tuned_model
                logger.info("Step: Skipping blend (only one model selected).")

            # 7. Save model
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            model_path_base = os.path.join(model_dir, f"{run_name}_pycaret_model")
            logger.info(f"Saving model to {model_path_base}.pkl...")
            save_model(final_model, model_path_base)

            # 8. Log metrics to our MLflow run
            try:
                final_metrics = pull()
                if not final_metrics.empty:
                    row = final_metrics.iloc[0]
                    for k, v in row.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k.lower().replace(" ", "_"), float(v))
            except Exception as me:
                logger.warning(f"Could not pull metrics: {me}")

            # Log model artifact
            model_pkl = f"{model_path_base}.pkl"
            if os.path.exists(model_pkl):
                mlflow.log_artifact(model_pkl, artifact_path="model")

            logger.info("PyCaret experiment completed successfully.")
            return {
                "success": True,
                "predictor": final_model,
                "run_id": run_id,
                "type": "pycaret",
                "model_path": model_pkl
            }

    except StopIteration as si:
        logger.warning(f"Cancelled: {si}")
        raise
    except Exception as e:
        logger.error(f"PyCaret Error: {e}")
        logger.error(traceback.format_exc())
        raise e
    finally:
        # Always clean up any dangling run
        try:
            mlflow.end_run()
        except Exception:
            pass
