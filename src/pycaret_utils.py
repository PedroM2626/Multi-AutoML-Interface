import os
import logging
import traceback
import queue
import time
import pandas as pd
from typing import Dict, Any, Optional

import mlflow

from src.mlflow_utils import safe_set_experiment
from src.onnx_utils import export_to_onnx


def run_pycaret_experiment(
    train_df: pd.DataFrame,
    target_col: str,
    run_name: str,
    time_limit: Optional[int],
    log_queue: queue.Queue,
    stop_event=None,
    val_df: Optional[pd.DataFrame] = None,
    task_type: str = "Classification",
    **kwargs
) -> Dict[str, Any]:
    """
    Run PyCaret experiment.
    Dynamically loads classification, regression, or time_series depending on task_type.
    """
    logger = logging.getLogger("pycaret")
    logger.info(f"Starting PyCaret experiment: {run_name} (Task: {task_type})")
    logger.info(f"Dataset shape: {train_df.shape}, Target: {target_col}")

    # Dynamic imports based on task_type
    if task_type == "Regression":
        from pycaret.regression import setup, compare_models, pull, tune_model, blend_models, save_model
        sort_metric = "R2"
        include_models = ["lr", "rf", "et", "lightgbm"]
    elif task_type == "Time Series Forecasting":
        from pycaret.time_series import setup, compare_models, pull, tune_model, blend_models, save_model
        sort_metric = "MASE"
        include_models = ["naive", "snaive", "arima", "ets"]
    else:
        from pycaret.classification import setup, compare_models, pull, tune_model, blend_models, save_model
        sort_metric = "F1"
        include_models = ["lr", "nb", "rf", "et", "lightgbm"]

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
        logger.info("Step: Setting up PyCaret environment...")
        
        setup_kwargs = {
            "data": train_df,
            "target": target_col,
            "session_id": 42,
            "verbose": False,
            "fold": 3,
            "log_experiment": False,
            "system_log": False,
            "n_jobs": 1
        }
        
        if task_type == "Time Series Forecasting":
            setup_kwargs["fh"] = kwargs.get("fh", 12)
            setup_kwargs["seasonal_period"] = kwargs.get("seasonal_period", 12)
        else:
            setup_kwargs["test_data"] = val_df
            setup_kwargs["normalize"] = True
            setup_kwargs["index"] = False
            setup_kwargs["feature_selection"] = False
            setup_kwargs["memory"] = False

        clf_setup = setup(**setup_kwargs)

        if stop_event and stop_event.is_set():
            raise StopIteration("Experiment cancelled after setup.")

        # 3. Start our own MLflow run AFTER PyCaret setup
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            mlflow.log_param("framework", "pycaret")
            mlflow.log_param("model_type", "pycaret")
            mlflow.log_param("task_type", task_type)

            # 4. Model Comparison
            logger.info("Step: Comparing models...")
            n_select = 3
            logger.info(f"Including models: {include_models} (Sorting by {sort_metric})")

            best_models = compare_models(
                n_select=n_select,
                sort=sort_metric,
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

            # 5. Tuning (Time Series tuning might require different params, keeping generic)
            logger.info("Step: Tuning best model...")
            n_iter = 10 if time_limit is None or time_limit >= 300 else 5
            
            # search_library="scikit-learn" shouldn't be passed to pycaret.time_series
            tune_kwargs = {
                "estimator": best_model,
                "optimize": sort_metric,
                "n_iter": n_iter,
                "verbose": False,
                "choose_better": True
            }
            if task_type != "Time Series Forecasting":
                tune_kwargs["search_library"] = "scikit-learn"
                tune_kwargs["search_algorithm"] = "random"

            tuned_model = tune_model(**tune_kwargs)

            if stop_event and stop_event.is_set():
                raise StopIteration("Experiment cancelled after tuning.")

            # 6. Blending (only if we have multiple models)
            if len(best_models) > 1:
                logger.info("Step: Blending top models...")
                final_model = blend_models(
                    estimator_list=best_models,
                    optimize=sort_metric,
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

            # ONNX Export
            try:
                onnx_path = os.path.join(model_dir, f"{run_name}_pycaret.onnx")
                # PyCaret 'final_model' is a scikit-learn pipeline
                export_to_onnx(final_model, "pycaret", target_col, onnx_path, input_sample=train_df[:1])
                mlflow.log_artifact(onnx_path, artifact_path="model")
            except Exception as e:
                logger.warning(f"Failed to export PyCaret model to ONNX: {e}")

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

