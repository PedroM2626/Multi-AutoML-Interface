import os
import threading
import pandas as pd
import mlflow
import shutil
import logging
from flaml import AutoML
import matplotlib.pyplot as plt
import time
from src.mlflow_utils import safe_set_experiment
from src.onnx_utils import export_to_onnx

logger = logging.getLogger(__name__)

class MultiFLAMLPredictor:
    def __init__(self, predictors_by_target):
        self.predictors_by_target = predictors_by_target
        self.best_loss = sum(getattr(p, 'best_loss', 0.0) for p in predictors_by_target.values()) / len(predictors_by_target)
        self.model = self
        self.estimator = self

    def predict(self, X):
        predictions = {}
        for target_name, predictor in self.predictors_by_target.items():
            predictions[target_name] = predictor.predict(X)
        return pd.DataFrame(predictions, index=X.index)

def train_flaml_model(train_data: pd.DataFrame, target, run_name: str, 
                      valid_data: pd.DataFrame = None, test_data: pd.DataFrame = None,
                       time_budget: int = 60, task: str = 'classification', metric: str = 'auto',
                       estimator_list: list = 'auto', seed: int = 42, cv_folds: int = 0,
                       n_jobs: int = 1,
                       stop_event=None, telemetry_queue=None):
    """
    Trains a FLAML model and logs results to MLflow.
    """
    import json
    safe_set_experiment("FLAML_Experiments")
    logging.info(f"Starting FLAML training for run: {run_name}")
    
    # Ensure flaml logger is also at INFO level
    import flaml
    from flaml import AutoML
    flaml_logger = logging.getLogger('flaml')
    flaml_logger.setLevel(logging.INFO)
    
    # Ensure no leaked runs in this thread
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except:
        pass

    target_columns = target if isinstance(target, list) else [target]
    is_multitarget = len(target_columns) > 1

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # Data cleaning: drop rows where targets are NaN
        train_data = train_data.dropna(subset=target_columns)
        logging.info(f"Data ready: {len(train_data)} rows.")
        
        # Log parameters
        mlflow.log_param("target", json.dumps(target_columns) if is_multitarget else target_columns[0])
        mlflow.log_param("time_budget", time_budget)
        mlflow.log_param("task", task)
        mlflow.log_param("metric", metric)
        mlflow.log_param("estimator_list", str(estimator_list))
        mlflow.log_param("seed", seed)
        
        X_train = train_data.drop(columns=target_columns)
        
        X_val = None
        if valid_data is not None:
            valid_data = valid_data.dropna(subset=target_columns)
            X_val = valid_data.drop(columns=target_columns)
            mlflow.log_param("has_validation_data", True)
            
        if test_data is not None:
             mlflow.log_param("has_test_data", True)
        
        # Train model
        logging.info("Executing hyperparameter search...")
        if is_multitarget:
            predictors_by_target = {}
            per_target_time_budget = max(10, int((time_budget or 60) / len(target_columns))) if time_budget else None
            
            for target_name in target_columns:
                if stop_event and stop_event.is_set():
                    raise StopIteration("Training cancelled by user")
                
                y_tr = train_data[target_name]
                y_v = valid_data[target_name] if valid_data is not None else None
                
                local_settings = {
                    "metric": metric,
                    "task": task,
                    "estimator_list": estimator_list,
                    "log_file_name": f"flaml_{target_name}.log",
                    "seed": seed,
                    "n_jobs": n_jobs,
                    "verbose": 0,
                }
                if per_target_time_budget is not None:
                    local_settings["time_budget"] = per_target_time_budget
                if cv_folds > 0:
                    local_settings["eval_method"] = "cv"
                    local_settings["n_splits"] = cv_folds
                if X_val is not None:
                    local_settings["X_val"] = X_val
                    local_settings["y_val"] = y_v
                
                # Telemetry callback
                if telemetry_queue:
                    def _telemetry_callback(iter_count, time_used, best_loss, best_config, estimator, trial_id, tgt=target_name):
                        try:
                            telemetry_queue.put({
                                "status": "running",
                                "target": tgt,
                                "iterations": iter_count,
                                "time_used": time_used,
                                "best_loss": best_loss,
                                "best_estimator": str(estimator),
                                "best_config_preview": str(best_config)[:200]
                            })
                        except: pass
                    local_settings["callbacks"] = [_telemetry_callback]
                
                automl_single = AutoML()
                
                # Start watcher thread for cancel
                if stop_event is not None:
                    def _watch_single(a=automl_single):
                        stop_event.wait()
                        try: a._state.time_budget = 0
                        except: pass
                    threading.Thread(target=_watch_single, daemon=True).start()
                
                # Temporarily end MLflow run to prevent FLAML from capturing locks
                active_run = mlflow.active_run()
                if active_run:
                    mlflow.end_run()
                try:
                    automl_single.fit(X_train=X_train, y_train=y_tr, **local_settings)
                except StopIteration:
                    if not hasattr(automl_single, 'best_estimator') or automl_single.best_estimator is None:
                        raise RuntimeError(f"FLAML stopped without finding a model for target {target_name}.")
                finally:
                    if active_run:
                        mlflow.start_run(run_id=active_run.info.run_id)
                
                predictors_by_target[target_name] = automl_single
                
            automl = MultiFLAMLPredictor(predictors_by_target)
        else:
            y_train = train_data[target_columns[0]]
            y_val = valid_data[target_columns[0]] if valid_data is not None else None
            
            settings = {
                "metric": metric,
                "task": task,
                "estimator_list": estimator_list,
                "log_file_name": "flaml.log",
                "seed": seed,
                "n_jobs": n_jobs,
                "verbose": 0,
            }
            if time_budget is not None:
                settings["time_budget"] = time_budget
            if cv_folds > 0:
                settings["eval_method"] = "cv"
                settings["n_splits"] = cv_folds
            if X_val is not None:
                settings["X_val"] = X_val
                settings["y_val"] = y_val
                
            if telemetry_queue:
                def _telemetry_callback(iter_count, time_used, best_loss, best_config, estimator, trial_id):
                    try:
                        telemetry_queue.put({
                            "status": "running",
                            "iterations": iter_count,
                            "time_used": time_used,
                            "best_loss": best_loss,
                            "best_estimator": str(estimator),
                            "best_config_preview": str(best_config)[:200]
                        })
                    except: pass
                settings["callbacks"] = [_telemetry_callback]
                
            automl = AutoML()
            if stop_event is not None:
                def _watch():
                    stop_event.wait()
                    try: automl._state.time_budget = 0
                    except: pass
                threading.Thread(target=_watch, daemon=True).start()
                
            # Temporarily end MLflow run to prevent FLAML from capturing locks
            active_run = mlflow.active_run()
            if active_run:
                mlflow.end_run()
            try:
                automl.fit(X_train=X_train, y_train=y_train, **settings)
            except StopIteration:
                if not hasattr(automl, 'best_estimator') or automl.best_estimator is None:
                    raise RuntimeError("FLAML stopped without finding a valid model.")
            finally:
                if active_run:
                    mlflow.start_run(run_id=active_run.info.run_id)
        
        if stop_event and stop_event.is_set():
            raise StopIteration("Training cancelled by user")
        
        # Log metrics
        if hasattr(automl, 'best_loss'):
            mlflow.log_metric("best_loss", automl.best_loss)
            logging.info(f"Best final Loss: {automl.best_loss:.4f}")
        
        # Save best model
        model_path = os.path.join("models", f"flaml_{run_name}.pkl")
        os.makedirs("models", exist_ok=True)
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump(automl, f)
            
        # Log as artifact
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_param("model_type", "flaml")
        
        # ONNX Export
        if not is_multitarget:
            try:
                onnx_path = os.path.join("models", f"flaml_{run_name}.onnx")
                # For FLAML, we can often export the underlying best estimator or the AutoML object if it's scikit-learn compatible
                # We pass X_train[:1] as sample input for shape inference
                export_to_onnx(automl.model.estimator, "flaml", target, onnx_path, input_sample=X_train[:1])
                mlflow.log_artifact(onnx_path, artifact_path="model")
            except Exception as e:
                logger.warning(f"Failed to export FLAML model to ONNX: {e}")
        
        # Generate and log consumption code sample
        try:
            from src.code_gen_utils import generate_consumption_code
            code_sample = generate_consumption_code("flaml", run.info.run_id, target)
            code_path = "consumption_sample.py"
            with open(code_path, "w") as f:
                f.write(code_sample)
            mlflow.log_artifact(code_path)
            if os.path.exists(code_path):
                os.remove(code_path)
        except Exception as e:
            logger.warning(f"Failed to generate consumption code: {e}")
            
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
    raise FileNotFoundError("FLAML model not found in artifacts.")
