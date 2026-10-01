import os
import importlib.util
import threading
import pandas as pd
import mlflow
import shutil
import logging
from flaml import AutoML
import time
from src.mlflow_utils import safe_set_experiment
from src.onnx_utils import export_to_onnx

logger = logging.getLogger(__name__)

# FLAML resolves these learner names through optional packages. When one is absent it
# ends up calling None deep inside cross-validation ("TypeError: 'NoneType' object is
# not callable"), so the missing package is reported before the search starts.
_LEARNER_PACKAGES = {"lgbm": "lightgbm", "catboost": "catboost", "xgboost": "xgboost"}

# FLAML forwards the "callbacks" setting to the estimator's own fit(). Only the
# LightGBM-family learners accept it; sklearn forests raise
# "BaseForest.fit() got an unexpected keyword argument 'callbacks'", and "auto" mixes
# both kinds, so live telemetry is limited to searches that use boosting learners only.
_CALLBACK_LEARNERS = ("lgbm", "xgboost", "catboost")

# flaml.tune keeps its trial runner in a module global (_runner in flaml/tune/tune.py), so two
# searches in one process overwrite each other's runner and the first dies with
# "'NoneType' object has no attribute 'stop_trial'" - routine here, because one server process
# serves several sessions. Same treatment as PyCaret's experiment lock below.
_EXPERIMENT_LOCK = threading.Lock()
_LOCK_POLL_SECONDS = 5


def _apply_evaluation_settings(settings, cv_folds, X_val, y_val):
    """Give FLAML an evaluation scheme it accepts.

    A custom validation frame and eval_method="cv" cannot be combined: fit() raises
    "AssertionError: eval_method must be 'auto' or 'holdout' for custom validation data", and the
    UI's split section hands over a validation frame whenever Simple Holdout is on.
    """
    if X_val is not None:
        settings["eval_method"] = "holdout"
        settings["X_val"] = X_val
        settings["y_val"] = y_val
        settings.pop("n_splits", None)
    elif cv_folds > 0:
        settings["eval_method"] = "cv"
        settings["n_splits"] = cv_folds
    return settings


def _supports_callbacks(estimator_list) -> bool:
    if not isinstance(estimator_list, (list, tuple)) or not estimator_list:
        return False
    return all(
        any(learner in str(name) for learner in _CALLBACK_LEARNERS)
        for name in estimator_list
    )


def _require_learner_packages(estimator_list):
    if not isinstance(estimator_list, (list, tuple)):
        return
    missing = sorted({
        package
        for name in estimator_list
        for learner, package in _LEARNER_PACKAGES.items()
        if learner in str(name) and importlib.util.find_spec(package) is None
    })
    if missing:
        raise ImportError(
            f"FLAML needs {' and '.join(missing)} for the selected estimators. "
            f"Install it with: pip install {' '.join(missing)}"
        )

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

def _forecast_layout(train_data: pd.DataFrame, valid_data, target_columns, time_col, period):
    """Order the frame for FLAML's native time series task and return its extra fit settings.

    FLAML asserts that a forecast task has an integer 'period' and reads the timestamps from
    'time_col'; without both it raises before the search starts.
    """
    if not time_col:
        raise ValueError("FLAML time series forecasting needs the date column (time_col).")
    if time_col not in train_data.columns:
        raise ValueError(f"Date column '{time_col}' is not in the training data.")
    horizon = int(period) if period else 0
    if horizon < 1:
        raise ValueError("FLAML time series forecasting needs a forecast horizon of at least 1.")

    train_sorted = train_data.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    valid_sorted = None
    if valid_data is not None:
        valid_sorted = valid_data.dropna(subset=target_columns)
        valid_sorted = valid_sorted.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    return train_sorted, valid_sorted, {"time_col": time_col, "period": horizon}


def _ranking_layout(train_data: pd.DataFrame, valid_data, target_column, group_col):
    """Prepare the frame for FLAML's learning-to-rank task.

    The ranker behind it reads queries as consecutive row blocks and wants integer relevance
    grades, so rows are sorted by the query column and the target is cast here; an unsorted
    frame or a float target fails inside LightGBM instead of producing a model.
    """
    if not group_col:
        raise ValueError("FLAML ranking needs the query/group column (group_col).")
    if group_col not in train_data.columns:
        raise ValueError(f"Query/group column '{group_col}' is not in the training data.")

    def _cast(target_frame: pd.DataFrame) -> pd.DataFrame:
        grades = pd.to_numeric(target_frame[target_column], errors="coerce")
        if grades.isna().any():
            raise ValueError(
                "FLAML ranking needs integer relevance grades in the target column; "
                f"some rows of '{target_column}' are not numeric."
            )
        rounded = grades.round().astype("int64")
        if not (grades == rounded).all():
            raise ValueError("FLAML ranking needs integer relevance grades in the target column.")
        prepared = target_frame.copy()
        prepared[target_column] = rounded
        return prepared

    def _group_sizes(frame: pd.DataFrame) -> list:
        labels = frame[group_col]
        if labels.isna().any():
            raise ValueError("The query/group column must have a value in every training row.")
        return frame.groupby(labels, sort=False).size().tolist()

    train_sorted = _cast(train_data.sort_values(group_col, kind="mergesort").reset_index(drop=True))
    extra = {"groups": _group_sizes(train_sorted)}

    if valid_data is not None:
        valid_sorted = _cast(
            valid_data.dropna(subset=[target_column, group_col])
            .sort_values(group_col, kind="mergesort")
            .reset_index(drop=True)
        )
        extra["groups_val"] = _group_sizes(valid_sorted)
        return train_sorted, valid_sorted, extra

    return train_sorted, None, extra


def train_flaml_model(train_data: pd.DataFrame, target, run_name: str,
                      valid_data: pd.DataFrame = None, test_data: pd.DataFrame = None,
                      time_budget: int = 60, task: str = 'classification', metric: str = 'auto',
                      estimator_list: list = 'auto', seed: int = 42, cv_folds: int = 0,
                      n_jobs: int = 1,
                      time_col: str = None, period: int = None, group_col: str = None,
                      stop_event=None, telemetry_queue=None):
    """Wait for the search slot, then train; see _EXPERIMENT_LOCK. A queued run can still be
    cancelled, which is why the acquire is polled instead of blocking."""
    while not _EXPERIMENT_LOCK.acquire(timeout=_LOCK_POLL_SECONDS):
        if stop_event is not None and stop_event.is_set():
            raise StopIteration("Experiment cancelled while waiting for the FLAML slot.")
    try:
        return _train_flaml_model(
            train_data=train_data, target=target, run_name=run_name, valid_data=valid_data,
            test_data=test_data, time_budget=time_budget, task=task, metric=metric,
            estimator_list=estimator_list, seed=seed, cv_folds=cv_folds, n_jobs=n_jobs,
            time_col=time_col, period=period, group_col=group_col, stop_event=stop_event,
            telemetry_queue=telemetry_queue,
        )
    finally:
        _EXPERIMENT_LOCK.release()


def _train_flaml_model(train_data: pd.DataFrame, target, run_name: str, 
                      valid_data: pd.DataFrame = None, test_data: pd.DataFrame = None,
                       time_budget: int = 60, task: str = 'classification', metric: str = 'auto',
                       estimator_list: list = 'auto', seed: int = 42, cv_folds: int = 0,
                       n_jobs: int = 1,
                       time_col: str = None, period: int = None, group_col: str = None,
                       stop_event=None, telemetry_queue=None):
    """
    Trains a FLAML model and logs results to MLflow.
    """
    import json
    safe_set_experiment("FLAML_Experiments")
    _require_learner_packages(estimator_list)
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
    except Exception:
        pass

    target_columns = target if isinstance(target, list) else [target]
    is_multitarget = len(target_columns) > 1

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # Data cleaning: drop rows where targets are NaN
        train_data = train_data.dropna(subset=target_columns)
        logging.info(f"Data ready: {len(train_data)} rows.")

        task_settings = {}
        if task in ('ts_forecast', 'rank'):
            if is_multitarget:
                raise ValueError(f"FLAML task '{task}' trains one target column at a time.")
            if task == 'ts_forecast':
                train_data, valid_data, task_settings = _forecast_layout(
                    train_data, valid_data, target_columns, time_col, period
                )
            else:
                train_data, valid_data, task_settings = _ranking_layout(
                    train_data, valid_data, target_columns[0], group_col
                )

        # Log parameters
        mlflow.log_param("target", json.dumps(target_columns) if is_multitarget else target_columns[0])
        mlflow.log_param("time_budget", time_budget)
        mlflow.log_param("task", task)
        mlflow.log_param("metric", metric)
        mlflow.log_param("estimator_list", str(estimator_list))
        mlflow.log_param("seed", seed)
        for key, value in task_settings.items():
            mlflow.log_param(key if key != "groups" else "train_groups", str(value))
        
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
                _apply_evaluation_settings(local_settings, cv_folds, X_val, y_v)
                
                # Telemetry callback
                if telemetry_queue and _supports_callbacks(estimator_list):
                    def _telemetry_callback(callback_env, tgt=target_name):
                        # FLAML hands these to LightGBM, which calls each callback with a
                        # single CallbackEnv; a wider signature raises TypeError at the
                        # call site, outside this function's own try/except.
                        try:
                            results = getattr(callback_env, "evaluation_result_list", None) or []
                            best_loss = results[0][2] if results and len(results[0]) > 2 else None
                            telemetry_queue.put({
                                "status": "running",
                                "target": tgt,
                                "iterations": getattr(callback_env, "iteration", 0),
                                "best_loss": best_loss
                            })
                        except Exception: pass
                    local_settings["callbacks"] = [_telemetry_callback]
                
                automl_single = AutoML()
                
                # Start watcher thread for cancel
                _training_done_single = threading.Event()
                if stop_event is not None:
                    def _watch_single(a=automl_single, done=_training_done_single):
                        while not done.is_set():
                            if stop_event.wait(timeout=5):
                                try: a._state.time_budget = 0
                                except Exception: pass
                                break
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
                    # Releases the cancellation watcher even when fit() raises.
                    _training_done_single.set()
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
                "log_file_name": f"flaml_{run_name}.log",
                "seed": seed,
                "n_jobs": n_jobs,
                "verbose": 0,
            }
            if time_budget is not None:
                settings["time_budget"] = time_budget
            settings.update(task_settings)
            _apply_evaluation_settings(settings, cv_folds, X_val, y_val)
                
            if telemetry_queue and _supports_callbacks(estimator_list):
                def _telemetry_callback(callback_env):
                    # One CallbackEnv argument, as LightGBM invokes it; see the note in the
                    # multi-target branch above.
                    try:
                        results = getattr(callback_env, "evaluation_result_list", None) or []
                        best_loss = results[0][2] if results and len(results[0]) > 2 else None
                        telemetry_queue.put({
                            "status": "running",
                            "iterations": getattr(callback_env, "iteration", 0),
                            "best_loss": best_loss
                        })
                    except Exception: pass
                settings["callbacks"] = [_telemetry_callback]
                
            automl = AutoML()
            _training_done = threading.Event()
            if stop_event is not None:
                def _watch(done=_training_done):
                    while not done.is_set():
                        if stop_event.wait(timeout=5):
                            try: automl._state.time_budget = 0
                            except Exception: pass
                            break
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
                # Releases the cancellation watcher even when fit() raises.
                _training_done.set()
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
                    logger.warning("Loading FLAML model via pickle. Ensure the model artifact is from a trusted source (CWE-502).")
                    return pickle.load(f)
    raise FileNotFoundError("FLAML model not found in artifacts.")
