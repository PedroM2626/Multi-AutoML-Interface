import os
import logging
import traceback
import queue
import time
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Optional

import mlflow

# Lale core imports
import lale
from lale.lib.lale import Hyperopt
from lale.lib.sklearn import LogisticRegression, RandomForestClassifier
from lale.lib.sklearn import MinMaxScaler, PCA
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from src.mlflow_utils import safe_set_experiment


def _preprocess_for_lale(X: pd.DataFrame, y: pd.Series):
    """
    Encode non-numeric features so that sklearn estimators can handle them.
    Returns (X_encoded, y_encoded, encoders) where encoders can be used for inverse transforms.
    """
    X = X.copy()

    # Encode categorical / object columns
    col_encoders = {}
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == 'category':
            le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X[col] = le.fit_transform(X[[col]]).ravel()
            col_encoders[col] = le

    # Fill any remaining NaNs
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else 0)

    # Encode target if string
    y_encoder = None
    if y.dtype == object or str(y.dtype) == 'category':
        y_encoder = LabelEncoder()
        y = pd.Series(y_encoder.fit_transform(y), name=y.name)

    return X, y, col_encoders, y_encoder


def run_lale_experiment(
    train_df: pd.DataFrame,
    target_col: str,
    run_name: str,
    time_limit: Optional[int],
    log_queue: queue.Queue,
    stop_event=None,
    val_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Run Lale experiment using scikit-learn compatible classification via Hyperopt.
    Handles text/categorical features with automatic encoding.
    """
    logger = logging.getLogger("lale")
    logger.info(f"Starting Lale experiment: {run_name}")
    logger.info(f"Dataset shape: {train_df.shape}, Target: {target_col}")

    # Drop NaNs on target
    train_df_c = train_df.dropna(subset=[target_col])
    X_raw = train_df_c.drop(columns=[target_col])
    y_raw = train_df_c[target_col]

    # Pre-process: encode categoricals/text for sklearn compatibility
    logger.info("Step: Encoding categorical/text features...")
    X, y, col_encoders, y_encoder = _preprocess_for_lale(X_raw, y_raw)
    logger.info(f"Features after encoding: {list(X.columns)} | Classes: {y.unique()[:5].tolist()}")

    # Validate MLflow tracking
    safe_set_experiment("Multi_AutoML_Project")

    # Always end any dangling run (Hyperopt can leave runs open)
    try:
        mlflow.end_run()
    except Exception:
        pass

    if stop_event and stop_event.is_set():
        raise StopIteration("Experiment cancelled before setup.")

    try:
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            mlflow.log_param("model_type", "lale")
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", X.shape[0])

            # 1. Pipeline Definition (only numeric-friendly preprocessors)
            logger.info("Step: Defining Lale Planned Pipeline...")
            planned_pipeline = (
                (MinMaxScaler | PCA) >>
                (LogisticRegression | RandomForestClassifier)
            )

            if stop_event and stop_event.is_set():
                raise StopIteration("Experiment cancelled before Hyperopt setup.")

            # 2. Hyperparameter Tuning
            logger.info("Step: Tuning with Hyperopt...")
            max_evals = 10 if time_limit is None or time_limit >= 300 else 5
            time_args = {}
            if time_limit and time_limit > 0:
                time_args['max_eval_time'] = time_limit

            optimizer = Hyperopt(
                estimator=planned_pipeline,
                max_evals=max_evals,
                cv=3,
                scoring="accuracy",
                show_progressbar=False,
                verbose=True,   # show per-trial info so we can debug failures
                **time_args
            )

            # 3. Fit Model
            logger.info(f"Step: Fitting Lale Optimizer (evals={max_evals})...")
            start_time = time.time()
            trained_optimizer = optimizer.fit(X.values, y.values)

            if stop_event and stop_event.is_set():
                raise StopIteration("Experiment cancelled after fitting.")

            best_model = trained_optimizer.get_pipeline()

            # Extract score
            try:
                summary = trained_optimizer.summary()
                best_score = -summary.iloc[0]['loss'] if 'loss' in summary.columns else 0.0
            except Exception:
                best_score = 0.0

            elapsed_time = time.time() - start_time
            logger.info(f"Best Score (CV accuracy): {best_score:.4f}")
            logger.info(f"Optimization time: {elapsed_time:.1f}s")

            # 4. Save Model
            logger.info("Step: Saving model locally...")
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{run_name}_lale_model.pkl")
            joblib.dump({"model": best_model, "col_encoders": col_encoders, "y_encoder": y_encoder}, model_path)

            # Log metrics
            mlflow.log_metric("best_cv_accuracy", best_score)
            mlflow.log_metric("optimization_time", elapsed_time)
            mlflow.log_param("max_evals", max_evals)
            mlflow.log_artifact(model_path, artifact_path="model")

            logger.info("Lale experiment completed successfully.")

            return {
                "success": True,
                "predictor": best_model,
                "run_id": run_id,
                "type": "lale",
                "model_path": model_path,
                "metrics": {"best_cv_accuracy": best_score}
            }

    except StopIteration as si:
        logger.warning(f"Cancelled: {si}")
        raise
    except Exception as e:
        logger.error(f"Lale Error: {e}")
        logger.error(traceback.format_exc())
        raise e
