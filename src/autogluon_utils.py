import os
import importlib.metadata
import pandas as pd
import mlflow
import re
import shutil
import logging
import time
import threading
import json
from typing import Dict
from src.mlflow_utils import safe_set_experiment
from src.onnx_utils import export_to_onnx
from src.data_utils import CV_METADATA_COLUMNS, resolve_inside_dir

logger = logging.getLogger(__name__)


def _pandas_lacks_the_downcasting_option() -> bool:
    """True when this interpreter's pandas is older than the key AutoGluon's tabular fit opens.

    `autogluon.tabular.learner.abstract_learner` wraps its preprocessing in
    `pd.option_context("future.no_silent_downcasting", True)`, a registered option only from
    pandas 2.2. PyCaret 3.3.2 pins `pandas<2.2`, and in an interpreter that has both the fit dies
    with a bare `OptionError` after the data has already been processed.
    """
    version = importlib.metadata.version("pandas")
    match = re.match(r"(\d+)\.(\d+)", version)
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) < (2, 2)


class MultiLabelAutoGluonPredictor:
    """Simple multi-target wrapper around one TabularPredictor per target."""

    def __init__(self, predictors_by_target: Dict[str, object]):
        self.predictors_by_target = predictors_by_target

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        predictions = {}
        for target_name, predictor in self.predictors_by_target.items():
            predictions[target_name] = predictor.predict(data)
        return pd.DataFrame(predictions, index=data.index)


def train_model(train_data: pd.DataFrame, target, run_name: str,
                valid_data: pd.DataFrame = None, test_data: pd.DataFrame = None, 
                time_limit: int = 60, presets: str = 'medium_quality', seed: int = 42, cv_folds: int = 0,
                stop_event=None, task_type: str = "Classification", data_category: str = "Tabular",
                multimodal_text_columns=None, multimodal_image_columns=None, telemetry_queue=None):
    """
    Trains an AutoGluon model and logs results to MLflow using generic artifact logging.
    Supports both Tabular data and Computer Vision tasks (via MultiModalPredictor).
    """
    is_cv_task = task_type and task_type.startswith("Computer Vision")
    # Text is the same engine path as Multimodal: MultiModalPredictor with the columns the user
    # marked as text. TabularPredictor would treat a free-text column as one categorical feature.
    is_multimodal_task = data_category in ("Multimodal", "Text")
    is_segmentation = task_type == "Computer Vision - Image Segmentation"
    is_cv_multilabel = task_type == "Computer Vision - Multi-Label Classification"
    is_tabular_multilabel = data_category == "Tabular" and task_type in ["Multi-Label Classification", "Multi-Task Classification"]
    target_columns = target if isinstance(target, list) else [target]

    if is_tabular_multilabel:
        if len(target_columns) < 2:
            raise ValueError("Tabular Multi-Label Classification requires at least two target columns.")
        for column in target_columns:
            if column not in train_data.columns:
                raise ValueError(f"Target column '{column}' not found in training data.")

    if is_cv_multilabel:
        if len(target_columns) < 2:
            raise ValueError(
                "Computer Vision Multi-Label Classification needs at least two label columns. "
                "Upload the images together with an annotations CSV that has an 'image' column "
                "and one 0/1 column per label."
            )
        for column in target_columns:
            if column not in train_data.columns:
                raise ValueError(f"Label column '{column}' is not in the annotation table.")
    
    if not (is_cv_task or is_multimodal_task) and _pandas_lacks_the_downcasting_option():
        raise ValueError(
            "AutoGluon's tabular predictor cannot run in this interpreter: its fit opens "
            f"pd.option_context('future.no_silent_downcasting'), a key pandas "
            f"{importlib.metadata.version('pandas')} does not define (it arrived in 2.2). "
            "Pick another engine for this run, or use an environment with pandas >= 2.2 - "
            "requirements.txt has it, the all-engine lock cannot (PyCaret pins pandas<2.2)."
        )

    if is_cv_task:
        from autogluon.multimodal import MultiModalPredictor
        
        def build_image_df(path_df):
            if path_df is None or "Image_Directory" not in path_df.columns:
                return path_df
            img_dir = path_df.iloc[0]["Image_Directory"]
            if "image" in path_df.columns:
                # Annotated dataset: the upload stored a table of image names plus one column
                # per label, which is the only shape that can carry multi-label image targets.
                prepared = path_df.drop(columns=[c for c in CV_METADATA_COLUMNS if c in path_df.columns])
                prepared = prepared.copy()
                prepared["image"] = [
                    value if os.path.isabs(value) else os.path.normpath(os.path.join(img_dir, value))
                    for value in prepared["image"].astype(str)
                ]
                return prepared
            data = []
            for root, _, files in os.walk(img_dir):
                label = os.path.basename(root)
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        data.append({"image": os.path.join(root, file), target: label})
            return pd.DataFrame(data)

        train_data = build_image_df(train_data)
        valid_data = build_image_df(valid_data)
        test_data = build_image_df(test_data)
    elif is_multimodal_task:
        from autogluon.multimodal import MultiModalPredictor

        def _prepare_multimodal_df(path_df):
            if path_df is None:
                return path_df
            if target not in path_df.columns:
                return path_df
            prepared_df = path_df.dropna(subset=[target]).copy()
            for column in (multimodal_text_columns or []):
                if column in prepared_df.columns:
                    prepared_df[column] = prepared_df[column].fillna("").astype(str)
            for column in (multimodal_image_columns or []):
                if column in prepared_df.columns:
                    prepared_df[column] = prepared_df[column].fillna("").astype(str)
            return prepared_df

        train_data = _prepare_multimodal_df(train_data)
        valid_data = _prepare_multimodal_df(valid_data)
        test_data = _prepare_multimodal_df(test_data)
    else:
        from autogluon.tabular import TabularPredictor
    
    safe_set_experiment("AutoGluon_Experiments")
    
    # Ensure no leaked runs in this thread
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # Data cleaning: drop rows where target is NaN
        train_data = train_data.dropna(subset=target_columns)
        
        # Log parameters
        mlflow.log_param("target", json.dumps(target_columns) if len(target_columns) > 1 else target_columns[0])
        mlflow.log_param("time_limit", time_limit)
        mlflow.log_param("presets", presets)
        mlflow.log_param("seed", seed)
        mlflow.log_param("data_category", data_category)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("is_tabular_multilabel", str(is_tabular_multilabel).lower())
        mlflow.log_param("is_cv_multilabel", str(is_cv_multilabel).lower())
        if is_tabular_multilabel or is_cv_multilabel:
            mlflow.log_param("multilabel_targets", json.dumps(target_columns))
        if is_multimodal_task:
            mlflow.log_param("multimodal_text_columns", str(multimodal_text_columns or []))
            mlflow.log_param("multimodal_image_columns", str(multimodal_image_columns or []))
        
        # Output directory for AutoGluon
        model_path = resolve_inside_dir(os.path.join("models", run_name), "models")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            
        # Clean validation and test formats if present
        if valid_data is not None:
            for tgt_col in target_columns:
                if tgt_col not in valid_data.columns:
                    raise ValueError(f"Target column '{tgt_col}' not found in Validation data. Make sure it has the same structure as the training dataset.")
            valid_data = valid_data.dropna(subset=target_columns)
            mlflow.log_param("has_validation_data", True)
        if test_data is not None:
            for tgt_col in target_columns:
                if tgt_col not in test_data.columns:
                    raise ValueError(f"Target column '{tgt_col}' not found in Test data. Make sure the test set includes the target variable.")
            test_data = test_data.dropna(subset=target_columns)
            mlflow.log_param("has_test_data", True)
            
        if (is_cv_task or is_multimodal_task) and not is_cv_multilabel:
            mm_fit_args = {"train_data": train_data, "time_limit": time_limit}
            if valid_data is not None:
                mm_fit_args["tuning_data"] = valid_data
            
            problem_type = None
            if is_segmentation:
                problem_type = "semantic_segmentation"
            elif task_type == "Computer Vision - Object Detection":
                problem_type = "object_detection"
            elif is_multimodal_task:
                problem_type = "regression" if task_type == "Regression" else "classification"
                
            mm_presets = "high_quality" if presets in ["best_quality", "high_quality"] else "medium_quality"
            predictor = MultiModalPredictor(label=target, problem_type=problem_type, path=model_path).fit(**mm_fit_args, presets=mm_presets)
        elif is_tabular_multilabel:
            predictors_by_target = {}
            label_scores = {}
            per_label_time_limit = max(30, int((time_limit or 300) / max(1, len(target_columns)))) if time_limit else None

            for target_name in target_columns:
                if stop_event and stop_event.is_set():
                    raise StopIteration("Training cancelled by user")

                target_model_path = os.path.join(model_path, target_name)
                drop_targets = [col for col in target_columns if col != target_name]
                train_subset = train_data.drop(columns=drop_targets, errors="ignore")
                valid_subset = valid_data.drop(columns=drop_targets, errors="ignore") if valid_data is not None else None
                test_subset = test_data.drop(columns=drop_targets, errors="ignore") if test_data is not None else None

                fit_args = {
                    "train_data": train_subset,
                    "time_limit": per_label_time_limit,
                    "presets": presets,
                }
                if cv_folds > 0:
                    fit_args["num_bag_folds"] = cv_folds
                if valid_subset is not None:
                    fit_args["tuning_data"] = valid_subset
                    if cv_folds > 0 or presets in ["best_quality", "high_quality"]:
                        fit_args["use_bag_holdout"] = True

                predictor_single = TabularPredictor(label=target_name, path=target_model_path).fit(**fit_args)
                predictors_by_target[target_name] = predictor_single

                eval_subset = test_subset if test_subset is not None else (valid_subset if valid_subset is not None else train_subset)
                lb_single = predictor_single.leaderboard(eval_subset, silent=True)
                if not lb_single.empty:
                    label_scores[target_name] = float(lb_single.iloc[0]["score_val"])

            predictor = MultiLabelAutoGluonPredictor(predictors_by_target)
            if label_scores:
                mlflow.log_metric("best_model_score_avg", float(sum(label_scores.values()) / len(label_scores)))
                for label_name, label_score in label_scores.items():
                    safe_metric_name = label_name.replace(" ", "_").replace("-", "_").lower()
                    mlflow.log_metric(f"best_model_score_{safe_metric_name}", label_score)

            leaderboard_path = "leaderboard.csv"
            leaderboard_df = pd.DataFrame(
                [{"target": lbl, "best_score": score} for lbl, score in label_scores.items()]
            )
            leaderboard_df.to_csv(leaderboard_path, index=False)
        elif is_cv_multilabel:
            # AutoGluon's MultiModalPredictor has no multilabel problem type - asking for one
            # asserts inside fit() and lists classification, binary, multiclass, regression,
            # object_detection, semantic_segmentation, the similarity and the NER types. The row is
            # therefore trained like the tabular multi-label one: one predictor per label column.
            mm_presets = "high_quality" if presets in ["best_quality", "high_quality"] else "medium_quality"
            per_label_time_limit = (
                max(30, int((time_limit or 300) / max(1, len(target_columns)))) if time_limit else None
            )

            predictors_by_target = {}
            label_metrics = {}

            for target_name in target_columns:
                if stop_event and stop_event.is_set():
                    raise StopIteration("Training cancelled by user")

                drop_targets = [col for col in target_columns if col != target_name]
                train_subset = train_data.drop(columns=drop_targets, errors="ignore")
                valid_subset = valid_data.drop(columns=drop_targets, errors="ignore") if valid_data is not None else None
                test_subset = test_data.drop(columns=drop_targets, errors="ignore") if test_data is not None else None

                fit_args = {"train_data": train_subset, "time_limit": per_label_time_limit, "presets": mm_presets}
                if valid_subset is not None:
                    fit_args["tuning_data"] = valid_subset

                predictor_single = MultiModalPredictor(
                    label=target_name,
                    problem_type="classification",
                    path=os.path.join(model_path, target_name),
                ).fit(**fit_args)
                predictors_by_target[target_name] = predictor_single

                eval_subset = (
                    test_subset if test_subset is not None
                    else (valid_subset if valid_subset is not None else train_subset)
                )
                scores = predictor_single.evaluate(eval_subset)
                if not isinstance(scores, dict):
                    scores = {"score": scores}
                numeric = {}
                for key, value in scores.items():
                    try:
                        numeric[str(key)] = float(value)
                    except (TypeError, ValueError):
                        logger.info("Skipping non-numeric evaluation entry %s=%r", key, value)
                label_metrics[target_name] = numeric

            predictor = MultiLabelAutoGluonPredictor(predictors_by_target)
            for target_name, numeric in label_metrics.items():
                safe_label = target_name.replace(" ", "_").replace("-", "_").lower()
                for key, value in numeric.items():
                    mlflow.log_metric(f"{safe_label}_{key}", value)

            leaderboard_path = "leaderboard.csv"
            rows = [{"label": name, **metrics} for name, metrics in label_metrics.items()]
            pd.DataFrame(rows or [{"label": None}]).to_csv(leaderboard_path, index=False)
        else:
            fit_args = {
                "train_data": train_data,
                "time_limit": time_limit, 
                "presets": presets
            }
            if cv_folds > 0:
                fit_args["num_bag_folds"] = cv_folds
                
            if valid_data is not None:
                fit_args["tuning_data"] = valid_data
                # If bagging is enabled (manually or by presets), we must set use_bag_holdout=True to use separate tuning_data
                if cv_folds > 0 or presets in ["best_quality", "high_quality"]:
                    fit_args["use_bag_holdout"] = True
                

            # Streaming updates thread
            _ag_training_done = threading.Event()
            def _push_ag_telemetry():
                while not _ag_training_done.is_set() and not (stop_event and stop_event.is_set()):
                    try:
                        if os.path.exists(model_path):
                            # AutoGluon sometimes locks the file, so we try-except
                            from autogluon.tabular import TabularPredictor
                            try:
                                temp_predictor = TabularPredictor.load(path=model_path)
                                lb = temp_predictor.leaderboard(silent=True)
                                if len(lb) > 0:
                                    best_model = lb.iloc[0]['model']
                                    best_score = lb.iloc[0]['score_val']
                                    if telemetry_queue:
                                        telemetry_queue.put({
                                            "status": "running",
                                            "models_trained": len(lb),
                                            "best_model": best_model,
                                            "best_value": best_score,
                                            "leaderboard_preview": lb.head(5).to_dict(orient='records')
                                        })
                            except Exception:
                                pass
                    except Exception:
                        pass
                    time.sleep(10)
            
            if telemetry_queue:
                t_telemetry = threading.Thread(target=_push_ag_telemetry, daemon=True)
                t_telemetry.start()

            predictor = TabularPredictor(label=target, path=model_path).fit(**fit_args)
            if telemetry_queue:
                _ag_training_done.set()
        
        # Check if cancelled before continuing
        if stop_event and stop_event.is_set():
            raise StopIteration("Training cancelled by user")
        
        eval_data = test_data if test_data is not None else (valid_data if valid_data is not None else train_data)

        if (is_cv_task or is_multimodal_task) and not is_cv_multilabel:
            # MultiModalPredictor trains a single model and exposes evaluate(), not
            # leaderboard(); asking for the leaderboard raised AttributeError after the training
            # had already finished, which threw away the run's whole result.
            scores = predictor.evaluate(eval_data)
            if not isinstance(scores, dict):
                scores = {"score": scores}
            numeric_scores = {}
            for key, value in scores.items():
                try:
                    numeric_scores[str(key)] = float(value)
                except (TypeError, ValueError):
                    logger.info("Skipping non-numeric evaluation entry %s=%r", key, value)
            if numeric_scores:
                mlflow.log_metrics(numeric_scores)
            leaderboard_path = "leaderboard.csv"
            pd.DataFrame([numeric_scores or {"score": None}]).to_csv(leaderboard_path, index=False)
        elif not (is_tabular_multilabel or is_cv_multilabel):
            leaderboard = predictor.leaderboard(eval_data, silent=True)
            # Log the best model's score
            best_model_score = leaderboard.iloc[0]['score_val']
            mlflow.log_metric("best_model_score", best_model_score)
            leaderboard_path = "leaderboard.csv"
            leaderboard.to_csv(leaderboard_path, index=False)
        try:
            mlflow.log_artifact(leaderboard_path)
        except Exception as e:
            logger.warning(f"Failed to log leaderboard artifact: {e}")
        finally:
            if os.path.exists(leaderboard_path):
                os.remove(leaderboard_path)
        
        # Log AutoGluon model directory as a generic artifact
        # We use a try-except here because disk space issues frequently occur during artifact copy
        model_folder_removed = False
        try:
            mlflow.log_artifacts(model_path, artifact_path="model")
            mlflow.log_param("model_type", "autogluon")
            
            # ONNX Export (Best effort for Tabular)
            if not is_cv_task and not is_multimodal_task and not is_tabular_multilabel:
                try:
                    onnx_path = os.path.join("models", f"ag_{run_name}.onnx")
                    # AutoGluon Tabular supports ONNX export for some models
                    # This might require specific dependencies or AG version
                    # We call our utility which handles AG logic
                    export_to_onnx(predictor, "autogluon", target, onnx_path, input_sample=train_data[:1])
                    mlflow.log_artifact(onnx_path, artifact_path="model")
                except Exception as e:
                    logger.warning(f"Failed to export AutoGluon model to ONNX: {e}")

            logger.info(f"AutoGluon artifacts logged successfully for {run_name}")
            
            # CRITICAL: Delete local model folder after successful MLflow logging to save disk space
            # Only do this if it was logged successfully to the tracking server/local mlruns
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                model_folder_removed = True
                logger.info(f"Cleaned up local model folder: {model_path}")
        except Exception as e:
            logger.error(f"Failed to log model artifacts to MLflow (likely disk space): {e}")
            # Do NOT delete model_path here so the user can potentially recover it manually
            # if the MLflow log failed.

        if model_folder_removed:
            # The predictor that was just fitted reads its estimators from
            # models/<run>/models/*/model.pkl, which the cleanup above removed, so returning it
            # handed the app an object that failed on the first prediction after a successful
            # training. The artifact copy is the model's home now: reload from it.
            try:
                predictor = load_model_from_mlflow(run.info.run_id)
                logger.info(f"Reloaded {run_name} from its MLflow artifact for in-session use.")
            except Exception as e:
                logger.warning(
                    f"Could not reload {run_name} from MLflow ({e}); the predictor in this session "
                    f"points at the deleted folder {model_path}."
                )
        
        # Generate and log consumption code sample
        try:
            from src.code_gen_utils import generate_consumption_code
            code_sample = generate_consumption_code("autogluon", run.info.run_id, target)
            code_path = "consumption_sample.py"
            with open(code_path, "w") as f:
                f.write(code_sample)
            mlflow.log_artifact(code_path)
            if os.path.exists(code_path):
                os.remove(code_path)
        except Exception as e:
            logger.warning(f"Failed to generate consumption code: {e}")
        
        return predictor, run.info.run_id

def load_model_from_mlflow(run_id: str):
    """
    Loads a model from MLflow artifacts.
    """
    import mlflow
    is_tabular_multilabel = False
    is_cv_multilabel = False
    multilabel_targets_raw = "[]"
    try:
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        data_category = run.data.params.get("data_category", "Tabular")
        task_type = run.data.params.get("task_type", "Classification")
        is_tabular_multilabel = run.data.params.get("is_tabular_multilabel", "false") == "true"
        is_cv_multilabel = run.data.params.get("is_cv_multilabel", "false") == "true"
        multilabel_targets_raw = run.data.params.get("multilabel_targets", "[]")
    except Exception:
        data_category = "Tabular"
        task_type = "Classification"

    # Download the artifact folder
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")

    try:
        target_columns = json.loads(multilabel_targets_raw)
    except Exception:
        target_columns = []

    # Load the predictor from the local path
    if is_cv_multilabel:
        from autogluon.multimodal import MultiModalPredictor

        predictors_by_target = {}
        for target_name in target_columns:
            target_model_dir = os.path.join(local_path, target_name)
            if os.path.isdir(target_model_dir):
                predictors_by_target[target_name] = MultiModalPredictor.load(target_model_dir)

        if not predictors_by_target:
            raise FileNotFoundError("No per-label multimodal predictors found for this CV multi-label run.")
        predictor = MultiLabelAutoGluonPredictor(predictors_by_target)
    elif data_category == "Multimodal" or task_type.startswith("Computer Vision"):
        from autogluon.multimodal import MultiModalPredictor

        predictor = MultiModalPredictor.load(local_path)
    elif is_tabular_multilabel:
        from autogluon.tabular import TabularPredictor

        predictors_by_target = {}
        for target_name in target_columns:
            target_model_dir = os.path.join(local_path, target_name)
            if os.path.isdir(target_model_dir):
                predictors_by_target[target_name] = TabularPredictor.load(target_model_dir)

        if not predictors_by_target:
            raise FileNotFoundError("No target-specific predictors found for tabular multi-label AutoGluon run.")
        predictor = MultiLabelAutoGluonPredictor(predictors_by_target)
    else:
        from autogluon.tabular import TabularPredictor

        predictor = TabularPredictor.load(local_path)
    return predictor

def get_leaderboard(predictor):
    return predictor.leaderboard(silent=True)
