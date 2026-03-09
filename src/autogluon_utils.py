import os
import pandas as pd
import mlflow
import shutil
import logging
from src.mlflow_utils import safe_set_experiment
from src.onnx_utils import export_to_onnx

logger = logging.getLogger(__name__)

def train_model(train_data: pd.DataFrame, target: str, run_name: str, 
                valid_data: pd.DataFrame = None, test_data: pd.DataFrame = None, 
                time_limit: int = 60, presets: str = 'medium_quality', seed: int = 42, cv_folds: int = 0,
                stop_event=None, task_type: str = "Classification"):
    """
    Trains an AutoGluon model and logs results to MLflow using generic artifact logging.
    Supports both Tabular data and Computer Vision tasks (via MultiModalPredictor).
    """
    is_cv_task = task_type and task_type.startswith("Computer Vision")
    is_segmentation = task_type == "Computer Vision - Image Segmentation"
    is_multilabel = task_type == "Computer Vision - Multi-Label Classification"
    
    if is_cv_task:
        from autogluon.multimodal import MultiModalPredictor
        
        def build_image_df(path_df):
            if path_df is None or "Image_Directory" not in path_df.columns:
                return path_df
            img_dir = path_df.iloc[0]["Image_Directory"]
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
    else:
        from autogluon.tabular import TabularPredictor
    
    safe_set_experiment("AutoGluon_Experiments")
    
    # Ensure no leaked runs in this thread
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except:
        pass

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # Data cleaning: drop rows where target is NaN
        train_data = train_data.dropna(subset=[target])
        
        # Log parameters
        mlflow.log_param("target", target)
        mlflow.log_param("time_limit", time_limit)
        mlflow.log_param("presets", presets)
        mlflow.log_param("seed", seed)
        
        # Output directory for AutoGluon
        model_path = os.path.join("models", run_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            
        # Clean validation and test formats if present
        if valid_data is not None:
            if target not in valid_data.columns:
                raise ValueError(f"Target column '{target}' not found in Validation data. Make sure it has the same structure as the training dataset.")
            valid_data = valid_data.dropna(subset=[target])
            mlflow.log_param("has_validation_data", True)
        if test_data is not None:
            if target not in test_data.columns:
                raise ValueError(f"Target column '{target}' not found in Test data. Make sure the test set includes the target variable.")
            test_data = test_data.dropna(subset=[target])
            mlflow.log_param("has_test_data", True)
            
        if is_cv_task:
            mm_fit_args = {"train_data": train_data, "time_limit": time_limit}
            if valid_data is not None:
                mm_fit_args["tuning_data"] = valid_data
            
            problem_type = None
            if is_segmentation:
                problem_type = "semantic_segmentation"
            elif task_type == "Computer Vision - Object Detection":
                problem_type = "object_detection"
                
            mm_presets = "high_quality" if presets in ["best_quality", "high_quality"] else "medium_quality"
            predictor = MultiModalPredictor(label=target, problem_type=problem_type, path=model_path).fit(**mm_fit_args, presets=mm_presets)
        else:
            fit_args = {
                "train_data": train_data,
                "time_limit": time_limit, 
                "presets": presets
            }
            if cv_folds > 0:
                fit_args["num_bag_folds"] = cv_folds
                
            if valid_data is not None and cv_folds == 0:
                fit_args["tuning_data"] = valid_data
                
            if is_multilabel:
                fit_args["problem_type"] = "multiclass" # AutoGluon often handles multilabel implicitly or via multiclass depending on format. Setting multiclass to be safe if it's one-hot, or we let it infer. Let's let it infer by default or set explicitly if needed.
                # Actually, AutoGluon natively supports multilabel if properties are right, but often infer is best. We will let it infer, but we can explicitly log it.
                mlflow.log_param("is_multilabel", True)
                
            predictor = TabularPredictor(label=target, path=model_path).fit(**fit_args)
        
        # Check if cancelled before continuing
        if stop_event and stop_event.is_set():
            raise StopIteration("Training cancelled by user")
        
        eval_data = test_data if test_data is not None else (valid_data if valid_data is not None else train_data)
        
        if is_cv_task:
            scores = predictor.evaluate(eval_data)
            best_model_score = scores.get('accuracy', scores.get('roc_auc', 0.0))
            mlflow.log_metrics(scores)
            leaderboard_path = "leaderboard.csv"
            pd.DataFrame([scores]).to_csv(leaderboard_path, index=False)
        else:
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
        try:
            mlflow.log_artifacts(model_path, artifact_path="model")
            mlflow.log_param("model_type", "autogluon")
            
            # ONNX Export (Best effort for Tabular)
            if not is_cv_task:
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
                logger.info(f"Cleaned up local model folder: {model_path}")
        except Exception as e:
            logger.error(f"Failed to log model artifacts to MLflow (likely disk space): {e}")
            # Do NOT delete model_path here so the user can potentially recover it manually
            # if the MLflow log failed.
        
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
    from autogluon.tabular import TabularPredictor
    
    # Download the artifact folder
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    
    # Load the predictor from the local path
    predictor = TabularPredictor.load(local_path)
    return predictor

def get_leaderboard(predictor):
    return predictor.leaderboard(silent=True)
