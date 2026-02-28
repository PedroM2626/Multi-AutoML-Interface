import os
import pandas as pd
import mlflow
import shutil
import logging
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.mlflow_utils import safe_set_experiment

logger = logging.getLogger(__name__)

def check_java_availability():
    """Checks if Java is available in the system"""
    try:
        import subprocess
        import os
        
        # Try to find Java in PATH
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True
        
        # If not found in PATH, try common paths on Windows
        java_paths = [
            r"C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot\bin\java.exe",
            r"C:\Program Files\Eclipse Adoptium\jdk-11.0.23.9-hotspot\bin\java.exe",
            r"C:\Program Files\Java\jdk-11\bin\java.exe",
            r"C:\Program Files\Java\jdk-17\bin\java.exe",
        ]
        
        for java_path in java_paths:
            if os.path.exists(java_path):
                result = subprocess.run([java_path, '-version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return True
        
        return False
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False

def initialize_h2o():
    """Initializes the H2O cluster with Java check"""
    if not check_java_availability():
        raise RuntimeError(
            "Java is not installed on the system. H2O AutoML requires Java to function.\n\n"
            "Options:\n"
            "1. Install Java locally (JRE/JDK)\n"
            "2. Use Docker: docker build -t multi-automl-interface . && docker run -p 8501:8501 multi-automl-interface\n"
            "3. Use AutoGluon or FLAML as alternatives (they do not require Java)\n"
            "\nTo install Java on Windows:\n"
            "- Download from: https://adoptium.net/\n"
            "- Or use: winget install EclipseAdoptium.Temurin.11.JDK"
        )
    
    try:
        import h2o
        h2o.init(max_mem_size="4G", nthreads=-1)
        logger.info("H2O Cluster initialized successfully")
        return h2o
    except Exception as e:
        logger.error(f"Error initializing H2O: {e}")
        raise

def cleanup_h2o():
    """Finalizes the H2O cluster"""
    try:
        import h2o
        h2o.cluster().shutdown()
        logger.info("H2O Cluster finalized")
    except Exception as e:
        logger.warning(f"Error finalizing H2O: {e}")

def prepare_data_for_h2o(train_data: pd.DataFrame, target: str):
    """Prepares data for H2O AutoML"""
    import h2o
    
    # Drop null values
    train_data_clean = train_data.dropna(subset=[target])
    
    # For textual data, create basic numerical features
    if train_data_clean.select_dtypes(include=['object']).shape[1] > 0:
        logger.info("Text columns detected, generating basic numerical features...")
        
        # For each text column, build basic features
        for col in train_data_clean.select_dtypes(include=['object']).columns:
            if col != target:
                # Text length
                train_data_clean[f'{col}_length'] = train_data_clean[col].astype(str).str.len()
                # Word count
                train_data_clean[f'{col}_word_count'] = train_data_clean[col].astype(str).str.split().str.len()
                
        # Drop text columns except target
        text_cols = train_data_clean.select_dtypes(include=['object']).columns
        text_cols = [col for col in text_cols if col != target]
        train_data_clean = train_data_clean.drop(columns=text_cols)
    
    # Convert to H2OFrame
    h2o_frame = h2o.H2OFrame(train_data_clean)
    
    # Convert target to factor (categorical) if classification
    if train_data_clean[target].dtype == 'object' or train_data_clean[target].nunique() < 20:
        h2o_frame[target] = h2o_frame[target].asfactor()
    
    return h2o_frame, train_data_clean

def train_h2o_model(train_data: pd.DataFrame, target: str, run_name: str, 
                   valid_data: pd.DataFrame = None, test_data: pd.DataFrame = None,
                   max_runtime_secs: int = 300, max_models: int = 10, 
                   nfolds: int = 3, balance_classes: bool = True, seed: int = 42,
                   sort_metric: str = "AUTO", exclude_algos: list = None):
    """
    Trains H2O AutoML model and registers in MLflow
    """
    import h2o
    from h2o.automl import H2OAutoML
    
    safe_set_experiment("H2O_Experiments")
    logging.info(f"Starting H2O AutoML training for run: {run_name}")
    
    # Initialize H2O
    h2o_instance = initialize_h2o()
    
    try:
        with mlflow.start_run(run_name=run_name) as run:
            # Prepare data
            h2o_frame, clean_data = prepare_data_for_h2o(train_data, target)
            
            # Log parameters
            mlflow.log_param("target", target)
            mlflow.log_param("max_runtime_secs", max_runtime_secs)
            mlflow.log_param("max_models", max_models)
            mlflow.log_param("nfolds", nfolds)
            mlflow.log_param("balance_classes", balance_classes)
            mlflow.log_param("seed", seed)
            mlflow.log_param("sort_metric", sort_metric)
            mlflow.log_param("model_type", "h2o_automl")
            if exclude_algos:
                mlflow.log_param("exclude_algos", exclude_algos)
            
            # Define features (all except target)
            features = [col for col in clean_data.columns if col != target]
            mlflow.log_param("features", features)
            
            # Configurar AutoML
            aml = H2OAutoML(
                max_runtime_secs=max_runtime_secs,
                max_models=max_models,
                seed=seed,
                nfolds=nfolds,
                balance_classes=balance_classes,
                keep_cross_validation_predictions=True,
                keep_cross_validation_models=False,
                verbosity='info',
                sort_metric=sort_metric,
                exclude_algos=exclude_algos or []
            )
            
            # Prepare test and validation data if present
            h2o_valid = None
            if valid_data is not None:
                if target not in valid_data.columns:
                    raise ValueError(f"Target column '{target}' not found in Validation data.")
                valid_data = valid_data.dropna(subset=[target])
                h2o_valid, _ = prepare_data_for_h2o(valid_data, target)
                mlflow.log_param("has_validation_data", True)
                
            h2o_test = None
            if test_data is not None:
                if target not in test_data.columns:
                    raise ValueError(f"Target column '{target}' not found in Test data.")
                test_data = test_data.dropna(subset=[target])
                h2o_test, _ = prepare_data_for_h2o(test_data, target)
                mlflow.log_param("has_test_data", True)
            
            # Train model
            logger.info("Starting H2O AutoML training...")
            start_time = time.time()
            train_kwargs = {"x": features, "y": target, "training_frame": h2o_frame}
            if h2o_valid is not None:
                train_kwargs["validation_frame"] = h2o_valid
            if h2o_test is not None:
                train_kwargs["leaderboard_frame"] = h2o_test
                
            aml.train(**train_kwargs)
            training_duration = time.time() - start_time
            
            logger.info(f"Training completed in {training_duration:.2f} seconds")
            
            # Get leaderboard
            leaderboard = aml.leaderboard
            
            # Check if leaderboard is empty
            if leaderboard.nrow == 0:
                logger.warning("⚠️ No models trained. Leaderboard is empty.")
                logger.warning("This can happen if:")
                logger.warning("1. Max runtime is too short")
                logger.warning("2. Data is not adequate for algorithms")
                logger.warning("3. Data has underlying issues")
                
                # Log basic metrics even without models
                mlflow.log_metric("total_models_trained", 0)
                mlflow.log_metric("training_duration", training_duration)
                mlflow.log_metric("best_model_score", 0.0)
                
                # Return AutoML even without models
                return aml, run.info.run_id
            
            logger.info("\nTop 5 models:")
            print(leaderboard.head(5))
            
            # Save leaderboard as metric with safe wrapper
            try:
                # Check available columns in leaderboard
                leaderboard_df = None
                try:
                    leaderboard_df = leaderboard.as_data_frame()
                    logger.info(f"Available columns: {list(leaderboard_df.columns)}")
                except Exception as e:
                    logger.warning(f"Could not convert leaderboard to DataFrame: {e}")
                
                # Try to get the best available metric
                best_model_score = 0.0
                if leaderboard_df is not None and len(leaderboard_df) > 0:
                    # Search for metrics in preference order
                    for metric in ['auc', 'logloss', 'rmse', 'mae', 'r2']:
                        if metric in leaderboard_df.columns:
                            best_model_score = leaderboard_df.iloc[0][metric]
                            logger.info(f"Using metric '{metric}': {best_model_score}")
                            break
                    
                    mlflow.log_metric("total_models_trained", len(leaderboard_df))
                else:
                    # Fallback: use the first value in H2O leaderboard
                    try:
                        available_columns = leaderboard.columns
                        logger.info(f"Available H2O columns: {available_columns}")
                        
                        # Try accessing first row, first metric col
                        if len(available_columns) > 0:
                            first_col = available_columns[0]
                            best_model_score = leaderboard[0, first_col]
                            logger.info(f"Using first available column '{first_col}': {best_model_score}")
                        
                        mlflow.log_metric("total_models_trained", leaderboard.nrow)
                    except Exception as e:
                        logger.warning(f"Could not extract metrics from leaderboard: {e}")
                        mlflow.log_metric("total_models_trained", 0)
                
                mlflow.log_metric("best_model_score", best_model_score)
                mlflow.log_metric("training_duration", training_duration)
                
            except Exception as e:
                logger.warning(f"Error processing leaderboard metrics: {e}")
                # Default fallback
                mlflow.log_metric("best_model_score", 0.0)
                mlflow.log_metric("training_duration", training_duration)
                mlflow.log_metric("total_models_trained", 0)
            
            # Try saving leaderboard with error handling
            try:
                leaderboard_df = leaderboard.as_data_frame()
                leaderboard_path = f"h2o_leaderboard_{run_name}.csv"
                leaderboard_df.to_csv(leaderboard_path, index=False)
                mlflow.log_artifact(leaderboard_path)
            except Exception as e:
                logger.warning(f"Could not save leaderboard as CSV: {e}")
                # Save as plain text if CSV fails
                try:
                    leaderboard_text = str(leaderboard.head(10))
                    leaderboard_path = f"h2o_leaderboard_{run_name}.txt"
                    with open(leaderboard_path, "w") as f:
                        f.write(f"H2O AutoML Leaderboard - {run_name}\n")
                        f.write("=" * 50 + "\n")
                        f.write(leaderboard_text)
                    mlflow.log_artifact(leaderboard_path)
                except Exception as e2:
                    logger.warning(f"Could not save leaderboard as text: {e2}")
            
            # Save local model (only if there are models)
            if hasattr(aml, 'leader') and aml.leader is not None:
                model_dir = "models/h2o_models"
                os.makedirs(model_dir, exist_ok=True)
                model_path = f"{model_dir}/h2o_model_{run_name}"
                
                # Save best model (leader) rather than AutoML object
                best_model = aml.leader
                h2o.save_model(best_model, path=model_path)
                logger.info(f"Model saved at: {model_path}")
                
                # Log model to MLflow
                temp_model_path = f"temp_h2o_model_{run_name}"
                os.makedirs(temp_model_path, exist_ok=True)
                h2o.save_model(best_model, path=temp_model_path)
                mlflow.log_artifacts(temp_model_path, artifact_path="model")
                
                # Clean temp directory
                import shutil
                if os.path.exists(temp_model_path):
                    shutil.rmtree(temp_model_path)
            else:
                logger.warning("⚠️ No model to save (no models were trained)")
                
                # Create a placeholder file explaining the situation
                no_model_path = f"no_model_{run_name}.txt"
                with open(no_model_path, "w") as f:
                    f.write(f"H2O AutoML - {run_name}\n")
                    f.write("=" * 50 + "\n")
                    f.write("No models were trained during this run.\n")
                    f.write("Possible causes:\n")
                    f.write("1. Insufficient training time\n")
                    f.write("2. Data inadequate for algorithms\n")
                    f.write("3. Data quality issues\n")
                    f.write(f"Training time: {training_duration:.2f} seconds\n")
                
                mlflow.log_artifact(no_model_path)
            
            # Generate classification report for classification tasks (only if models exist)
            if (clean_data[target].dtype == 'object' or clean_data[target].nunique() < 20) and hasattr(aml, 'leader') and aml.leader is not None:
                try:
                    best_model = aml.leader
                    predictions = best_model.predict(h2o_frame)
                    pred_array = predictions['predict'].as_data_frame()['predict'].values
                    true_labels = clean_data[target].values
                    
                    # Calculate metrics
                    accuracy = accuracy_score(true_labels, pred_array)
                    f1_macro = f1_score(true_labels, pred_array, average='macro')
                    f1_weighted = f1_score(true_labels, pred_array, average='weighted')
                    
                    logger.info(f"\nValidation metrics:")
                    logger.info(f"Accuracy: {accuracy:.4f}")
                    logger.info(f"F1-Score (macro): {f1_macro:.4f}")
                    logger.info(f"F1-Score (weighted): {f1_weighted:.4f}")
                    
                    # Log validation metrics
                    mlflow.log_metric("validation_accuracy", accuracy)
                    mlflow.log_metric("validation_f1_macro", f1_macro)
                    mlflow.log_metric("validation_f1_weighted", f1_weighted)
                    
                    # Generate report
                    class_report = classification_report(true_labels, pred_array)
                    report_path = f"classification_report_{run_name}.txt"
                    with open(report_path, "w") as f:
                        f.write(f"Classification Report - H2O AutoML\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(class_report)
                    
                    mlflow.log_artifact(report_path)
                    
                except Exception as e:
                    logger.warning(f"Could not generate classification report: {e}")
            else:
                logger.info("Skipping report generation (no models trained or not a classification problem)")
            
            # Clean temporary files
            if os.path.exists(leaderboard_path):
                os.remove(leaderboard_path)
            
            report_path_temp = f"classification_report_{run_name}.txt"
            if os.path.exists(report_path_temp):
                os.remove(report_path_temp)
            
            return aml, run.info.run_id
            
    except Exception as e:
        logger.error(f"Error during H2O training: {e}")
        raise
    finally:
        cleanup_h2o()

def load_h2o_model(run_id: str):
    """
    Loads H2O model from MLflow
    """
    import h2o
    
    # Initialize H2O if not active
    try:
        h2o.init(max_mem_size="2G", nthreads=-1)
    except:
        pass  # H2O might already be active
    
    try:
        # Download artifact
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        
        # Find and load the model
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.endswith(".zip"):
                    model_path = os.path.join(root, file)
                    logger.info(f"Loading H2O model from: {model_path}")
                    model = h2o.load_model(model_path)
                    
                    # Check if model loaded correctly
                    if model is None:
                        raise ValueError("Loaded model is None")
                    
                    logger.info(f"H2O model loaded successfully: {type(model)}")
                    return model
        
        raise FileNotFoundError("H2O model not found in artifacts.")
        
    except Exception as e:
        logger.error(f"Error loading H2O model: {e}")
        raise

def predict_with_h2o(model, data: pd.DataFrame):
    """
    Makes predictions using an H2O model
    """
    import h2o
    
    # Check if model is valid
    if model is None:
        raise ValueError("H2O model is None. Ensure the model was loaded correctly.")
    
    try:
        logger.info(f"Starting prediction with H2O model: {type(model)}")
        
        # Prepare data the same way as training
        h2o_frame, _ = prepare_data_for_h2o(data, target="dummy")  # target not used for prediction
        
        # Do predictions
        predictions = model.predict(h2o_frame)
        pred_array = predictions['predict'].as_data_frame()['predict'].values
        
        logger.info(f"Prediction complete: {len(pred_array)} predictions")
        return pred_array
        
    except Exception as e:
        logger.error(f"Error in H2O prediction: {e}")
        raise
    finally:
        # Clean H2O frame to release memory
        try:
            if 'h2o_frame' in locals():
                h2o_frame = None
        except:
            pass
