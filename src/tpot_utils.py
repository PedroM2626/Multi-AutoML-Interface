import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shutil
import logging
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, r2_score
from tpot import TPOTClassifier, TPOTRegressor
from src.mlflow_utils import safe_set_experiment

logger = logging.getLogger(__name__)

def detect_problem_type(y):
    """Detect if problem is classification or regression"""
    if pd.api.types.is_numeric_dtype(y):
        unique_values = y.nunique()
        if unique_values <= 20 and all(y % 1 == 0 for val in y.dropna()):
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'

def create_feature_pipeline(df, target_column, text_columns=None):
    """Create feature engineering pipeline for TPOT"""
    # Identify column types
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target column from features
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)
    
    # Handle text columns separately
    if text_columns is None:
        text_columns = []
        # Auto-detect text columns (high cardinality object columns)
        for col in categorical_columns:
            if df[col].nunique() > len(df) * 0.5 and df[col].dtype == 'object':
                text_columns.append(col)
                categorical_columns.remove(col)
    
    transformers = []
    
    # Text processing
    if text_columns:
        for col in text_columns:
            transformers.append((f'tfidf_{col}', TfidfVectorizer(
                max_features=500, 
                ngram_range=(1, 2), 
                stop_words='english',
                dtype=np.float64,
                token_pattern=r'(?u)\b\w+\b'  # Handle empty strings better
            ), col))
    
    # Numerical features
    if numerical_columns:
        transformers.append(('num', StandardScaler(), numerical_columns))
    
    # Categorical features (non-text)
    if categorical_columns:
        # For TPOT, we use OrdinalEncoder to prevent dimension explosion on high cardinality
        transformers.append(('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_columns))
    
    if transformers:
        # TPOT requires dense arrays for most operations, so we force sparse_threshold=0
        preprocessor = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
    else:
        preprocessor = None
    
    return preprocessor, text_columns, categorical_columns, numerical_columns

def prepare_data_for_tpot(df, target_column, test_size=0.2, random_state=42):
    """Prepare data for TPOT training"""
    # Drop rows with missing target
    df_clean = df.dropna(subset=[target_column]).copy()
    
    # Split features and target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column].copy()
    
    # Handle missing values in features
    # For text columns, fill with empty string
    # For numerical columns, fill with median
    # For categorical columns, fill with mode
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('').astype(str)
        else:
            X[col] = X[col].fillna(X[col].median())
    
    # Convert all object columns to string to avoid mixed types
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype(str)
    
    # Handle target encoding for classification
    problem_type = detect_problem_type(y)
    if problem_type == 'classification' and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        label_encoder = None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if problem_type == 'classification' else None
    )
    
    return X_train, X_test, y_train, y_test, problem_type, label_encoder

def train_tpot_model(df, target_column, run_name, generations=5, population_size=20, cv=5, 
                    scoring=None, max_time_mins=30, max_eval_time_mins=5, random_state=42, 
                    verbosity=2, n_jobs=-1, config_dict='TPOT sparse'):
    """
    Train TPOT model with MLflow tracking
    """
    try:
        # Setup experiment
        safe_set_experiment("TPOT_Experiments")
        
        # Prepare data
        X_train, X_test, y_train, y_test, problem_type, label_encoder = prepare_data_for_tpot(
            df, target_column, random_state=random_state
        )
        
        # Create feature pipeline
        preprocessor, text_columns, cat_columns, num_columns = create_feature_pipeline(X_train, target_column)
        
        # Process features
        if preprocessor is not None:
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
        else:
            X_train_processed = X_train
            X_test_processed = X_test
        
        logger.info(f"Problem type: {problem_type}")
        logger.info(f"Training data shape: {X_train_processed.shape}")
        logger.info(f"Test data shape: {X_test_processed.shape}")
        
        # Check for high dimensional data and adjust parameters
        n_features = X_train_processed.shape[1]
        n_samples = X_train_processed.shape[0]
        
        if n_features > 10000:
            logger.warning(f"High dimensional data detected: {n_features} features")
            # Reduce complexity for high dimensional data
            generations = min(generations, 2)
            population_size = min(population_size, 10)
            max_eval_time_mins = min(max_eval_time_mins, 2)
            config_dict = 'TPOT light'
            logger.info(f"Adjusted parameters for high dimensional data: generations={generations}, population_size={population_size}")
        
        if n_samples > 50000:
            logger.warning(f"Large dataset detected: {n_samples} samples")
            # Reduce complexity for large datasets
            cv = min(cv, 3)
            max_eval_time_mins = min(max_eval_time_mins, 1)
            logger.info(f"Adjusted parameters for large dataset: cv={cv}, max_eval_time_mins={max_eval_time_mins}")
        
        # Default scoring based on problem type
        if scoring is None:
            if problem_type == 'classification':
                scoring = 'f1_macro'
            else:
                scoring = 'neg_mean_squared_error'
        
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Iniciando treinamento TPOT para a run: {run_name}")
            
            # Choose TPOT class based on problem type
            if problem_type == 'classification':
                tpot = TPOTClassifier(
                    generations=generations,
                    population_size=population_size,
                    cv=cv,
                    scoring=scoring,
                    max_time_mins=max_time_mins,
                    max_eval_time_mins=max_eval_time_mins,
                    random_state=random_state,
                    verbosity=verbosity,
                    n_jobs=n_jobs,
                    config_dict=config_dict
                )
            else:
                tpot = TPOTRegressor(
                    generations=generations,
                    population_size=population_size,
                    cv=cv,
                    scoring=scoring,
                    max_time_mins=max_time_mins,
                    max_eval_time_mins=max_eval_time_mins,
                    random_state=random_state,
                    verbosity=verbosity,
                    n_jobs=n_jobs,
                    config_dict=config_dict
                )
            
            # Log parameters
            mlflow.log_param("problem_type", problem_type)
            mlflow.log_param("generations", generations)
            mlflow.log_param("population_size", population_size)
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.log_param("max_time_mins", max_time_mins)
            mlflow.log_param("max_eval_time_mins", max_eval_time_mins)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("config_dict", config_dict)
            mlflow.log_param("n_jobs", n_jobs)
            mlflow.log_param("target", target_column)
            mlflow.log_param("n_features", n_features)
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("text_columns", text_columns)
            mlflow.log_param("categorical_columns", cat_columns)
            mlflow.log_param("numerical_columns", num_columns)
            
            # Train model with error handling
            try:
                start_time = time.time()
                tpot.fit(X_train_processed, y_train)
                training_duration = time.time() - start_time
                
                logger.info(f"Treinamento concluído em {training_duration:.2f} segundos")
                
            except Exception as tpot_error:
                logger.error(f"Erro durante treinamento TPOT: {tpot_error}")
                
                # Try with simpler configuration
                logger.info("Tentando com configuração mais simples...")
                tpot = TPOTClassifier(
                    generations=1,
                    population_size=5,
                    cv=2,
                    scoring='accuracy' if problem_type == 'classification' else 'neg_mean_squared_error',
                    max_time_mins=min(max_time_mins, 5),
                    max_eval_time_mins=1,
                    random_state=random_state,
                    verbosity=1,
                    n_jobs=1,
                    config_dict='TPOT light'
                )
                
                tpot.fit(X_train_processed, y_train)
                training_duration = time.time() - start_time
                logger.info(f"Treinamento simplificado concluído em {training_duration:.2f} segundos")
            
            # Predictions
            y_pred = tpot.predict(X_test_processed)
            
            # Calculate metrics
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro')
                f1_weighted = f1_score(y_test, y_pred, average='weighted')
                
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"F1-Score (macro): {f1_macro:.4f}")
                logger.info(f"F1-Score (weighted): {f1_weighted:.4f}")
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_macro", f1_macro)
                mlflow.log_metric("f1_weighted", f1_weighted)
                
                # Classification report
                try:
                    if label_encoder is not None:
                        # Convert back to original labels for report
                        y_test_orig = label_encoder.inverse_transform(y_test)
                        y_pred_orig = label_encoder.inverse_transform(y_pred)
                        class_names = label_encoder.classes_
                    else:
                        y_test_orig = y_test
                        y_pred_orig = y_pred
                        class_names = [f"Class_{i}" for i in np.unique(y_test)]
                    
                    report = classification_report(y_test_orig, y_pred_orig, target_names=class_names)
                    logger.info(f"\nClassification Report:\n{report}")
                    
                    # Save classification report
                    report_path = f"classification_report_{run_name}.txt"
                    with open(report_path, "w") as f:
                        f.write(f"TPOT AutoML Classification Report\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(f"Problem Type: {problem_type}\n")
                        f.write(f"Best Pipeline: {tpot.fitted_pipeline_}\n\n")
                        f.write(report)
                    
                    mlflow.log_artifact(report_path)
                    
                except Exception as e:
                    logger.warning(f"Não foi possível gerar relatório de classificação: {e}")
            
            else:  # Regression
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"MSE: {mse:.4f}")
                logger.info(f"RMSE: {rmse:.4f}")
                logger.info(f"R²: {r2:.4f}")
                
                # Log metrics
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
            
            mlflow.log_metric("training_duration", training_duration)
            
            # Create complete pipeline for saving
            if preprocessor is not None:
                final_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', tpot.fitted_pipeline_)
                ])
            else:
                final_pipeline = tpot.fitted_pipeline_
            
            # Save model info
            model_info = {
                "problem_type": problem_type,
                "best_pipeline": str(tpot.fitted_pipeline_),
                "generations": generations,
                "population_size": population_size,
                "cv": cv,
                "scoring": scoring,
                "training_duration": training_duration,
                "n_features": n_features,
                "n_samples": n_samples
            }
            
            if problem_type == 'classification':
                model_info.update({
                    "accuracy": accuracy if 'accuracy' in locals() else None,
                    "f1_macro": f1_macro if 'f1_macro' in locals() else None,
                    "f1_weighted": f1_weighted if 'f1_weighted' in locals() else None
                })
            else:
                model_info.update({
                    "mse": mse if 'mse' in locals() else None,
                    "rmse": rmse if 'rmse' in locals() else None,
                    "r2": r2 if 'r2' in locals() else None
                })
            
            # Export pipeline
            pipeline_path = f"tpot_models/best_pipeline_{run_name}.py"
            os.makedirs("tpot_models", exist_ok=True)
            tpot.export(pipeline_path)
            logger.info(f"Pipeline exportado para {pipeline_path}")
            
            # Save model info
            info_path = f"tpot_models/model_info_{run_name}.txt"
            with open(info_path, "w") as f:
                f.write("TPOT AutoML Model Information\n")
                f.write(f"{'='*50}\n\n")
                for key, value in model_info.items():
                    f.write(f"{key}: {value}\n")
            
            mlflow.log_artifact(pipeline_path)
            mlflow.log_artifact(info_path)
            
            # Log the fitted pipeline
            mlflow.sklearn.log_model(final_pipeline, "model", registered_model_name=f"TPOT_{run_name}")
            
            logger.info("Modelo TPOT registrado no MLflow com sucesso")
            
            return tpot, final_pipeline, run.info.run_id, model_info
            
    except Exception as e:
        logger.error(f"Erro durante treinamento TPOT: {e}")
        raise

def load_tpot_model(run_id, model_path="model"):
    """Load TPOT model from MLflow"""
    try:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_path}")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo TPOT: {e}")
        raise

def predict_with_tpot(model, data, preprocessor=None):
    """Make predictions with TPOT model"""
    try:
        if preprocessor is not None:
            data_processed = preprocessor.transform(data)
        else:
            data_processed = data
        
        predictions = model.predict(data_processed)
        return predictions
    except Exception as e:
        logger.error(f"Erro durante predição TPOT: {e}")
        raise
