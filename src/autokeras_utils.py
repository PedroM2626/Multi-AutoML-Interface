import os
import shutil
import time
import pandas as pd
import numpy as np
import mlflow
import logging
from src.mlflow_utils import safe_set_experiment

logger = logging.getLogger(__name__)

def run_autokeras_experiment(train_data: pd.DataFrame, target: str, run_name: str, 
                            valid_data: pd.DataFrame = None, task_type: str = "Computer Vision - Image Classification",
                            time_limit: int = 60, stop_event=None, log_queue=None):
    """
    Trains an AutoKeras model for Image tasks.
    train_data contains a dataframe with 'Image_Directory' pointing to the dataset path.
    """
    safe_set_experiment("AutoKeras_Experiments")
    
    try:
        import autokeras as ak
        import tensorflow as tf
    except ImportError:
        raise ImportError("AutoKeras or TensorFlow not installed. Please install them to use AutoKeras.")

    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

    def qlog(msg):
        if log_queue:
            log_queue.put(msg)
        logger.info(msg)

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_param("framework", "autokeras")
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("time_limit", time_limit)
        
        if "Image_Directory" not in train_data.columns:
            raise ValueError("AutoKeras requires 'Image_Directory' in the training payload for CV tasks.")
            
        img_dir = train_data.iloc[0]["Image_Directory"]
        qlog(f"Scanning image directory: {img_dir}")
        
        # We need to construct tf.data.Dataset from directory
        # Since AutoKeras ImageClassifier accepts tf.data.Dataset
        batch_size = 32
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            img_dir,
            validation_split=0.2 if valid_data is None else None,
            subset="training" if valid_data is None else None,
            seed=42,
            image_size=(256, 256),
            batch_size=batch_size
        )
        
        if valid_data is None:
            val_ds = tf.keras.utils.image_dataset_from_directory(
                img_dir,
                validation_split=0.2,
                subset="validation",
                seed=42,
                image_size=(256, 256),
                batch_size=batch_size
            )
        else:
            val_img_dir = valid_data.iloc[0]["Image_Directory"]
            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_img_dir,
                seed=42,
                image_size=(256, 256),
                batch_size=batch_size
            )
            
        mlflow.log_param("num_classes", len(train_ds.class_names))
        
        model_path = os.path.join("models", run_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            
        qlog("Starting AutoKeras topology search...")
        
        # Estimate max trials based on time_limit pseudo translation (1 trial ~ 100s for small data)
        max_trials = max(1, time_limit // 100)
        
        def dataset_to_numpy(ds):
            x_all, y_all = [], []
            for x, y in ds:
                x_all.append(x.numpy())
                y_all.append(y.numpy())
            if not x_all: return None, None
            return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0)
            
        x_train, y_train = dataset_to_numpy(train_ds)
        
        x_val, y_val = None, None
        if val_ds:
            x_val, y_val = dataset_to_numpy(val_ds)
            
        if task_type == "Computer Vision - Image Classification":
            clf = ak.ImageClassifier(overwrite=True, max_trials=max_trials, directory=model_path)
            if val_ds:
                clf.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5) # Default short epoch
            else:
                clf.fit(x_train, y_train, epochs=5)
        else:
            # We don't natively support bounding boxes or segmentation masks right now without specific parser
            raise NotImplementedError(f"AutoKeras task '{task_type}' requires labels not inherently present in the directory structure or is unsupported by AutoKeras basic API.")
            
        if stop_event and stop_event.is_set():
            raise StopIteration("Training cancelled by user")

        qlog("Evaluating best model...")
        loss, accuracy = clf.evaluate(val_ds)
        mlflow.log_metric("val_loss", loss)
        mlflow.log_metric("val_accuracy", accuracy)

        qlog("Saving and logging artifacts...")
        export_path = os.path.join(model_path, "best_model")
        
        try:
            model = clf.export_model()
            model.save(export_path, save_format="tf")
            mlflow.log_artifacts(export_path, artifact_path="model")
            mlflow.log_param("model_type", "autokeras")
            qlog("AutoKeras artifacts logged successfully.")
        except Exception as e:
            qlog(f"Warning: Model export failed: {e}")

        # Return a dictionary of useful data for UI
        return {
            "run_id": run.info.run_id,
            "type": "autokeras",
            # Can't pass TF model across processes easily via queues, so we pass None
            "predictor": None
        }
