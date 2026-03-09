import os
import logging
import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

def export_to_onnx(model: Any, model_type: str, target_col: str, output_path: str, input_sample: Optional[pd.DataFrame] = None) -> str:
    """
    Exports a trained model to ONNX format.
    Supports: flaml, pycaret, autogluon (tabular), autokeras (tensorflow).
    """
    logger.info(f"Exporting {model_type} model to ONNX: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        if model_type in ["flaml", "pycaret", "tpot"]:
            # These usually return or contain a scikit-learn pipeline/model
            from skl2onnx import to_onnx
            from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
            
            if input_sample is None:
                raise ValueError("input_sample is required for scikit-learn based ONNX export")
            
            # Clean input_sample: remove target if present
            if target_col in input_sample.columns:
                input_sample = input_sample.drop(columns=[target_col])
            
            # Define initial types based on input_sample
            initial_types = []
            for col, dtype in zip(input_sample.columns, input_sample.dtypes):
                if np.issubdtype(dtype, np.floating):
                    initial_types.append((col, FloatTensorType([None, 1])))
                elif np.issubdtype(dtype, np.integer):
                    initial_types.append((col, Int64TensorType([None, 1])))
                else:
                    initial_types.append((col, StringTensorType([None, 1])))
            
            # If it's a PyCaret model, we might need to extract the underlying object
            # or handle its specific pipeline structure. 
            # For simplicity, we assume 'model' is the final estimator or a simple pipeline.
            onx = to_onnx(model, input_sample[:1], initial_types=None) # skl2onnx can often infer from input
            with open(output_path, "wb") as f:
                f.write(onx.SerializeToString())

        elif model_type == "autokeras":
            import tf2onnx
            import tensorflow as tf
            
            # AutoKeras model is essentially a Keras model
            # For TensorFlow, we save to ONNX using tf2onnx
            input_signature = [tf.TensorSpec([None] + list(input_sample.shape[1:]), tf.float32, name='input')]
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
            onnx.save_model(onnx_model, output_path)

        elif model_type == "autogluon":
            # AutoGluon has its own ONNX export for Tabular via a specific method or integration
            # If not natively supported as a single call, we might need a custom wrapper
            # For now, we simulate or use the tabular export if available
            try:
                model.export_onnx(output_path)
            except AttributeError:
                logger.warning("AutoGluon model does not support direct export_onnx. Native export required.")
                # Fallback: Many AG models are scikit-learn or XGBoost based
                # This part would require more complex logic to handle AG's ensemble
                raise NotImplementedError("AutoGluon ONNX export is partially implemented.")

        else:
            raise ValueError(f"Unsupported model type for ONNX export: {model_type}")

        logger.info(f"Successfully exported model to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to export {model_type} model to ONNX: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def load_onnx_session(onnx_path: str) -> ort.InferenceSession:
    """Loads an ONNX model into an inference session."""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    return ort.InferenceSession(onnx_path)

def predict_onnx(session: ort.InferenceSession, df: pd.DataFrame) -> np.ndarray:
    """Runs inference on a DataFrame using an ONNX session."""
    # Prepare inputs: ONNX expects a dict of {input_name: numpy_array}
    # This logic assumes inputs are matched by column name
    inputs = {}
    for node in session.get_inputs():
        name = node.name
        if name in df.columns:
            inputs[name] = df[[name]].values.astype(np.float32) # Assume float32 for now
        else:
            # Fallback for models that take a single matrix
            if len(session.get_inputs()) == 1:
                inputs[name] = df.values.astype(np.float32)
                break
    
    outputs = session.run(None, inputs)
    return outputs[0] # Return first output (usually predictions)
