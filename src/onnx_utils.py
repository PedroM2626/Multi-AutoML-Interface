import os
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Global flags for availability
ONNX_AVAILABLE = None

def _check_onnx_availability():
    global ONNX_AVAILABLE
    if ONNX_AVAILABLE is not None:
        return ONNX_AVAILABLE
    try:
        import onnx
        import onnxruntime as ort
        ONNX_AVAILABLE = True
    except Exception as e:
        logger.warning(f"ONNX or ONNXRuntime not available: {e}")
        ONNX_AVAILABLE = False
    return ONNX_AVAILABLE

def export_to_onnx(model: Any, model_type: str, target_col: str, output_path: str, input_sample: Optional[Any] = None) -> str:
    """
    Exports a trained model to ONNX format.
    Supports: flaml, pycaret, autogluon (tabular), autokeras (tensorflow).
    """
    if not _check_onnx_availability():
        raise ImportError("ONNX or ONNXRuntime is not available in this environment.")
    
    import onnx
    import pandas as pd
    import numpy as np
    
    logger.info(f"Exporting {model_type} model to ONNX: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        if model_type in ["flaml", "pycaret", "tpot"]:
            from skl2onnx import to_onnx
            
            if input_sample is None:
                raise ValueError("input_sample is required for scikit-learn based ONNX export")
            
            if isinstance(input_sample, pd.DataFrame) and target_col in input_sample.columns:
                input_sample = input_sample.drop(columns=[target_col])
            
            onx = to_onnx(model, input_sample[:1], initial_types=None)
            with open(output_path, "wb") as f:
                f.write(onx.SerializeToString())

        elif model_type == "autokeras":
            import tf2onnx
            import tensorflow as tf
            
            if input_sample is None:
                raise ValueError("input_sample is required for TensorFlow/AutoKeras ONNX export")
                
            input_signature = [tf.TensorSpec([None] + list(input_sample.shape[1:]), tf.float32, name='input')]
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
            onnx.save_model(onnx_model, output_path)

        elif model_type == "autogluon":
            try:
                model.export_onnx(output_path)
            except AttributeError:
                logger.warning("AutoGluon model does not support direct export_onnx.")
                raise NotImplementedError("AutoGluon ONNX export fallback not implemented.")

        else:
            raise ValueError(f"Unsupported model type for ONNX export: {model_type}")

        logger.info(f"Successfully exported model to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to export {model_type} model to ONNX: {e}")
        raise

def load_onnx_session(onnx_path: str):
    """Loads an ONNX model into an inference session."""
    if not _check_onnx_availability():
        raise ImportError("ONNXRuntime is not available.")
    
    import onnxruntime as ort
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    return ort.InferenceSession(onnx_path)

def predict_onnx(session: Any, df: Any) -> Any:
    """Runs inference on a DataFrame using an ONNX session."""
    import numpy as np
    
    inputs = {}
    for node in session.get_inputs():
        name = node.name
        if name in df.columns:
            inputs[name] = df[[name]].values.astype(np.float32)
        else:
            if len(session.get_inputs()) == 1:
                inputs[name] = df.values.astype(np.float32)
                break
    
    outputs = session.run(None, inputs)
    return outputs[0]
