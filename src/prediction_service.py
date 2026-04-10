import os
import importlib
import pandas as pd


_PYCARET_CLASSIFICATION_MODULE = ".".join(["pycaret", "classification"])


def _get_pycaret_module_name(task_type: str | None) -> str:
    if task_type == "Regression":
        return ".".join(["pycaret", "regression"])
    if task_type == "Time Series Forecasting":
        return ".".join(["pycaret", "time_series"])
    if task_type == "Anomaly Detection":
        return ".".join(["pycaret", "anomaly"])
    if task_type == "Clustering":
        return ".".join(["pycaret", "clustering"])
    return _PYCARET_CLASSIFICATION_MODULE


def load_model_by_framework(framework_name: str, run_id: str):
    """
    Load a predictor from MLflow artifacts according to selected framework label.
    Returns (predictor, normalized_model_type).
    """
    if framework_name == "AutoGluon":
        from src.autogluon_utils import load_model_from_mlflow

        return load_model_from_mlflow(run_id), "autogluon"

    if framework_name == "FLAML":
        from src.flaml_utils import load_flaml_model

        return load_flaml_model(run_id), "flaml"

    if framework_name == "H2O AutoML":
        from src.h2o_utils import load_h2o_model

        return load_h2o_model(run_id), "h2o"

    if framework_name == "TPOT":
        from src.tpot_utils import load_tpot_model

        return load_tpot_model(run_id), "tpot"

    if framework_name == "PyCaret":
        mlflow = importlib.import_module("mlflow")
        task_type = "Classification"
        try:
            run = mlflow.tracking.MlflowClient().get_run(run_id)
            task_type = run.data.params.get("task_type", "Classification")
        except Exception:
            pass

        pycaret_cls = importlib.import_module(_get_pycaret_module_name(task_type))
        pycaret_load = getattr(pycaret_cls, "load_model")

        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        model_path_without_ext = None
        for root, _, files in os.walk(local_path):
            for file_name in files:
                if file_name.endswith(".pkl"):
                    model_path_without_ext = os.path.join(root, file_name).replace(".pkl", "")
                    break
            if model_path_without_ext is not None:
                break

        if model_path_without_ext is None:
            raise FileNotFoundError("PyCaret .pkl not found.")

        return pycaret_load(model_path_without_ext), "pycaret"

    if framework_name == "Lale":
        mlflow = importlib.import_module("mlflow")
        joblib = importlib.import_module("joblib")
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        bundle = None
        for root, _, files in os.walk(local_path):
            for file_name in files:
                if file_name.endswith(".pkl"):
                    bundle = joblib.load(os.path.join(root, file_name))
                    break
            if bundle is not None:
                break

        if bundle is None:
            raise FileNotFoundError("Lale .pkl not found.")

        return bundle, "lale"

    raise ValueError(f"Unsupported framework for MLflow load: {framework_name}")


def _decode_predictions_if_needed(predictions, training_df: pd.DataFrame | None, target_col: str | None):
    """Decode numeric class IDs back to original labels when target was categorical."""
    if training_df is None or not target_col or target_col not in training_df.columns:
        return predictions

    target_series = training_df[target_col]
    if not (target_series.dtype == object or str(target_series.dtype) == "category"):
        return predictions

    pred_series = pd.Series(predictions)
    if not pd.api.types.is_numeric_dtype(pred_series):
        return predictions

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.fit(target_series.astype(str))

    decoded = []
    for value in pred_series:
        try:
            index = int(value)
            if 0 <= index < len(encoder.classes_):
                decoded.append(encoder.inverse_transform([index])[0])
            else:
                decoded.append(value)
        except Exception:
            decoded.append(value)

    return decoded


def run_predictions(
    predictor,
    model_type: str,
    predict_df: pd.DataFrame,
    target_col: str | list[str] | None = None,
    training_df: pd.DataFrame | None = None,
    task_type: str | None = None,
):
    """
    Execute predictions in a unified path and return (result_df, prediction_input_df).
    """
    if predictor is None:
        raise ValueError("No model is loaded.")

    if isinstance(target_col, list):
        target_columns = [col for col in target_col if col in predict_df.columns]
    elif isinstance(target_col, str):
        target_columns = [target_col] if target_col in predict_df.columns else []
    else:
        target_columns = []

    if target_columns:
        prediction_input_df = predict_df.drop(columns=target_columns)
    else:
        prediction_input_df = predict_df.copy()

    if model_type == "autogluon":
        predictions = predictor.predict(prediction_input_df)
    elif model_type == "onnx":
        from src.onnx_utils import predict_onnx

        predictions = predict_onnx(predictor, prediction_input_df)
    elif model_type == "h2o":
        from src.h2o_utils import predict_with_h2o

        predictions = predict_with_h2o(predictor, prediction_input_df)
    elif model_type == "pycaret":
        pycaret_cls = importlib.import_module(_get_pycaret_module_name(task_type))
        pycaret_predict = getattr(pycaret_cls, "predict_model")

        preds_df = pycaret_predict(predictor, data=prediction_input_df)
        if task_type == "Anomaly Detection" and "Anomaly" in preds_df.columns:
            predictions = preds_df["Anomaly"]
        elif task_type == "Anomaly Detection" and "Label" in preds_df.columns:
            predictions = preds_df["Label"]
        elif task_type == "Clustering" and "Cluster" in preds_df.columns:
            predictions = preds_df["Cluster"]
        elif task_type == "Clustering" and "Label" in preds_df.columns:
            predictions = preds_df["Label"]
        elif "prediction_label" in preds_df.columns:
            predictions = preds_df["prediction_label"]
        elif "Label" in preds_df.columns:
            predictions = preds_df["Label"]
        else:
            predictions = preds_df.iloc[:, -1]
    elif model_type == "lale":
        if isinstance(predictor, dict):
            model = predictor["model"]
            col_encoders = predictor.get("col_encoders", {})
            y_encoder = predictor.get("y_encoder", None)
        else:
            model = predictor
            col_encoders = {}
            y_encoder = None

        transformed_df = prediction_input_df.copy()
        for col, encoder in col_encoders.items():
            if col in transformed_df.columns:
                transformed_df[col] = encoder.transform(transformed_df[[col]].astype(str)).ravel()

        for col in transformed_df.columns:
            if transformed_df[col].dtype == object:
                try:
                    transformed_df[col] = pd.to_numeric(transformed_df[col])
                except Exception:
                    transformed_df[col] = 0.0

        raw = model.predict(transformed_df.values)
        predictions = y_encoder.inverse_transform(raw) if y_encoder else raw
    else:
        predictions = predictor.predict(prediction_input_df)

    decode_target = target_col if isinstance(target_col, str) else None
    predictions = _decode_predictions_if_needed(predictions, training_df=training_df, target_col=decode_target)

    result_df = prediction_input_df.copy()
    if isinstance(predictions, pd.DataFrame):
        for column_name in predictions.columns:
            result_df[f"Prediction_{column_name}"] = predictions[column_name].values
    else:
        result_df["Predictions"] = predictions
    return result_df, prediction_input_df
