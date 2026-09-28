import sys
import types

import pandas as pd
import pytest

from src.experiment_manager import ExperimentEntry, ExperimentManager
from src.prediction_service import _get_pycaret_module_name, load_model_by_framework, run_predictions


class DummyPredictor:
    def __init__(self, values):
        self._values = values

    def predict(self, _df):
        return self._values


def test_cancel_flow_keeps_cancelled_status():
    manager = ExperimentManager()
    entry = ExperimentEntry(key="exp_1", metadata={"framework": "FLAML"}, status="running")
    manager.add(entry)

    manager.cancel("exp_1")

    assert entry.stop_event.is_set() is True
    assert entry.status == "cancelled"
    assert entry.finished_at is not None


def test_cancelled_run_records_late_result_without_relabelling():
    manager = ExperimentManager()
    entry = ExperimentEntry(key="exp_2", metadata={"framework": "FLAML"}, status="running")
    manager.add(entry)
    manager.cancel("exp_2")

    entry.result_queue.put({"success": True, "run_id": "abc123"})
    manager.refresh_all()

    assert entry.status == "cancelled"
    assert entry.result == {"success": True, "run_id": "abc123"}


def test_load_by_run_id_autogluon_branch(monkeypatch):
    expected = {"predictor": "mock"}

    fake_module = types.SimpleNamespace(load_model_from_mlflow=lambda run_id: {"run_id": run_id, **expected})
    monkeypatch.setitem(sys.modules, "src.autogluon_utils", fake_module)

    predictor, model_type = load_model_by_framework("AutoGluon", "run_123", trust_artifacts=True)

    assert model_type == "autogluon"
    assert predictor["run_id"] == "run_123"
    assert predictor["predictor"] == "mock"


def test_load_by_run_id_requires_trusted_artifacts(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "src.autogluon_utils",
        types.SimpleNamespace(load_model_from_mlflow=lambda run_id: {}),
    )

    with pytest.raises(PermissionError):
        load_model_by_framework("AutoGluon", "run_123")


def test_load_by_run_id_rejects_path_like_run_id():
    for bad in ["../run", "runs/abc", "", "a" * 65, "run 123"]:
        with pytest.raises(ValueError):
            load_model_by_framework("FLAML", bad, trust_artifacts=True)


def test_flaml_callback_and_learner_guards():
    from src.flaml_utils import _supports_callbacks, _require_learner_packages

    # 'auto' and mixed sklearn/boosting lists reject FLAML's callbacks kwarg outright.
    assert _supports_callbacks(["lgbm"]) is True
    assert _supports_callbacks(["lgbm", "rf"]) is False
    assert _supports_callbacks("auto") is False
    assert _supports_callbacks([]) is False

    # A learner whose package is missing must be named before the search starts.
    try:
        import lightgbm  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    if installed:
        _require_learner_packages(["lgbm"])
    else:
        with pytest.raises(ImportError, match="lightgbm"):
            _require_learner_packages(["lgbm"])


def test_load_by_run_id_invalid_framework():
    with pytest.raises(ValueError):
        load_model_by_framework("UnknownFramework", "run_123")


def test_batch_prediction_drops_target_column():
    predictor = DummyPredictor(values=[1, 0])
    predict_df = pd.DataFrame({"f1": [10, 20], "target": [0, 1]})

    result_df, pred_input_df = run_predictions(
        predictor=predictor,
        model_type="flaml",
        predict_df=predict_df,
        target_col="target",
        training_df=None,
    )

    assert "target" not in pred_input_df.columns
    assert "Predictions" in result_df.columns
    assert result_df["Predictions"].tolist() == [1, 0]


def test_batch_prediction_decodes_categorical_target_ids():
    predictor = DummyPredictor(values=[0, 1])
    predict_df = pd.DataFrame({"f1": [10, 20]})
    training_df = pd.DataFrame({"f1": [1, 2], "target": ["cat", "dog"]})

    result_df, _ = run_predictions(
        predictor=predictor,
        model_type="tpot",
        predict_df=predict_df,
        target_col="target",
        training_df=training_df,
    )

    assert result_df["Predictions"].tolist() == ["cat", "dog"]


def test_batch_prediction_drops_multiple_target_columns():
    predictor = DummyPredictor(values=[1, 0])
    predict_df = pd.DataFrame({"f1": [10, 20], "target_a": [0, 1], "target_b": [1, 0]})

    result_df, pred_input_df = run_predictions(
        predictor=predictor,
        model_type="flaml",
        predict_df=predict_df,
        target_col=["target_a", "target_b"],
        training_df=None,
    )

    assert "target_a" not in pred_input_df.columns
    assert "target_b" not in pred_input_df.columns
    assert "Predictions" in result_df.columns
    assert result_df["Predictions"].tolist() == [1, 0]


def test_pycaret_module_resolution_for_anomaly_task():
    assert _get_pycaret_module_name("Anomaly Detection") == ".".join(["pycaret", "anomaly"])
    assert _get_pycaret_module_name("Clustering") == ".".join(["pycaret", "clustering"])
    assert _get_pycaret_module_name("Regression") == ".".join(["pycaret", "regression"])
    assert _get_pycaret_module_name("Classification") == ".".join(["pycaret", "classification"])
