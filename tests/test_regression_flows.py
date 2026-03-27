import sys
import types

import pandas as pd
import pytest

from src.experiment_manager import ExperimentEntry, ExperimentManager
from src.prediction_service import load_model_by_framework, run_predictions


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


def test_load_by_run_id_autogluon_branch(monkeypatch):
    expected = {"predictor": "mock"}

    fake_module = types.SimpleNamespace(load_model_from_mlflow=lambda run_id: {"run_id": run_id, **expected})
    monkeypatch.setitem(sys.modules, "src.autogluon_utils", fake_module)

    predictor, model_type = load_model_by_framework("AutoGluon", "run_123")

    assert model_type == "autogluon"
    assert predictor["run_id"] == "run_123"
    assert predictor["predictor"] == "mock"


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
