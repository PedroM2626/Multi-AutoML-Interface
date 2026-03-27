import sys
import types

import numpy as np
import pandas as pd


import pytest


@pytest.fixture
def sample_h2o_df():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feature1": rng.normal(size=150),
            "feature2": rng.normal(size=150),
            "feature3": rng.choice(["A", "B", "C"], size=150),
            "feature4": rng.uniform(0, 100, size=150),
            "target": rng.choice(["Class0", "Class1", "Class2"], size=150),
        }
    )
    return df


def _run_h2o_basic_flow():
    from src.h2o_utils import cleanup_h2o, initialize_h2o

    instance = initialize_h2o()
    cleanup_h2o()
    return instance


def _run_h2o_train_load_predict(df: pd.DataFrame):
    from src.h2o_utils import load_h2o_model, predict_with_h2o, train_h2o_model

    automl, run_id = train_h2o_model(
        train_data=df,
        target="target",
        run_name="test_h2o_run",
        max_runtime_secs=30,
        max_models=3,
        nfolds=2,
        balance_classes=True,
    )

    loaded_model = load_h2o_model(run_id)
    test_data = df.head(8).drop(columns=["target"])
    predictions = predict_with_h2o(loaded_model, test_data)

    return automl, run_id, loaded_model, predictions


def _read_mlflow_h2o_runs():
    import mlflow

    experiment = mlflow.get_experiment_by_name("H2O_Experiments")
    if not experiment:
        return pd.DataFrame()

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs


def test_h2o_basic_initialization_and_cleanup(monkeypatch):
    calls = []

    def fake_initialize_h2o():
        calls.append("init")
        return {"status": "ok"}

    def fake_cleanup_h2o():
        calls.append("cleanup")

    fake_module = types.SimpleNamespace(
        initialize_h2o=fake_initialize_h2o,
        cleanup_h2o=fake_cleanup_h2o,
    )
    monkeypatch.setitem(sys.modules, "src.h2o_utils", fake_module)

    instance = _run_h2o_basic_flow()

    assert instance == {"status": "ok"}
    assert calls == ["init", "cleanup"]


def test_h2o_training_load_prediction_flow(monkeypatch, sample_h2o_df):
    class FakeAutoML:
        leader = types.SimpleNamespace(model_id="h2o_leader_1")

    class FakeLoadedModel:
        pass

    fake_module = types.SimpleNamespace(
        train_h2o_model=lambda **kwargs: (FakeAutoML(), "run_h2o_123"),
        load_h2o_model=lambda run_id: FakeLoadedModel() if run_id == "run_h2o_123" else None,
        predict_with_h2o=lambda model, data: np.array(["Class0"] * len(data)),
    )
    monkeypatch.setitem(sys.modules, "src.h2o_utils", fake_module)

    automl, run_id, loaded_model, predictions = _run_h2o_train_load_predict(sample_h2o_df)

    assert run_id == "run_h2o_123"
    assert automl.leader.model_id == "h2o_leader_1"
    assert loaded_model is not None
    assert predictions.tolist() == ["Class0"] * 8


def test_mlflow_h2o_experiment_lookup(monkeypatch):
    fake_experiment = types.SimpleNamespace(experiment_id="42")

    def fake_get_experiment_by_name(name):
        assert name == "H2O_Experiments"
        return fake_experiment

    def fake_search_runs(experiment_ids):
        assert experiment_ids == ["42"]
        return pd.DataFrame(
            {
                "run_id": ["r1", "r2"],
                "status": ["FINISHED", "FINISHED"],
            }
        )

    fake_mlflow = types.SimpleNamespace(
        get_experiment_by_name=fake_get_experiment_by_name,
        search_runs=fake_search_runs,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    runs = _read_mlflow_h2o_runs()

    assert not runs.empty
    assert runs["run_id"].tolist() == ["r1", "r2"]
