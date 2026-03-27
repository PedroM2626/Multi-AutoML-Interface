import types
import sys

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feature1": rng.normal(size=120),
            "feature2": rng.normal(size=120),
            "feature3": rng.choice(["A", "B", "C"], size=120),
            "feature4": rng.uniform(0, 100, size=120),
            "target": rng.choice([0, 1], size=120),
        }
    )
    df["feature3"] = df["feature3"].astype("category")
    df["target"] = df["target"].astype("category")
    return df


def _run_h2o_training_flow(df: pd.DataFrame, run_name: str):
    from h2o_utils import check_java_availability, train_h2o_model

    if not check_java_availability():
        raise RuntimeError("Java not available")

    automl, run_id = train_h2o_model(
        df,
        "target",
        run_name,
        max_runtime_secs=10,
        max_models=2,
        nfolds=2,
        balance_classes=True,
        seed=42,
        sort_metric="AUTO",
        exclude_algos=["DeepLearning"],
    )

    assert run_id
    assert hasattr(automl, "leader")
    assert automl.leader is not None
    return automl, run_id


def _run_h2o_load_flow(run_id: str):
    from h2o_utils import load_h2o_model

    model = load_h2o_model(run_id)
    assert model is not None
    return model


def _run_h2o_predict_flow(model, df: pd.DataFrame):
    from h2o_utils import predict_with_h2o

    preds = predict_with_h2o(model, df.drop(columns=["target"]).head(5))
    assert len(preds) == 5
    return preds


def test_h2o_training_flow_with_mocked_utils(monkeypatch, sample_data):
    fake_automl = types.SimpleNamespace(leader=types.SimpleNamespace(model_id="leader_1"))

    fake_module = types.SimpleNamespace(
        check_java_availability=lambda: True,
        train_h2o_model=lambda *args, **kwargs: (fake_automl, "run_abc"),
    )
    monkeypatch.setitem(sys.modules, "h2o_utils", fake_module)

    automl, run_id = _run_h2o_training_flow(sample_data, "h2o_run_test")

    assert run_id == "run_abc"
    assert automl.leader.model_id == "leader_1"


def test_h2o_training_flow_raises_without_java(monkeypatch, sample_data):
    fake_module = types.SimpleNamespace(
        check_java_availability=lambda: False,
        train_h2o_model=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "h2o_utils", fake_module)

    with pytest.raises(RuntimeError, match="Java not available"):
        _run_h2o_training_flow(sample_data, "h2o_run_test")


def test_h2o_load_and_predict_flow_with_mocked_utils(monkeypatch, sample_data):
    class FakeModel:
        pass

    fake_model = FakeModel()
    fake_module = types.SimpleNamespace(
        load_h2o_model=lambda run_id: fake_model if run_id == "run_abc" else None,
        predict_with_h2o=lambda model, df: np.array([0] * len(df)),
    )
    monkeypatch.setitem(sys.modules, "h2o_utils", fake_module)

    loaded = _run_h2o_load_flow("run_abc")
    preds = _run_h2o_predict_flow(loaded, sample_data)

    assert loaded is fake_model
    assert preds.tolist() == [0, 0, 0, 0, 0]
