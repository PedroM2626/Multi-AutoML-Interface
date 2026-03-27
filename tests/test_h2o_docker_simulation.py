import sys
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def docker_like_h2o_df():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feature1": rng.normal(size=100),
            "feature2": rng.normal(size=100),
            "target": rng.choice([0, 1], size=100),
        }
    )
    return df


def _check_h2o_runtime_readiness(df: pd.DataFrame):
    from h2o_utils import check_java_availability, initialize_h2o, prepare_data_for_h2o

    if not check_java_availability():
        raise RuntimeError("Java not available")

    instance = initialize_h2o()
    frame, clean_data = prepare_data_for_h2o(df, "target")
    return instance, frame, clean_data


def _run_h2o_minimal_training(df: pd.DataFrame, run_name: str):
    from h2o_utils import predict_with_h2o, train_h2o_model

    automl, run_id = train_h2o_model(
        df,
        "target",
        run_name,
        max_runtime_secs=30,
        max_models=2,
        nfolds=2,
        balance_classes=True,
        seed=42,
        sort_metric="AUTO",
        exclude_algos=["DeepLearning", "GLM"],
    )

    assert hasattr(automl, "leader")
    assert automl.leader is not None

    preds = predict_with_h2o(automl.leader, df.head(5).drop(columns=["target"]))
    return run_id, preds


def test_h2o_docker_runtime_contract(monkeypatch, docker_like_h2o_df):
    fake_h2o_frame = types.SimpleNamespace(shape=(100, 3))

    def fake_prepare_data_for_h2o(df, target):
        assert target == "target"
        return fake_h2o_frame, df.copy()

    fake_module = types.SimpleNamespace(
        check_java_availability=lambda: True,
        initialize_h2o=lambda: {"cluster": "ok"},
        prepare_data_for_h2o=fake_prepare_data_for_h2o,
    )
    monkeypatch.setitem(sys.modules, "h2o_utils", fake_module)

    instance, frame, clean_data = _check_h2o_runtime_readiness(docker_like_h2o_df)

    assert instance == {"cluster": "ok"}
    assert frame is fake_h2o_frame
    assert clean_data.shape == docker_like_h2o_df.shape


def test_h2o_docker_runtime_raises_without_java(monkeypatch, docker_like_h2o_df):
    fake_module = types.SimpleNamespace(
        check_java_availability=lambda: False,
        initialize_h2o=lambda: None,
        prepare_data_for_h2o=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "h2o_utils", fake_module)

    with pytest.raises(RuntimeError, match="Java not available"):
        _check_h2o_runtime_readiness(docker_like_h2o_df)


def test_h2o_docker_minimal_training_contract(monkeypatch, docker_like_h2o_df):
    fake_leader = types.SimpleNamespace(model_id="leader_docker")
    fake_automl = types.SimpleNamespace(leader=fake_leader)

    fake_module = types.SimpleNamespace(
        train_h2o_model=lambda *args, **kwargs: (fake_automl, "run_docker_1"),
        predict_with_h2o=lambda model, data: np.array([1] * len(data)),
    )
    monkeypatch.setitem(sys.modules, "h2o_utils", fake_module)

    run_id, preds = _run_h2o_minimal_training(docker_like_h2o_df, "docker_h2o_test")

    assert run_id == "run_docker_1"
    assert preds.tolist() == [1, 1, 1, 1, 1]
