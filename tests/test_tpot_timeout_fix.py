import sys
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_tpot_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature1": rng.normal(size=80),
            "feature2": rng.normal(size=80),
            "feature3": rng.choice(["A", "B", "C"], size=80),
            "target": rng.choice([0, 1], size=80),
        }
    )


def _run_tpot_training(df: pd.DataFrame, run_name: str, **overrides):
    from tpot_utils import train_tpot_model

    params = {
        "generations": 2,
        "population_size": 5,
        "cv": 2,
        "scoring": "f1_macro",
        "max_time_mins": 2,
        "max_eval_time_mins": 1,
        "random_state": 42,
        "verbosity": 1,
        "n_jobs": 1,
        "config_dict": "TPOT light",
    }
    params.update(overrides)

    tpot, pipeline, run_id, model_info = train_tpot_model(df, "target", run_name, **params)

    assert run_id
    assert isinstance(model_info, dict)
    return tpot, pipeline, run_id, model_info, params


def test_tpot_timeout_flow_uses_expected_parameters(monkeypatch, small_tpot_df):
    captured = {}

    class FakeTPOT:
        fitted_pipeline_ = "mock_pipeline"

    def fake_train_tpot_model(df, target_column, run_name, **kwargs):
        captured["target"] = target_column
        captured["run_name"] = run_name
        captured["kwargs"] = kwargs
        return FakeTPOT(), object(), "run_tpot_1", {"problem_type": "classification", "n_features": 3}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "tpot_utils", fake_module)

    _, _, run_id, model_info, params = _run_tpot_training(small_tpot_df, "tpot_timeout_test")

    assert run_id == "run_tpot_1"
    assert model_info["problem_type"] == "classification"
    assert captured["target"] == "target"
    assert captured["run_name"] == "tpot_timeout_test"
    assert captured["kwargs"] == params


def test_tpot_high_dimensional_flow_returns_feature_metadata(monkeypatch):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        **{f"num_feature_{i}": rng.normal(size=120) for i in range(20)},
        **{f"cat_feature_{i}": rng.choice(["A", "B", "C"], size=120) for i in range(4)},
        "target": rng.choice([0, 1, 2], size=120),
    })

    class FakeTPOT:
        fitted_pipeline_ = "mock_high_dim_pipeline"

    def fake_train_tpot_model(df_in, target_column, run_name, **kwargs):
        assert target_column == "target"
        assert kwargs["max_eval_time_mins"] == 2
        return FakeTPOT(), object(), "run_tpot_hd", {"problem_type": "classification", "n_features": df_in.shape[1] - 1}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "tpot_utils", fake_module)

    tpot, _, run_id, model_info, _ = _run_tpot_training(df, "tpot_high_dim_test", max_eval_time_mins=2)

    assert run_id == "run_tpot_hd"
    assert tpot.fitted_pipeline_ == "mock_high_dim_pipeline"
    assert model_info["n_features"] == df.shape[1] - 1
