import sys
import types

import numpy as np
import pandas as pd


import pytest


@pytest.fixture
def high_dim_df():
    rng = np.random.default_rng(42)
    n_samples = 200

    base = {
        **{f"num_feature_{i}": rng.normal(size=n_samples) for i in range(30)},
        **{f"cat_feature_{i}": rng.choice(["A", "B", "C"], size=n_samples) for i in range(5)},
    }
    for i in range(3):
        base[f"text_feature_{i}"] = [
            " ".join(rng.choice(["word1", "word2", "word3", "word4"], size=6))
            for _ in range(n_samples)
        ]

    base["target"] = rng.choice([0, 1], size=n_samples)
    return pd.DataFrame(base)


@pytest.fixture
def large_df():
    rng = np.random.default_rng(7)
    n_samples = 1000
    df = pd.DataFrame({f"feature_{i}": rng.normal(size=n_samples) for i in range(20)})
    for i in range(3):
        df[f"cat_feature_{i}"] = rng.choice(["A", "B", "C"], size=n_samples)
    df["target"] = rng.choice([0, 1], size=n_samples)
    return df


def _run_tpot_large_train(df: pd.DataFrame, run_name: str, **kwargs):
    from tpot_utils import train_tpot_model

    return train_tpot_model(df, "target", run_name, **kwargs)


def test_tpot_high_dimensional_contract(monkeypatch, high_dim_df):
    class FakeTPOT:
        fitted_pipeline_ = "mock_high_dim"

    def fake_train_tpot_model(df, target_column, run_name, **kwargs):
        assert target_column == "target"
        assert kwargs["config_dict"] in ("TPOT sparse", "TPOT light")
        return FakeTPOT(), object(), "run_high_dim", {"n_features": df.shape[1] - 1, "problem_type": "classification"}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "tpot_utils", fake_module)

    tpot, _, run_id, model_info = _run_tpot_large_train(
        high_dim_df,
        "tpot_test_high_dim",
        generations=5,
        population_size=20,
        cv=3,
        scoring="f1_macro",
        max_time_mins=5,
        max_eval_time_mins=2,
        random_state=42,
        verbosity=1,
        n_jobs=1,
        config_dict="TPOT sparse",
    )

    assert run_id == "run_high_dim"
    assert tpot.fitted_pipeline_ == "mock_high_dim"
    assert model_info["n_features"] == high_dim_df.shape[1] - 1


def test_tpot_large_dataset_contract(monkeypatch, large_df):
    class FakeTPOT:
        fitted_pipeline_ = "mock_large_ds"

    def fake_train_tpot_model(df, target_column, run_name, **kwargs):
        assert kwargs["max_time_mins"] == 3
        return FakeTPOT(), object(), "run_large_ds", {"n_samples": len(df), "problem_type": "classification"}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "tpot_utils", fake_module)

    tpot, _, run_id, model_info = _run_tpot_large_train(
        large_df,
        "tpot_test_large",
        generations=3,
        population_size=10,
        cv=5,
        scoring="f1_macro",
        max_time_mins=3,
        max_eval_time_mins=1,
        random_state=42,
        verbosity=1,
        n_jobs=1,
        config_dict="TPOT light",
    )

    assert run_id == "run_large_ds"
    assert tpot.fitted_pipeline_ == "mock_large_ds"
    assert model_info["n_samples"] == len(large_df)


def test_tpot_parameter_adjustment_contract(monkeypatch, high_dim_df):
    captured = {}

    class FakeTPOT:
        fitted_pipeline_ = "mock_adjusted"

    def fake_train_tpot_model(df, target_column, run_name, **kwargs):
        captured.update(kwargs)
        return FakeTPOT(), object(), "run_adjust", {"problem_type": "classification", "n_features": df.shape[1] - 1}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "tpot_utils", fake_module)

    tpot, _, run_id, model_info = _run_tpot_large_train(
        high_dim_df,
        "tpot_test_adjust",
        generations=10,
        population_size=50,
        cv=5,
        scoring="f1_macro",
        max_time_mins=10,
        max_eval_time_mins=5,
        random_state=42,
        verbosity=1,
        n_jobs=1,
        config_dict="TPOT sparse",
    )

    assert run_id == "run_adjust"
    assert tpot.fitted_pipeline_ == "mock_adjusted"
    assert model_info["n_features"] == high_dim_df.shape[1] - 1
    assert captured["generations"] == 10
    assert captured["population_size"] == 50
