import sys
import types

import numpy as np
import pandas as pd


import pytest


@pytest.fixture
def mixed_tpot_df():
    rng = np.random.default_rng(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "feature1": rng.normal(size=n_samples),
            "feature2": rng.uniform(0, 100, size=n_samples),
            "feature3": rng.choice(["A", "B", "C", "D"], size=n_samples),
            "feature4": rng.choice(["X", "Y"], size=n_samples),
            "feature5": [f"text data {i}" for i in range(n_samples)],
            "feature6": [np.nan if i % 8 == 0 else x for i, x in enumerate(rng.choice(["cat", "dog", "bird"], size=n_samples))],
            "target": rng.choice([0, 1, 2], size=n_samples),
        }
    )


@pytest.fixture
def categorical_tpot_df():
    rng = np.random.default_rng(7)
    n_samples = 60
    return pd.DataFrame(
        {
            "cat1": rng.choice(["A", "B", "C"], size=n_samples),
            "cat2": rng.choice(["X", "Y", "Z", "W"], size=n_samples),
            "cat3": rng.choice(["red", "blue", "green"], size=n_samples),
            "target": rng.choice([0, 1], size=n_samples),
        }
    )


def _run_tpot_sparse_flow(df: pd.DataFrame, run_name: str):
    from tpot_utils import train_tpot_model

    return train_tpot_model(
        df,
        "target",
        run_name,
        generations=1,
        population_size=5,
        cv=2,
        scoring="f1_macro",
        max_time_mins=2,
        max_eval_time_mins=1,
        random_state=42,
        verbosity=1,
        n_jobs=1,
        config_dict="TPOT light",
    )


def test_tpot_sparse_mixed_data_contract(monkeypatch, mixed_tpot_df):
    class FakeTPOT:
        fitted_pipeline_ = "mock_sparse_mixed"

    def fake_train_tpot_model(df, target_column, run_name, **kwargs):
        assert target_column == "target"
        assert "feature5" in df.columns
        assert df.isnull().sum().sum() > 0
        return FakeTPOT(), object(), "run_sparse_mixed", {"problem_type": "classification"}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "tpot_utils", fake_module)

    tpot, _, run_id, model_info = _run_tpot_sparse_flow(mixed_tpot_df, "tpot_test_mixed")

    assert run_id == "run_sparse_mixed"
    assert tpot.fitted_pipeline_ == "mock_sparse_mixed"
    assert model_info["problem_type"] == "classification"


def test_tpot_sparse_categorical_only_contract(monkeypatch, categorical_tpot_df):
    class FakeTPOT:
        fitted_pipeline_ = "mock_sparse_cat"

    def fake_train_tpot_model(df, target_column, run_name, **kwargs):
        assert target_column == "target"
        assert set(df.columns) == {"cat1", "cat2", "cat3", "target"}
        return FakeTPOT(), object(), "run_sparse_cat", {"problem_type": "classification"}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "tpot_utils", fake_module)

    tpot, _, run_id, model_info = _run_tpot_sparse_flow(categorical_tpot_df, "tpot_test_cat")

    assert run_id == "run_sparse_cat"
    assert tpot.fitted_pipeline_ == "mock_sparse_cat"
    assert model_info["problem_type"] == "classification"
