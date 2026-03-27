import sys
import types

import numpy as np
import pandas as pd


import pytest


@pytest.fixture
def nan_tpot_df():
    rng = np.random.default_rng(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "feature1": rng.normal(size=n_samples),
            "feature2": [np.nan if i % 10 == 0 else float(x) for i, x in enumerate(rng.normal(size=n_samples))],
            "feature3": ["A" if i % 3 == 0 else "B" if i % 3 == 1 else np.nan for i in range(n_samples)],
            "feature4": ["text data" if i % 5 == 0 else np.nan for i in range(n_samples)],
            "target": rng.choice([0, 1], size=n_samples),
        }
    )


def _run_tpot_nan_flow(df: pd.DataFrame, run_name: str):
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


def test_tpot_nan_data_training_contract(monkeypatch, nan_tpot_df):
    class FakeTPOT:
        fitted_pipeline_ = "mock_nan_pipeline"

    def fake_train_tpot_model(df, target_column, run_name, **kwargs):
        assert target_column == "target"
        assert df.isnull().sum().sum() > 0
        return FakeTPOT(), object(), "run_nan_1", {"problem_type": "classification"}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "tpot_utils", fake_module)

    tpot, _, run_id, model_info = _run_tpot_nan_flow(nan_tpot_df, "tpot_test_nan")

    assert run_id == "run_nan_1"
    assert tpot.fitted_pipeline_ == "mock_nan_pipeline"
    assert model_info["problem_type"] == "classification"
