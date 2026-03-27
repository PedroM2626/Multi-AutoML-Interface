import sys
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def problematic_h2o_df():
    rng = np.random.default_rng(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "feature1": [1.0] * n_samples,
            "feature2": rng.normal(size=n_samples) * 0.001,
            "feature3": ["A"] * n_samples,
            "target": rng.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        }
    )


def _run_empty_leaderboard_flow(df: pd.DataFrame, run_name: str):
    from h2o_utils import check_java_availability, train_h2o_model

    if not check_java_availability():
        raise RuntimeError("Java not available")

    automl, run_id = train_h2o_model(
        df,
        "target",
        run_name,
        max_runtime_secs=10,
        max_models=1,
        nfolds=2,
        balance_classes=True,
        seed=42,
        sort_metric="AUTO",
        exclude_algos=["DeepLearning", "GBM", "DRF"],
    )
    return automl, run_id


def test_empty_leaderboard_flow_without_leader(monkeypatch, problematic_h2o_df):
    fake_automl = types.SimpleNamespace(leader=None)

    fake_module = types.SimpleNamespace(
        check_java_availability=lambda: True,
        train_h2o_model=lambda *args, **kwargs: (fake_automl, "run_empty_1"),
    )
    monkeypatch.setitem(sys.modules, "h2o_utils", fake_module)

    automl, run_id = _run_empty_leaderboard_flow(problematic_h2o_df, "empty_test")

    assert run_id == "run_empty_1"
    assert hasattr(automl, "leader")
    assert automl.leader is None


def test_empty_leaderboard_flow_with_leader(monkeypatch, problematic_h2o_df):
    fake_leader = types.SimpleNamespace(model_id="leader_ok")
    fake_automl = types.SimpleNamespace(leader=fake_leader)

    fake_module = types.SimpleNamespace(
        check_java_availability=lambda: True,
        train_h2o_model=lambda *args, **kwargs: (fake_automl, "run_empty_2"),
    )
    monkeypatch.setitem(sys.modules, "h2o_utils", fake_module)

    automl, run_id = _run_empty_leaderboard_flow(problematic_h2o_df, "empty_test")

    assert run_id == "run_empty_2"
    assert automl.leader.model_id == "leader_ok"


def test_empty_leaderboard_raises_when_java_unavailable(monkeypatch, problematic_h2o_df):
    fake_module = types.SimpleNamespace(
        check_java_availability=lambda: False,
        train_h2o_model=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "h2o_utils", fake_module)

    with pytest.raises(RuntimeError, match="Java not available"):
        _run_empty_leaderboard_flow(problematic_h2o_df, "empty_test")
