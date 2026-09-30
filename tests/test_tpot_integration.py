import sys
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tpot_classification_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature1": rng.normal(size=120),
            "feature2": rng.normal(size=120),
            "feature3": rng.choice(["A", "B", "C"], size=120),
            "feature4": rng.uniform(0, 100, size=120),
            "target": rng.choice([0, 1], size=120),
        }
    )


@pytest.fixture
def tpot_regression_df():
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "feature1": rng.normal(size=100),
            "feature2": rng.normal(size=100),
            "feature3": rng.uniform(0, 50, size=100),
            "target": rng.normal(loc=5, scale=2, size=100),
        }
    )


@pytest.fixture
def tpot_text_df():
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "text_feature": ["positive review" if i % 2 == 0 else "negative review" for i in range(90)],
            "numeric_feature": rng.normal(size=90),
            "target": rng.choice([0, 1], size=90),
        }
    )


def _train_tpot(df: pd.DataFrame, run_name: str, **kwargs):
    from src.tpot_utils import train_tpot_model

    return train_tpot_model(df, "target", run_name, **kwargs)


def _detect_problem_type(y: pd.Series):
    from src.tpot_utils import detect_problem_type

    return detect_problem_type(y)


def _build_feature_pipeline(df: pd.DataFrame):
    from src.tpot_utils import create_feature_pipeline

    return create_feature_pipeline(df, "target", text_columns=["text_col"])


def test_tpot_classification_training_contract(monkeypatch, tpot_classification_df):
    captured = {}

    class FakeTPOT:
        fitted_pipeline_ = "mock_cls_pipeline"

    def fake_train_tpot_model(df, target_column, run_name, **kwargs):
        captured["target"] = target_column
        captured["run_name"] = run_name
        captured["kwargs"] = kwargs
        return FakeTPOT(), object(), "run_cls_1", {"problem_type": "classification", "accuracy": 0.9, "f1_macro": 0.88}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "src.tpot_utils", fake_module)

    tpot, _, run_id, model_info = _train_tpot(
        tpot_classification_df,
        "tpot_test_classification",
        generations=2,
        population_size=10,
        cv=3,
        scoring="f1_macro",
        max_time_mins=5,
        max_eval_time_mins=2,
        random_state=42,
        verbosity=1,
        n_jobs=1,
        config_dict="TPOT light",
    )

    assert run_id == "run_cls_1"
    assert model_info["problem_type"] == "classification"
    assert "accuracy" in model_info
    assert "f1_macro" in model_info
    assert tpot.fitted_pipeline_ == "mock_cls_pipeline"
    assert captured["target"] == "target"
    assert captured["run_name"] == "tpot_test_classification"


def test_tpot_regression_training_contract(monkeypatch, tpot_regression_df):
    class FakeTPOT:
        fitted_pipeline_ = "mock_reg_pipeline"

    def fake_train_tpot_model(df, target_column, run_name, **kwargs):
        return FakeTPOT(), object(), "run_reg_1", {"problem_type": "regression", "rmse": 1.5, "r2": 0.72}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "src.tpot_utils", fake_module)

    tpot, _, run_id, model_info = _train_tpot(
        tpot_regression_df,
        "tpot_test_regression",
        generations=2,
        population_size=10,
        cv=3,
        scoring="neg_mean_squared_error",
        max_time_mins=5,
        max_eval_time_mins=2,
        random_state=42,
        verbosity=1,
        n_jobs=1,
        config_dict="TPOT light",
    )

    assert run_id == "run_reg_1"
    assert model_info["problem_type"] == "regression"
    assert "rmse" in model_info
    assert "r2" in model_info
    assert tpot.fitted_pipeline_ == "mock_reg_pipeline"


def test_tpot_text_training_contract(monkeypatch, tpot_text_df):
    class FakeTPOT:
        fitted_pipeline_ = "mock_text_pipeline"

    def fake_train_tpot_model(df, target_column, run_name, **kwargs):
        return FakeTPOT(), object(), "run_text_1", {"problem_type": "classification", "text_columns": ["text_feature"]}

    fake_module = types.SimpleNamespace(train_tpot_model=fake_train_tpot_model)
    monkeypatch.setitem(sys.modules, "src.tpot_utils", fake_module)

    tpot, _, run_id, model_info = _train_tpot(
        tpot_text_df,
        "tpot_test_text",
        generations=2,
        population_size=10,
        cv=3,
        scoring="f1_macro",
        max_time_mins=5,
        max_eval_time_mins=2,
        random_state=42,
        verbosity=1,
        n_jobs=1,
        config_dict="TPOT sparse",
    )

    assert run_id == "run_text_1"
    assert model_info["text_columns"] == ["text_feature"]
    assert tpot.fitted_pipeline_ == "mock_text_pipeline"


def test_problem_type_detection_contract(monkeypatch):
    def fake_detect_problem_type(y):
        if str(y.dtype) == "object":
            return "classification"
        return "regression"

    fake_module = types.SimpleNamespace(detect_problem_type=fake_detect_problem_type)
    monkeypatch.setitem(sys.modules, "src.tpot_utils", fake_module)

    assert _detect_problem_type(pd.Series(["A", "B", "A"])) == "classification"
    assert _detect_problem_type(pd.Series([1.2, 2.1, 3.4])) == "regression"


def test_feature_pipeline_contract(monkeypatch):
    df = pd.DataFrame(
        {
            "text_col": ["hello world", "test data", "more text"],
            "num_col1": [1.0, 2.0, 3.0],
            "num_col2": [4, 5, 6],
            "cat_col": ["A", "B", "A"],
            "target": [0, 1, 0],
        }
    )

    def fake_create_feature_pipeline(df_in, target_col, text_columns=None):
        assert target_col == "target"
        return object(), ["text_col"], ["cat_col"], ["num_col1", "num_col2"]

    fake_module = types.SimpleNamespace(create_feature_pipeline=fake_create_feature_pipeline)
    monkeypatch.setitem(sys.modules, "src.tpot_utils", fake_module)

    _, text_cols, cat_cols, num_cols = _build_feature_pipeline(df)

    assert text_cols == ["text_col"]
    assert "cat_col" in cat_cols
    assert "num_col1" in num_cols and "num_col2" in num_cols


class _TpotOneApiEstimator:
    """Stand-in for tpot 1.x, whose estimator signature dropped the 0.11 knobs."""

    def __init__(self, *, cv=None, max_time_mins=None, max_eval_time_mins=None,
                 random_state=None, verbose=None, n_jobs=None):
        self.received = {
            "cv": cv, "max_time_mins": max_time_mins, "max_eval_time_mins": max_eval_time_mins,
            "random_state": random_state, "verbose": verbose, "n_jobs": n_jobs,
        }


def _load_tpot_utils_with_stub(monkeypatch, estimator):
    import importlib.util

    stub = types.ModuleType("tpot")
    stub.TPOTClassifier = estimator
    stub.TPOTRegressor = estimator
    monkeypatch.setitem(sys.modules, "tpot", stub)

    spec = importlib.util.spec_from_file_location("tpot_utils_under_test", "src/tpot_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_estimator_kwargs_follow_the_installed_tpot_api(monkeypatch, caplog):
    """The UI sends generations/population_size/scoring/verbosity/config_dict, which tpot 1.x
    rejects deep inside the search - after the run was already reported as started."""
    module = _load_tpot_utils_with_stub(monkeypatch, _TpotOneApiEstimator)

    built = module._tpot_estimator(
        _TpotOneApiEstimator, generations=5, population_size=20, cv=3, scoring="f1",
        max_time_mins=2, max_eval_time_mins=1, random_state=7, verbosity=2, n_jobs=1,
        config_dict="TPOT sparse",
    )

    assert built.received == {
        "cv": 3, "max_time_mins": 2, "max_eval_time_mins": 1,
        "random_state": 7, "verbose": 1, "n_jobs": 1,
    }
    assert "scoring" in caplog.text and "generations" in caplog.text


def test_legacy_tpot_api_still_receives_its_own_kwargs(monkeypatch):
    class _TpotLegacyEstimator:
        def __init__(self, *, generations=None, population_size=None, cv=None, scoring=None,
                     max_time_mins=None, max_eval_time_mins=None, random_state=None,
                     verbosity=None, n_jobs=None, config_dict=None):
            self.received = locals()
            self.received.pop("self", None)

    module = _load_tpot_utils_with_stub(monkeypatch, _TpotLegacyEstimator)

    built = module._tpot_estimator(
        _TpotLegacyEstimator, generations=4, population_size=8, cv=2, scoring="f1",
        max_time_mins=1, max_eval_time_mins=1, random_state=0, verbosity=1, n_jobs=1,
        config_dict="TPOT light",
    )

    assert built.received["generations"] == 4
    assert built.received["config_dict"] == "TPOT light"


@pytest.mark.parametrize("values,expected", [
    ([0, 1, 0, 1], "classification"),
    ([0.0, 1.0, 2.0], "classification"),
    ([0.5, 1.5, 2.5], "regression"),
    ([0, 1, None], "classification"),
    (["a", "b", "a"], "classification"),
])
def test_problem_type_reads_the_whole_column(monkeypatch, values, expected):
    """The old test looped `all(y % 1 == 0 for val in ...)`, which boolean-casts a Series on
    every step, so every TPOT row died before the estimator was even built."""
    module = _load_tpot_utils_with_stub(monkeypatch, _TpotOneApiEstimator)

    assert module.detect_problem_type(pd.Series(values)) == expected
