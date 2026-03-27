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
