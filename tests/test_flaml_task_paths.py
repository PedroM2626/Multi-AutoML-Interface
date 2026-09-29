"""The two FLAML task paths that used to fail before the hyperparameter search started.

ts_forecast asserts a forecast 'period' and reads timestamps from 'time_col'; rank forwards a
'group' argument that only the boosting learners accept and wants integer relevance grades in
contiguous query blocks. These run the real engine, so they skip when FLAML or LightGBM is
absent instead of reporting a green suite.
"""
import importlib.util

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("lightgbm") is None:
    # train_flaml_model refuses lgbm without the package (by design, see _require_learner_packages),
    # so the whole module needs it - skip rather than fail on a lean interpreter.
    pytest.skip("lightgbm is not installed", allow_module_level=True)

try:
    from src.flaml_utils import train_flaml_model
except ImportError as exc:  # pragma: no cover - environment dependent
    # flaml is importable as a namespace on the CI runner without exposing AutoML, so
    # pytest.importorskip("flaml") would not skip this module and collection would error.
    pytest.skip(f"FLAML training path unavailable: {exc}", allow_module_level=True)


def _series_frame(rows=200):
    rng = np.random.default_rng(0)
    index = pd.date_range("2021-01-01", periods=rows, freq="D")
    frame = pd.DataFrame({"date": index, "a": rng.normal(size=rows), "b": rng.normal(size=rows)})
    frame["y"] = frame["a"] * 2 + frame["b"] + np.sin(np.arange(rows) / 7.0)
    return frame


def _ranking_frame(rows=240):
    rng = np.random.default_rng(0)
    queries = rng.integers(0, 12, rows)
    frame = pd.DataFrame({
        "a": rng.normal(size=rows),
        "b": rng.normal(size=rows),
        "query": [f"q{value}" for value in queries],
    })
    frame["grade"] = np.clip((frame["a"] * 2).round().astype(int) + 2, 0, 4)
    # Deliberately shuffled: the engine adapter has to restore the per-query blocks.
    return frame.sample(frac=1, random_state=1).reset_index(drop=True)


def test_forecast_trains_with_a_time_column_and_a_period():
    model, run_id = train_flaml_model(
        train_data=_series_frame(), target="y", run_name="test_ts_forecast",
        time_budget=6, task="ts_forecast", metric="auto", estimator_list=["lgbm"],
        seed=0, n_jobs=1, time_col="date", period=7,
    )
    assert run_id
    assert model.predict(_series_frame().drop(columns=["y"])).shape == (200,)


def test_forecast_without_a_time_column_is_reported_not_crashed():
    with pytest.raises(ValueError, match="date column"):
        train_flaml_model(
            train_data=_series_frame(120), target="y", run_name="test_ts_no_time_col",
            time_budget=2, task="ts_forecast", estimator_list=["lgbm"], seed=0, n_jobs=1,
        )


def test_forecast_needs_a_horizon():
    with pytest.raises(ValueError, match="horizon"):
        train_flaml_model(
            train_data=_series_frame(120), target="y", run_name="test_ts_no_period",
            time_budget=2, task="ts_forecast", estimator_list=["lgbm"], seed=0, n_jobs=1,
            time_col="date", period=0,
        )


def test_ranking_trains_on_a_shuffled_frame_with_a_query_column():
    frame = _ranking_frame()
    model, run_id = train_flaml_model(
        train_data=frame, target="grade", run_name="test_rank",
        time_budget=6, task="rank", metric="auto", estimator_list=["lgbm"],
        seed=0, n_jobs=1, group_col="query",
    )
    assert run_id
    predictions = model.predict(frame.drop(columns=["grade"]))
    assert len(predictions) == len(frame)


def test_ranking_without_a_group_column_is_reported_not_crashed():
    with pytest.raises(ValueError, match="query/group column"):
        train_flaml_model(
            train_data=_ranking_frame(120), target="grade", run_name="test_rank_no_group",
            time_budget=2, task="rank", estimator_list=["lgbm"], seed=0, n_jobs=1,
        )


def test_ranking_rejects_a_non_integer_target():
    frame = _ranking_frame(120)
    frame["grade"] = frame["grade"] + 0.5
    with pytest.raises(ValueError, match="integer relevance grades"):
        train_flaml_model(
            train_data=frame, target="grade", run_name="test_rank_float_target",
            time_budget=2, task="rank", estimator_list=["lgbm"], seed=0, n_jobs=1,
            group_col="query",
        )
