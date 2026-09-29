"""Guards for the two PyCaret defects that made four of its five catalog rows fail.

These import src.pycaret_utils only - the engine itself is loaded lazily inside the functions -
so they run in the PR gate. The end-to-end case needs pycaret and skips without it.
"""
import importlib.util
import threading

import pandas as pd
import pytest

from src import pycaret_utils
from src.pycaret_utils import _ts_include_models, run_pycaret_experiment


def _features(rows=20):
    return pd.DataFrame({"a": range(rows), "b": [float(i) for i in range(rows)]})


def test_exogenous_columns_narrow_the_forecaster_list():
    """PyCaret's time series module only keeps the pmdarima family available when the frame
    carries features next to the target; asking for the others raises before training."""
    assert _ts_include_models(_features().assign(value=1.0), "value") == ["arima", "auto_arima"]


def test_univariate_frame_keeps_the_sktime_forecasters():
    univariate = pd.DataFrame({"value": [float(i) for i in range(20)]})
    assert _ts_include_models(univariate, "value") == ["naive", "snaive", "arima", "ets"]


def test_an_experiment_queued_behind_another_is_still_cancellable(monkeypatch):
    """The engine holds a process-global experiment, so runs queue; queueing may not swallow the
    cancel button."""
    monkeypatch.setattr(pycaret_utils, "_LOCK_POLL_SECONDS", 0.05)
    assert pycaret_utils._EXPERIMENT_LOCK.acquire()
    started = []
    monkeypatch.setattr(
        pycaret_utils, "_run_pycaret_experiment",
        lambda **kwargs: started.append(kwargs["run_name"]),
    )
    stop_event = threading.Event()
    stop_event.set()
    try:
        with pytest.raises(StopIteration, match="waiting for the PyCaret slot"):
            run_pycaret_experiment(
                train_df=_features(), target_col="a", run_name="queued",
                time_limit=1, log_queue=None, stop_event=stop_event,
            )
    finally:
        pycaret_utils._EXPERIMENT_LOCK.release()
    assert started == []


def test_the_date_column_moves_into_the_index_for_the_time_series_setup():
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "value": [1.0, 2.0, 3.0]})

    prepared = pycaret_utils._as_timestamp_index(frame, "date")

    assert list(prepared.columns) == ["value"]
    assert isinstance(prepared.index, pd.DatetimeIndex)
    # Already-indexed frames (the Tabular route, where the processor dropped the column) pass by.
    assert list(pycaret_utils._as_timestamp_index(prepared, "date").columns) == ["value"]


@pytest.mark.skipif(importlib.util.find_spec("pycaret") is None, reason="pycaret not installed")
def test_clustering_setup_does_not_receive_a_fold_argument(tmp_path, monkeypatch):
    """pycaret.clustering.setup() has no `fold`; passing it raised TypeError and killed both
    unsupervised rows."""
    import os
    import tempfile

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///" + tempfile.mkdtemp().replace("\\", "/"))
    result = run_pycaret_experiment(
        train_df=_features(120).assign(c=[float(i % 3) for i in range(120)]),
        target_col=None, run_name="test_clustering", time_limit=15,
        log_queue=None, task_type="Clustering", n_jobs=1,
    )
    assert result["success"] is True
