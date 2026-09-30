"""H2O cluster lifecycle without a JVM.

These exercise src.h2o_utils against a fake h2o module, so they run in the plain suite. What they
pin down is the part a real run cannot show: that no operation leaves the Java cluster running,
that one operation cannot be interrupted by another session's shutdown, and that a handle is all
the UI keeps after training.
"""

import sys
import threading
import types

import numpy as np
import pandas as pd
import pytest

from src.h2o_utils import H2OSessionModel


class FakePredictions:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, key):
        assert key == "predict"
        return self

    def as_data_frame(self):
        return pd.DataFrame({"predict": self._values})


class FakeModel:
    def __init__(self, model_id, values, recorder):
        self.model_id = model_id
        self._values = values
        self._recorder = recorder

    def predict(self, frame):
        self._recorder["predict_with_cluster"] = self._recorder["cluster"] is not None
        return FakePredictions(self._values[: len(frame)])


@pytest.fixture
def h2o_utils(monkeypatch):
    from src import h2o_utils as module

    monkeypatch.setattr(module, "check_java_availability", lambda: True)
    # These tests deliberately let two operations contend for the cluster; waiting five seconds
    # per poll would make the suite crawl.
    monkeypatch.setattr(module, "_LOCK_POLL_SECONDS", 0.02)
    return module


@pytest.fixture
def fake_h2o(monkeypatch):
    recorder = {
        "init": 0,
        "shutdown": 0,
        "shutdown_prompt": None,
        "init_kwargs": [],
        "cluster": None,
        "alive": 0,
        "peak_alive": 0,
        "loaded_from": None,
        "load_with_cluster": None,
        "predict_with_cluster": None,
        "models": {},
    }

    class FakeCluster:
        def shutdown(self, prompt=True):
            recorder["shutdown_prompt"] = prompt
            recorder["shutdown"] += 1
            recorder["cluster"] = None
            recorder["alive"] -= 1

    def init(**kwargs):
        recorder["init"] += 1
        recorder["init_kwargs"].append(kwargs)
        recorder["cluster"] = FakeCluster()
        # A JVM that exists while another operation is running is the bug this whole module
        # prevents, so count how many clusters are up at the same time.
        recorder["alive"] += 1
        recorder["peak_alive"] = max(recorder["peak_alive"], recorder["alive"])

    def load_model(path):
        # Loading needs a cluster: it is what makes the order inside predict_with_h2o testable.
        if recorder["cluster"] is None:
            raise RuntimeError("H2O cluster is not running.")
        recorder["loaded_from"] = path
        recorder["load_with_cluster"] = True
        return recorder["models"][path]

    fake = types.SimpleNamespace(
        init=init,
        cluster=lambda: recorder["cluster"],
        H2OFrame=lambda frame: frame,
        load_model=load_model,
    )
    monkeypatch.setitem(sys.modules, "h2o", fake)
    return recorder


def store_run_model(fake_h2o, tmp_path, monkeypatch, h2o_utils, name, values):
    """Point mlflow at a directory holding an h2o.save_model archive for this run."""
    archive = tmp_path / name
    archive.write_bytes(b"archive")
    fake_h2o["models"][str(archive)] = FakeModel(name, values, fake_h2o)
    monkeypatch.setattr(
        h2o_utils,
        "mlflow",
        types.SimpleNamespace(
            artifacts=types.SimpleNamespace(
                download_artifacts=lambda run_id, artifact_path: str(tmp_path)
            )
        ),
    )
    return archive


def test_one_cluster_per_operation_even_when_the_body_nests_it(h2o_utils, fake_h2o):
    with h2o_utils.h2o_cluster():
        with h2o_utils.h2o_cluster():
            assert fake_h2o["init"] == 1
            assert fake_h2o["cluster"] is not None

    assert fake_h2o["init"] == 1
    assert fake_h2o["shutdown"] == 1
    assert fake_h2o["cluster"] is None


def test_a_failed_operation_still_releases_the_cluster(h2o_utils, fake_h2o):
    with pytest.raises(RuntimeError, match="training died"):
        with h2o_utils.h2o_cluster():
            raise RuntimeError("training died")

    assert fake_h2o["shutdown"] == 1
    assert fake_h2o["cluster"] is None
    # And the lock came back with it: a crash cannot block the next session.
    with h2o_utils.h2o_cluster():
        assert fake_h2o["init"] == 2


def test_a_cluster_is_exclusive_until_the_operation_ends(h2o_utils, fake_h2o):
    second_in = threading.Event()
    hold = threading.Semaphore(0)

    def worker():
        with h2o_utils.h2o_cluster():
            second_in.set()
            hold.acquire()

    thread = threading.Thread(target=worker, daemon=True)
    with h2o_utils.h2o_cluster():
        thread.start()
        # Blocked outside, with no JVM of its own: it cannot shut this operation's cluster down,
        # which is exactly how two sessions used to interfere.
        assert not second_in.wait(timeout=0.5)
        assert fake_h2o["init"] == 1

    hold.release()
    thread.join(timeout=5)
    assert second_in.is_set()
    assert not thread.is_alive()
    assert fake_h2o["init"] == 2
    assert fake_h2o["shutdown"] == 2
    assert fake_h2o["peak_alive"] == 1


def test_waiting_for_the_cluster_honours_the_stop_event(h2o_utils, fake_h2o):
    stop_event = threading.Event()
    stop_event.set()
    errors = []

    def worker():
        try:
            with h2o_utils.h2o_cluster(stop_event):
                pass
        except BaseException as exc:
            errors.append(exc)

    with h2o_utils.h2o_cluster():
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=5)

    assert errors and isinstance(errors[0], StopIteration)
    # The cancelled operation took nothing with it: its own cluster never started, and the one
    # still in use stayed up.
    assert fake_h2o["init"] == 1
    assert fake_h2o["shutdown"] == 1


def test_cleanup_is_quiet_when_no_cluster_is_connected(h2o_utils, fake_h2o):
    h2o_utils.cleanup_h2o()
    assert fake_h2o["shutdown"] == 0


def test_a_cluster_that_never_started_does_not_hold_the_lock(h2o_utils, fake_h2o, monkeypatch):
    real_init = sys.modules["h2o"].init
    attempts = []

    def init_fails_once(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise OSError("java is missing")
        real_init(**kwargs)

    monkeypatch.setattr(sys.modules["h2o"], "init", init_fails_once)

    with pytest.raises(OSError, match="java is missing"):
        with h2o_utils.h2o_cluster():
            pass

    # The failed operation gave the lock straight back, so the next session is not queued behind
    # a cluster that does not exist.
    with h2o_utils.h2o_cluster():
        assert fake_h2o["init"] == 1
    assert fake_h2o["shutdown"] == 1


def test_init_asks_h2o_for_a_private_cluster(h2o_utils, fake_h2o):
    with h2o_utils.h2o_cluster():
        kwargs = fake_h2o["init_kwargs"][0]

    assert fake_h2o["shutdown"] == 1
    assert kwargs["ip"] == "127.0.0.1"
    assert kwargs["port"] != 54321
    assert kwargs["max_mem_size"] == h2o_utils.H2O_CLUSTER_MEM_SIZE


def test_training_returns_a_handle_and_leaves_nothing_running(
    h2o_utils, fake_h2o, monkeypatch
):
    trained = types.SimpleNamespace(leader=types.SimpleNamespace(model_id="GLM_1"))
    monkeypatch.setattr(
        h2o_utils, "_train_h2o_model", lambda **kwargs: (trained, "run_abc")
    )

    handle, run_id = h2o_utils.train_h2o_model(
        train_data=pd.DataFrame({"a": [1, 2, 3], "target": ["x", "y", "x"]}),
        target="target",
        run_name="lifecycle_test",
    )

    assert run_id == "run_abc"
    assert isinstance(handle, H2OSessionModel)
    assert handle.run_id == "run_abc"
    # The UI holds this object across reruns: anything else would pin a JVM to the session.
    assert not hasattr(handle, "leader")
    assert fake_h2o["init"] == 1
    assert fake_h2o["shutdown"] == 1
    assert fake_h2o["cluster"] is None


def test_a_handle_resolves_to_the_model_saved_by_the_run(
    h2o_utils, fake_h2o, monkeypatch, tmp_path
):
    # h2o.save_model writes an archive named after the model id with no extension; older
    # releases ended it in .zip. Both have to load - it is what makes a handle useful.
    archive = store_run_model(
        fake_h2o, tmp_path, monkeypatch, h2o_utils, "GLM_1_AutoML_1", ["Class0", "Class1"]
    )

    with h2o_utils.h2o_cluster():
        resolved = h2o_utils.resolve_h2o_model(H2OSessionModel("run_abc"))

    assert fake_h2o["loaded_from"] == str(archive)
    assert fake_h2o["load_with_cluster"] is True
    assert resolved.model_id == "GLM_1_AutoML_1"


def test_a_zip_archive_wins_over_an_extensionless_sibling(
    h2o_utils, fake_h2o, monkeypatch, tmp_path
):
    archive = store_run_model(fake_h2o, tmp_path, monkeypatch, h2o_utils, "GLM_1.zip", ["Class0"])
    (tmp_path / "readme.txt").write_text("consumption sample, not a model")

    with h2o_utils.h2o_cluster():
        resolved = h2o_utils.resolve_h2o_model(H2OSessionModel("run"))

    assert fake_h2o["loaded_from"] == str(archive)
    assert resolved.model_id == "GLM_1.zip"


def test_the_leaderboard_is_read_from_the_run_without_starting_java(
    h2o_utils, fake_h2o, monkeypatch, tmp_path
):
    csv = tmp_path / f"{h2o_utils.H2O_LEADERBOARD_PREFIX}some_run.csv"
    csv.write_text(
        "model_id,auc,logloss\n"
        "GLM_1_AutoML_1_20260930_203357,0.989,0.126\n"
        "DRF_1_AutoML_1_20260930_203357,0.944,0.341\n"
    )
    stub_mlflow_artifacts(monkeypatch, h2o_utils, [(csv.name, str(csv))])

    frame, leader_id = h2o_utils.h2o_run_leaderboard("run_abc")

    # The Inspector redraws this on every rerun of every session: no cluster, no 2 GB heap.
    assert fake_h2o["init"] == 0
    assert list(frame["model_id"]) == [
        "GLM_1_AutoML_1_20260930_203357",
        "DRF_1_AutoML_1_20260930_203357",
    ]
    assert leader_id == "GLM_1_AutoML_1_20260930_203357"


def test_a_run_that_trained_no_model_says_so_instead_of_showing_an_empty_table(
    h2o_utils, fake_h2o, monkeypatch, tmp_path
):
    # _train_h2o_model logs no_model_<run>.txt when the leaderboard came back empty.
    note = tmp_path / "no_model_broken_run.txt"
    note.write_text("No models were trained during this run.")
    stub_mlflow_artifacts(monkeypatch, h2o_utils, [(note.name, str(note))])

    with pytest.raises(FileNotFoundError, match="trained no model"):
        h2o_utils.h2o_run_leaderboard("run_abc")


def test_the_training_path_names_the_leaderboard_with_the_same_constant(h2o_utils):
    import inspect

    # Drift here is silent: the writer would log a file the reader never looks for, and the
    # Inspector would report "trained no model" for a run that has a full leaderboard.
    assert "h2o_leaderboard_" not in inspect.getsource(h2o_utils._train_h2o_model)


def stub_mlflow_artifacts(monkeypatch, h2o_utils, files):
    """mlflow.artifacts.list_artifacts / download_artifacts over a local directory."""
    by_name = dict(files)

    monkeypatch.setattr(
        h2o_utils,
        "mlflow",
        types.SimpleNamespace(
            artifacts=types.SimpleNamespace(
                list_artifacts=lambda run_id, artifact_path: [
                    types.SimpleNamespace(path=name) for name in by_name
                ],
                download_artifacts=lambda run_id, artifact_path: by_name[artifact_path],
            )
        ),
    )


def test_a_live_model_is_used_as_it_is(h2o_utils, fake_h2o):
    live = types.SimpleNamespace(model_id="DRF_1")

    with h2o_utils.h2o_cluster():
        assert h2o_utils.resolve_h2o_model(live) is live

    assert fake_h2o["loaded_from"] is None


def test_predicting_from_a_handle_releases_the_cluster_on_the_way_out(
    h2o_utils, fake_h2o, monkeypatch, tmp_path
):
    store_run_model(
        fake_h2o,
        tmp_path,
        monkeypatch,
        h2o_utils,
        "GBM_1_AutoML_2",
        ["Class0", "Class1", "Class0"],
    )

    predictions = h2o_utils.predict_with_h2o(
        H2OSessionModel("run"), pd.DataFrame({"feature1": [0.1, 0.2, 0.3]})
    )

    assert predictions.tolist() == ["Class0", "Class1", "Class0"]
    assert fake_h2o["predict_with_cluster"] is True
    assert fake_h2o["shutdown"] == 1
    assert fake_h2o["shutdown_prompt"] is False
    assert fake_h2o["cluster"] is None
    # Materialised before the JVM went away, so the caller never touches a dead frame.
    assert isinstance(predictions, np.ndarray)


def test_a_training_failure_inside_the_cluster_still_shuts_it_down(
    h2o_utils, fake_h2o, monkeypatch
):
    def explode(**kwargs):
        raise ValueError("no models could be trained")

    monkeypatch.setattr(h2o_utils, "_train_h2o_model", explode)

    with pytest.raises(ValueError, match="no models"):
        h2o_utils.train_h2o_model(
            train_data=pd.DataFrame({"a": [1, 2], "target": ["x", "y"]}),
            target="target",
            run_name="broken",
        )

    assert fake_h2o["shutdown"] == 1
    assert fake_h2o["cluster"] is None
