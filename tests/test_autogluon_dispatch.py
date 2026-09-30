from types import ModuleType, SimpleNamespace
import os
import sys

import pandas as pd
import pytest

from src import autogluon_utils
from src.code_gen_utils import generate_consumption_code


class _FakePredictor:
    @classmethod
    def load(cls, local_path):
        return {"loaded_from": local_path, "predictor": cls.__name__}


def _patch_mlflow_run(monkeypatch, data_category, task_type="Classification"):
    fake_run = SimpleNamespace(data=SimpleNamespace(params={"data_category": data_category, "task_type": task_type}))
    fake_client = SimpleNamespace(get_run=lambda run_id: fake_run)
    monkeypatch.setattr(autogluon_utils.mlflow.tracking, "MlflowClient", lambda: fake_client)
    monkeypatch.setattr(autogluon_utils.mlflow.artifacts, "download_artifacts", lambda **kwargs: "/tmp/model")


def test_autogluon_loader_uses_multimodal_predictor(monkeypatch):
    autogluon_pkg = ModuleType("autogluon")
    multimodal_mod = ModuleType("autogluon.multimodal")
    tabular_mod = ModuleType("autogluon.tabular")
    multimodal_mod.MultiModalPredictor = _FakePredictor
    tabular_mod.TabularPredictor = _FakePredictor
    autogluon_pkg.multimodal = multimodal_mod
    autogluon_pkg.tabular = tabular_mod
    monkeypatch.setitem(sys.modules, "autogluon", autogluon_pkg)
    monkeypatch.setitem(sys.modules, "autogluon.multimodal", multimodal_mod)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular_mod)
    _patch_mlflow_run(monkeypatch, data_category="Multimodal", task_type="Classification")

    predictor = autogluon_utils.load_model_from_mlflow("run-1")

    assert predictor["predictor"] == "_FakePredictor"


def test_autogluon_loader_uses_tabular_predictor(monkeypatch):
    autogluon_pkg = ModuleType("autogluon")
    multimodal_mod = ModuleType("autogluon.multimodal")
    tabular_mod = ModuleType("autogluon.tabular")
    multimodal_mod.MultiModalPredictor = _FakePredictor
    tabular_mod.TabularPredictor = _FakePredictor
    autogluon_pkg.multimodal = multimodal_mod
    autogluon_pkg.tabular = tabular_mod
    monkeypatch.setitem(sys.modules, "autogluon", autogluon_pkg)
    monkeypatch.setitem(sys.modules, "autogluon.multimodal", multimodal_mod)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular_mod)
    _patch_mlflow_run(monkeypatch, data_category="Tabular", task_type="Classification")

    predictor = autogluon_utils.load_model_from_mlflow("run-2")

    assert predictor["predictor"] == "_FakePredictor"


def test_codegen_switches_autogluon_loader_for_multimodal(monkeypatch):
    fake_run = SimpleNamespace(data=SimpleNamespace(params={"data_category": "Multimodal", "task_type": "Classification"}))
    fake_client = SimpleNamespace(get_run=lambda run_id: fake_run)
    monkeypatch.setattr("src.code_gen_utils.mlflow.tracking.MlflowClient", lambda: fake_client)

    code = generate_consumption_code("autogluon", "run-3", "target")

    assert "MultiModalPredictor.load" in code


def test_cv_multilabel_trains_one_predictor_per_label_column(monkeypatch, tmp_path):
    """The CV upload stores an annotations CSV inside the image folder when one is given, and that
    table - image names plus one 0/1 column per label - is the only shape that can carry a
    multi-label image target, because a folder name holds exactly one class. AutoGluon's
    MultiModalPredictor has no multilabel problem type, so the row trains one predictor per label."""
    calls = []

    class _RecordingPredictor:
        def __init__(self, label=None, problem_type=None, path=None):
            calls.append({"label": label, "problem_type": problem_type, "path": path})
            self.label = label

        def fit(self, **kwargs):
            calls[-1]["fit"] = kwargs
            return self

        def evaluate(self, data):
            return {"accuracy": 0.5}

    multimodal_mod = ModuleType("autogluon.multimodal")
    multimodal_mod.MultiModalPredictor = _RecordingPredictor
    monkeypatch.setitem(sys.modules, "autogluon", ModuleType("autogluon"))
    monkeypatch.setitem(sys.modules, "autogluon.multimodal", multimodal_mod)

    images = tmp_path / "images"
    (images / "red").mkdir(parents=True)
    (images / "red" / "shot.png").write_bytes(b"\x89PNG")
    annotated = pd.DataFrame({
        "image": ["red/shot.png"],
        "warm": [1],
        "bright": [0],
        "Image_Directory": [str(images)],
    })

    predictor, run_id = autogluon_utils.train_model(
        train_data=annotated,
        target=["warm", "bright"],
        run_name="cv_multilabel_dispatch",
        time_limit=5,
        task_type="Computer Vision - Multi-Label Classification",
        data_category="Computer Vision",
    )

    assert [call["label"] for call in calls] == ["warm", "bright"]
    assert {call["problem_type"] for call in calls} == {"classification"}
    for call, other_label in zip(calls, ["bright", "warm"]):
        trained = call["fit"]["train_data"]
        assert trained["image"].iloc[0] == os.path.normpath(os.path.join(str(images), "red", "shot.png"))
        assert os.path.isabs(trained["image"].iloc[0]), "the annotation table kept relative image paths"
        assert "Image_Directory" not in trained.columns
        assert other_label not in trained.columns, "the other label column leaked into this predictor"
        assert call["path"].endswith(call["label"])

    assert isinstance(predictor, autogluon_utils.MultiLabelAutoGluonPredictor)
    assert set(predictor.predictors_by_target) == {"warm", "bright"}
    assert run_id


def test_tabular_rows_are_refused_when_pandas_is_too_old_for_autogluon(monkeypatch):
    """In the all-engine interpreter pandas is pinned below 2.2 by PyCaret, and AutoGluon's tabular
    fit dies inside preprocessing with a bare OptionError. Say what is wrong before the fit."""
    monkeypatch.setattr(autogluon_utils, "_pandas_lacks_the_downcasting_option", lambda: True)
    frame = pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0], "y": ["a", "b", "a", "b"]})

    with pytest.raises(ValueError, match="no_silent_downcasting"):
        autogluon_utils.train_model(
            train_data=frame, target="y", run_name="ag_pandas_guard", time_limit=5,
            task_type="Classification", data_category="Tabular",
        )