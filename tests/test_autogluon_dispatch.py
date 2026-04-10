from types import ModuleType, SimpleNamespace
import sys

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