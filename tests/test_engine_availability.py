"""Guards for the engine-availability filter that feeds the framework selector.

The desktop installer bundles only what requirements.txt installs, so most engines are
absent there; the selector has to offer what the interpreter can actually import.
"""
import ast

import pytest

from src.task_catalog import (
    FRAMEWORK_IMPORTS,
    TASK_FRAMEWORK_MAP,
    clear_availability_cache,
    install_hint,
    partition_frameworks,
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_availability_cache()
    yield
    clear_availability_cache()


def test_only_installed_engines_are_offered(monkeypatch):
    names = TASK_FRAMEWORK_MAP[("Tabular", "Classification")]
    monkeypatch.setattr(
        "src.task_catalog.importlib.util.find_spec",
        lambda name: object() if name == "flaml" else None,
    )
    installed, missing = partition_frameworks(names)
    assert installed == ["FLAML"]
    assert missing == [n for n in names if n != "FLAML"]


def test_unknown_label_counts_as_unavailable():
    installed, missing = partition_frameworks(["Not An Engine"])
    assert installed == []
    assert missing == ["Not An Engine"]


def test_positive_answers_are_cached_and_negatives_are_reprobed(monkeypatch):
    calls = []

    def fake_find_spec(name):
        calls.append(name)
        return object() if name == "flaml" else None

    monkeypatch.setattr("src.task_catalog.importlib.util.find_spec", fake_find_spec)
    partition_frameworks(["FLAML", "PyCaret"])
    partition_frameworks(["FLAML", "PyCaret"])
    # An engine can be installed while the app runs, so a "not found" must not be remembered;
    # a "found" is the expensive answer and may be.
    assert calls.count("flaml") == 1
    assert calls.count("pycaret") == 2


def test_install_hint_names_the_packages():
    assert install_hint(["H2O AutoML", "AutoGluon"]) == "autogluon h2o"


def test_the_orchestrator_refuses_an_engine_that_is_not_installed(monkeypatch):
    """The failure used to surface as "No module named 'autogluon'" from a worker thread."""
    from src.orchestrator import UniversalAutoMLOrchestrator

    monkeypatch.setattr("src.task_catalog.framework_available", lambda *args, **kwargs: False)
    orchestrator = UniversalAutoMLOrchestrator("AutoGluon", {"train_data": None})

    with pytest.raises(ModuleNotFoundError, match=r"pip install autogluon"):
        orchestrator.run_synchronously()


def test_rows_needing_an_engine_extra_check_that_extra(monkeypatch):
    """autogluon.tabular imports fine without autogluon.multimodal, which is what the vision,
    text and multimodal rows actually need."""
    installed = {"autogluon", "autogluon.tabular"}
    monkeypatch.setattr(
        "src.task_catalog.importlib.util.find_spec",
        lambda name: object() if name in installed else None,
    )
    assert partition_frameworks(["AutoGluon"], "Tabular") == (["AutoGluon"], [])
    assert partition_frameworks(["AutoGluon"], "Computer Vision") == ([], ["AutoGluon"])
    assert partition_frameworks(["AutoGluon"], "Text") == ([], ["AutoGluon"])


def test_every_extra_mapping_names_a_real_row():
    from src.task_catalog import FRAMEWORK_CATEGORY_MODULES, TASK_FRAMEWORK_MAP

    for framework, category in FRAMEWORK_CATEGORY_MODULES:
        offered = {
            task for (cat, task), engines in TASK_FRAMEWORK_MAP.items()
            if cat == category and framework in engines
        }
        assert offered, f"{framework} / {category} maps an extra but offers no row"


def test_the_ui_filters_both_framework_selectors():
    """A catalog label is not a promise that the engine is installed: the training selector and
    the model-source selector must both bind a filtered list, never the catalog itself."""
    source = open("app.py", encoding="utf-8").read()
    tree = ast.parse(source)
    filtered_lists = {"installed_frameworks", "loadable_frameworks"}

    options = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "selectbox" and len(node.args) > 1
                and isinstance(node.args[0], ast.Constant)
                and "Framework" in str(node.args[0].value)):
            options.append((node.args[0].value, node.args[1]))

    assert len(options) == 2, f"expected the training and model-framework selectors, found {len(options)}"
    for label, option in options:
        assert isinstance(option, ast.Name) and option.id in filtered_lists, (
            f"the '{label}' selector binds {ast.unparse(option)}; it has to bind one of "
            f"{sorted(filtered_lists)}"
        )
