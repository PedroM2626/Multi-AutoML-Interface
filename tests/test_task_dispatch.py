"""Guards for the task-type catalog and the UI -> engine dispatch contract.

These run with the minimal CI dependency set: no AutoML engine is imported, the
dispatch is checked against the engines' own source and against stub modules.
"""
import ast
import inspect
import re
import sys
import types

import pytest

from src.orchestrator import UniversalAutoMLOrchestrator
from src.task_catalog import (
    DATA_CATEGORIES,
    TASK_FRAMEWORK_MAP,
    TASK_OPTIONS_BY_CATEGORY,
    get_framework_options,
)

ENGINES = {"AutoGluon", "FLAML", "H2O AutoML", "TPOT", "PyCaret", "Lale", "AutoKeras", "HuggingFace"}

# The keys the UI injects for every run (app.py builds these dicts) plus the metadata
# entry it injects unconditionally at app.py:_kwargs["dataset_path"].
UI_INJECTED_METADATA = "dataset_path"


def test_catalog_is_self_consistent():
    declared = {(cat, task) for cat, tasks in TASK_OPTIONS_BY_CATEGORY.items() for task in tasks}
    assert declared == set(TASK_FRAMEWORK_MAP), (
        f"pairs only in options: {declared - set(TASK_FRAMEWORK_MAP)}; "
        f"pairs only in map: {set(TASK_FRAMEWORK_MAP) - declared}"
    )
    assert set(TASK_OPTIONS_BY_CATEGORY) == set(DATA_CATEGORIES)


def test_every_pair_offers_at_least_one_known_engine():
    for pair, frameworks in TASK_FRAMEWORK_MAP.items():
        assert frameworks, f"{pair} offers no engine"
        unknown = set(frameworks) - ENGINES
        assert not unknown, f"{pair} offers unknown engine(s): {unknown}"


def test_no_pair_relies_on_the_silent_flaml_fallback():
    # get_framework_options returns ["FLAML"] for an unknown pair; if a row were deleted
    # from the map the UI would keep working with a wrong engine and no test would notice.
    for cat, tasks in TASK_OPTIONS_BY_CATEGORY.items():
        for task in tasks:
            assert (cat, task) in TASK_FRAMEWORK_MAP
            assert get_framework_options(cat, task) == TASK_FRAMEWORK_MAP[(cat, task)]


def _branch_literals(module_file, names):
    """String literals compared inside a module, used to read the engines' contracts."""
    src = open(module_file, encoding="utf-8").read()
    tree = ast.parse(src)
    wanted = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id in names:
            for operand in node.comparators:
                if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                    wanted.add(operand.value)
    for match in re.finditer(r'startswith\("([^"]+)"\)', src):
        wanted.add(match.group(1))
    return wanted


def test_computer_vision_task_types_match_the_engine_branches():
    """CV names are stored bare in the catalog; the vision engines compare the prefixed
    form, which app.py now sends. A rename on either side must fail here."""
    ag_branches = _branch_literals("src/autogluon_utils.py", {"task_type"})
    ak_branches = _branch_literals("src/autokeras_utils.py", {"task_type"})

    for task in TASK_OPTIONS_BY_CATEGORY["Computer Vision"]:
        prefixed = f"Computer Vision - {task}"
        handled_by_autogluon = any(prefixed.startswith(b) or prefixed == b for b in ag_branches)
        assert handled_by_autogluon, f"AutoGluon has no branch for {prefixed!r}"
        if task in ("Image Classification", "Multi-Label Classification"):
            assert prefixed in ak_branches, f"AutoKeras has no branch for {prefixed!r}"


def test_ui_metadata_is_not_forwarded_to_engine_functions(monkeypatch, tmp_path):
    calls = {}

    def train(**kwargs):
        calls.update(kwargs)
        return "trained"

    stub = types.SimpleNamespace(train_flaml_model=train)
    monkeypatch.setitem(sys.modules, "src.flaml_utils", stub)

    config = {
        "train_data": object(),
        "target": "y",
        "run_name": "run_1",
        "task": "classification",
        UI_INJECTED_METADATA: str(tmp_path / "dataset.csv"),
    }
    orchestrator = UniversalAutoMLOrchestrator("FLAML", config)
    assert orchestrator.run_synchronously() == "trained"
    assert UI_INJECTED_METADATA not in calls, "dataset_path reached the engine kwargs"
    assert calls["run_name"] == "run_1"


def test_dataset_path_survives_into_entry_metadata(monkeypatch, tmp_path):
    stub = types.SimpleNamespace(train_flaml_model=lambda **kwargs: "trained")
    monkeypatch.setitem(sys.modules, "src.flaml_utils", stub)

    from src.experiment_manager import ExperimentManager

    path = str(tmp_path / "dataset.csv")
    orchestrator = UniversalAutoMLOrchestrator(
        "FLAML", {"train_data": object(), "target": "y", "run_name": "r", UI_INJECTED_METADATA: path}
    )
    entry = orchestrator.queue_experiment("r", exp_manager=ExperimentManager())
    assert entry.metadata["dataset_path"] == path
    assert UI_INJECTED_METADATA not in entry.metadata["config_snapshot"]


UI_KWARGS = {
    "AutoGluon": ["train_data", "target", "run_name", "valid_data", "test_data", "time_limit",
                  "presets", "seed", "cv_folds", "task_type", "data_category",
                  "multimodal_text_columns", "multimodal_image_columns"],
    "AutoKeras": ["train_data", "target", "run_name", "valid_data", "task_type", "time_limit"],
    "FLAML": ["train_data", "target", "run_name", "valid_data", "test_data", "time_budget",
              "task", "metric", "estimator_list", "seed", "cv_folds", "n_jobs"],
    "H2O AutoML": ["train_data", "target", "run_name", "valid_data", "test_data",
                   "max_runtime_secs", "max_models", "nfolds", "balance_classes", "seed",
                   "sort_metric", "exclude_algos"],
    "TPOT": ["df", "target_column", "run_name", "valid_data", "test_data", "generations",
             "population_size", "cv", "scoring", "max_time_mins", "max_eval_time_mins",
             "random_state", "verbosity", "n_jobs", "config_dict", "tfidf_max_features",
             "tfidf_ngram_range"],
    "PyCaret": ["df", "target_column", "run_name", "task_type", "data_category", "time_budget",
                "seed", "log_queue"],
    "Lale": ["train_df", "target_col", "run_name", "valid_df", "test_df", "task_type",
             "time_budget", "seed", "n_jobs"],
    "HuggingFace": ["train_data", "target", "run_name", "task_type", "time_limit"],
}


@pytest.mark.parametrize("framework", sorted(UI_KWARGS))
def test_engine_accepts_the_kwargs_the_ui_sends(framework):
    """Every key app.py builds for an engine must be declared by that engine's train
    function (or absorbed by **kwargs). Engines needing an optional dependency are
    skipped individually so one missing package cannot hide the rest."""
    _, module_path, func_name = UniversalAutoMLOrchestrator.FRAMEWORK_MAPPINGS[framework]
    try:
        module = __import__(module_path, fromlist=[func_name])
    except ImportError as exc:
        pytest.skip(f"{module_path} unavailable: {exc}")

    params = inspect.signature(getattr(module, func_name)).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    missing = [k for k in UI_KWARGS[framework] if k not in params]
    assert not missing, f"{framework}.{func_name} does not accept {missing}"


def test_ui_sends_the_prefixed_task_type_to_the_vision_engines():
    """The dispatch, not just the contract: app.py must pass the prefixed value for CV."""
    source = open("app.py", encoding="utf-8").read()
    for framework in ("AutoGluon", "AutoKeras"):
        block = source.split(f'framework == "{framework}"', 1)[1].split("_kwargs = dict(", 1)[1].split(")", 1)[0]
        assert "task_type=engine_task_type" in block, (
            f"{framework} receives the bare task type again; the vision engines branch on "
            "the 'Computer Vision - <task>' form"
        )

