"""Shared task/category metadata for the Streamlit UI and helpers."""

from __future__ import annotations

import importlib.util
import os
import threading
from typing import Iterable


DATA_CATEGORIES = ["Tabular", "Sequential", "Text", "Computer Vision", "Multimodal"]

TASK_OPTIONS_BY_CATEGORY = {
    "Tabular": [
        "Classification",
        "Regression",
        "Multi-Label Classification",
        "Multi-Task Classification",
        "Anomaly Detection",
        "Clustering",
        "Forecast",
        "Ranking",
    ],
    # "Sequential" only keeps Forecast: the other four sent the same engine task strings as
    # their Tabular twins, and no code path keyed on the category, so they were duplicates
    # that silently skipped the Tabular data-characteristics panel. Semi-Supervised
    # Classification was a task row while the real feature is the Classification checkbox.
    "Sequential": [
        "Forecast",
    ],
    # Text Clustering had no engine path: the map pointed at PyCaret's tabular clustering
    # module, which one-hot-encodes the text column instead of using any NLP featurizer.
    "Text": [
        "Classification",
        "Regression",
    ],
    # Multi-label image classification needs one column per label, which the folder layout cannot
    # express: an image sits in a single class folder. The CV upload therefore also takes an
    # annotations CSV ('image' plus one 0/1 column per label), and only that shape trains this row.
    # Object Detection and Image Segmentation stay removed: their AutoGluon pipeline needs mmcv, and
    # mmcv publishes no wheels on PyPI (its latest release is a sdist), so nothing here could
    # install it without compiling it against one exact torch build. The problem types stay in
    # autogluon_utils for a caller that brings an annotated dataframe.
    "Computer Vision": [
        "Image Classification",
        "Multi-Label Classification",
    ],
    "Multimodal": ["Classification", "Regression"],
}

TASK_FRAMEWORK_MAP = {
    ("Tabular", "Classification"): ["AutoGluon", "FLAML", "H2O AutoML", "PyCaret", "Lale"],
    ("Tabular", "Regression"): ["AutoGluon", "FLAML", "H2O AutoML", "PyCaret", "Lale"],
    ("Tabular", "Multi-Label Classification"): ["AutoGluon"],
    ("Tabular", "Multi-Task Classification"): ["AutoGluon", "FLAML", "H2O AutoML", "PyCaret", "Lale"],
    ("Tabular", "Anomaly Detection"): ["PyCaret"],
    ("Tabular", "Clustering"): ["PyCaret"],
    # Forecast runs two different pipelines on purpose. Under "Tabular" the data processor
    # builds lag/rolling features shifted by the horizon, so every engine solves a supervised
    # regression. Under "Sequential" the raw time-ordered frame reaches the engine and the
    # native time series paths are used (FLAML ts_forecast, PyCaret time_series) - which is
    # why AutoGluon is not offered there: its tabular predictor cannot forecast a future step
    # from same-row features.
    ("Tabular", "Forecast"): ["AutoGluon", "FLAML", "PyCaret"],
    ("Tabular", "Ranking"): ["FLAML"],
    ("Sequential", "Forecast"): ["FLAML", "PyCaret"],
    # Text goes through AutoGluon's MultiModalPredictor. HuggingFace was removed because
    # huggingface_utils.py only logged parameters and reported a successful run, and FLAML 2.x
    # no longer has the 'nlp' task; PyCaret was removed because its tabular setup one-hot
    # encodes a free-text column rather than featurizing it.
    ("Text", "Classification"): ["AutoGluon"],
    ("Text", "Regression"): ["AutoGluon"],
    ("Computer Vision", "Image Classification"): ["AutoGluon"],
    ("Computer Vision", "Multi-Label Classification"): ["AutoGluon"],
    ("Multimodal", "Classification"): ["AutoGluon"],
    ("Multimodal", "Regression"): ["AutoGluon"],
}

# AutoKeras is not offered either: `pip install autokeras` resolves to autokeras 3.0.0 against
# keras 3.x, and its classification head then fails before training - "Received an invalid value
# for `units`, expected a positive integer. Received: units=1" for image classification and a
# target-shape error for multi-label. There is no way back: 3.0.0 is autokeras' last release and
# its metadata requires keras>=3.0.0, so pinning keras down is not an option.
# src/autokeras_utils.py stays (with its numpy-array fit fix) for a future release that works.

# TPOT is not offered: tpot 1.1.0 raises TypeError from its own template ("TPOTEstimator
# .__init__() got an unexpected keyword argument 'scoring'") and tpot 0.12.2 only runs against
# scikit-learn < 1.5, while this project pins 1.9. Even in the interpreter that has scikit-learn
# 1.4.2 (requirements-all.txt) a classification run dies after fitting: MLflow logs the pipeline
# with skops and refuses it for untrusted types - tpot.builtins.stacking_estimator.StackingEstimator,
# sklearn.neighbors._kd_tree.KDTree, sklearn.metrics._dist_metrics.ManhattanDistance64 - and
# whitelisting those would switch off the same CWE-502 guard the model loader asks the user to
# confirm by hand. src/tpot_utils.py and the orchestrator entry stay for a caller that installs
# TPOT in its own environment.

DEFAULT_DATA_CATEGORY = "Tabular"

# Import name of each engine. The desktop installer bundles only what requirements.txt
# installs (FLAML, LightGBM, XGBoost), so most engines are absent there; offering them in the
# selector produced a ModuleNotFoundError inside a background thread instead of a usable app.
FRAMEWORK_IMPORTS = {
    "AutoGluon": "autogluon",
    "AutoKeras": "autokeras",
    "FLAML": "flaml",
    "H2O AutoML": "h2o",
    "PyCaret": "pycaret",
    "Lale": "lale",
    "TPOT": "tpot",
}

# Some rows need an optional extra of an engine that is already installed: `autogluon.tabular`
# resolves and imports without `autogluon.multimodal`, so a vision or text run died on
# "No module named 'autogluon.multimodal'" even though the framework looked available.
FRAMEWORK_CATEGORY_MODULES = {
    ("AutoGluon", "Tabular"): "autogluon.tabular",
    ("AutoGluon", "Computer Vision"): "autogluon.multimodal",
    ("AutoGluon", "Text"): "autogluon.multimodal",
    ("AutoGluon", "Multimodal"): "autogluon.multimodal",
}

_availability_cache: dict[str, bool] = {}
_availability_lock = threading.Lock()


def get_task_options(data_category: str) -> list[str]:
    return list(TASK_OPTIONS_BY_CATEGORY.get(data_category, TASK_OPTIONS_BY_CATEGORY[DEFAULT_DATA_CATEGORY]))


def get_framework_options(data_category: str, task_type: str) -> list[str]:
    return list(TASK_FRAMEWORK_MAP.get((data_category, task_type), ["FLAML"]))


def _module_available(module_name: str) -> bool:
    with _availability_lock:
        if _availability_cache.get(module_name):
            return True
    try:
        found = importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, AttributeError):
        found = False
    if found:
        with _availability_lock:
            _availability_cache[module_name] = True
    return found


def framework_import_name(framework: str, data_category: str | None = None) -> str | None:
    return FRAMEWORK_CATEGORY_MODULES.get((framework, data_category)) or FRAMEWORK_IMPORTS.get(framework)


def preload_torch_before_sklearn() -> None:
    """Load torch's native libraries before any module imports scikit-learn.

    scikit-learn <=1.4 vendors vcomp140.dll in sklearn/.libs and maps it from
    sklearn/_distributor_init.py. Once that MSVC OpenMP runtime is loaded, torch's c10.dll
    fails its DllMain with OSError [WinError 1114], and every autogluon.multimodal import
    after it dies - which is how the PyCaret-era lock broke the vision and text rows.
    Importing torch first leaves both usable. Wheels that do not vendor vcomp140.dll need
    no preload, so a modern interpreter does not pay for the import.
    """
    if importlib.util.find_spec("torch") is None:
        return
    sklearn_spec = importlib.util.find_spec("sklearn")
    if sklearn_spec is None or sklearn_spec.submodule_search_locations is None:
        return
    if not any(
        os.path.isfile(os.path.join(folder, ".libs", "vcomp140.dll"))
        for folder in sklearn_spec.submodule_search_locations
    ):
        return
    try:
        _import_torch()
    except Exception:
        # torch unusable: let the engine that needs it report the real error.
        pass


def _import_torch() -> None:
    import torch  # noqa: F401


def framework_available(framework: str, data_category: str | None = None) -> bool:
    """Return True when the module behind a catalog label can be imported.

    Only positive results are cached: find_spec on a package that exists is the expensive
    one, and an interpreter can gain an engine while the app serves several sessions, so a
    negative answer has to be re-checked on the next rerun.
    """
    module_name = framework_import_name(framework, data_category)
    if module_name is None:
        return False
    return _module_available(module_name)


def partition_frameworks(frameworks: Iterable[str], data_category: str | None = None) -> tuple[list[str], list[str]]:
    """Split catalog options into (installed, missing) preserving catalog order."""
    available: list[str] = []
    missing: list[str] = []
    for framework in frameworks:
        (available if framework_available(framework, data_category) else missing).append(framework)
    return available, missing


def install_hint(frameworks: Iterable[str], data_category: str | None = None) -> str:
    """pip requirement line for the given catalog labels, for the 'how do I get this' note.

    Uses the module the row actually imports, so a vision row asks for `autogluon.multimodal`
    rather than the umbrella that would not have been enough.
    """
    names = sorted({framework_import_name(f, data_category) for f in frameworks} - {None})
    return " ".join(names)


def clear_availability_cache() -> None:
    with _availability_lock:
        _availability_cache.clear()


def infer_multimodal_columns(df, target_column: str, sample_size: int = 25) -> tuple[list[str], list[str]]:
    """Heuristically suggest text and image columns for multimodal datasets."""
    text_columns: list[str] = []
    image_columns: list[str] = []

    for column in df.columns:
        if column == target_column:
            continue

        series = df[column].dropna().astype(str).head(sample_size)
        if series.empty:
            continue

        lower_sample = series.str.lower()
        image_ratio = lower_sample.str.contains(r"\.(png|jpg|jpeg|bmp|gif|webp|tif|tiff)$", regex=True).mean()
        if image_ratio >= 0.5:
            image_columns.append(column)
            continue

        if df[column].dtype == object or str(df[column].dtype) == "category":
            avg_length = series.str.len().mean()
            high_cardinality = df[column].nunique(dropna=True) > max(20, int(len(df) * 0.5))
            if avg_length >= 25 or high_cardinality:
                text_columns.append(column)

    return text_columns, image_columns


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen = set()
    ordered_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered_values.append(value)
    return ordered_values