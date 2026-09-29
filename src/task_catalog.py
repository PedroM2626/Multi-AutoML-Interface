"""Shared task/category metadata for the Streamlit UI and helpers."""

from __future__ import annotations

import importlib.util
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
    "Computer Vision": [
        "Image Classification",
        "Multi-Label Classification",
        "Object Detection",
        "Image Segmentation",
    ],
    "Multimodal": ["Classification", "Regression"],
}

TASK_FRAMEWORK_MAP = {
    ("Tabular", "Classification"): ["AutoGluon", "FLAML", "H2O AutoML", "TPOT", "PyCaret", "Lale"],
    ("Tabular", "Regression"): ["AutoGluon", "FLAML", "H2O AutoML", "TPOT", "PyCaret", "Lale"],
    ("Tabular", "Multi-Label Classification"): ["AutoGluon"],
    ("Tabular", "Multi-Task Classification"): ["AutoGluon", "FLAML", "H2O AutoML", "TPOT", "PyCaret", "Lale"],
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
    ("Computer Vision", "Image Classification"): ["AutoGluon", "AutoKeras"],
    ("Computer Vision", "Multi-Label Classification"): ["AutoGluon", "AutoKeras"],
    ("Computer Vision", "Object Detection"): ["AutoGluon"],
    ("Computer Vision", "Image Segmentation"): ["AutoGluon"],
    ("Multimodal", "Classification"): ["AutoGluon"],
    ("Multimodal", "Regression"): ["AutoGluon"],
}

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

_availability_cache: dict[str, bool] = {}
_availability_lock = threading.Lock()


def get_task_options(data_category: str) -> list[str]:
    return list(TASK_OPTIONS_BY_CATEGORY.get(data_category, TASK_OPTIONS_BY_CATEGORY[DEFAULT_DATA_CATEGORY]))


def get_framework_options(data_category: str, task_type: str) -> list[str]:
    return list(TASK_FRAMEWORK_MAP.get((data_category, task_type), ["FLAML"]))


def framework_available(framework: str) -> bool:
    """Return True when the engine behind a catalog label can be imported.

    Only positive results are cached: find_spec on a package that exists is the expensive
    one, and an interpreter can gain an engine while the app serves several sessions, so a
    negative answer has to be re-checked on the next rerun.
    """
    module_name = FRAMEWORK_IMPORTS.get(framework)
    if module_name is None:
        return False
    with _availability_lock:
        if _availability_cache.get(framework):
            return True
    try:
        found = importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, AttributeError):
        found = False
    if found:
        with _availability_lock:
            _availability_cache[framework] = True
    return found


def partition_frameworks(frameworks: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split catalog options into (installed, missing) preserving catalog order."""
    available: list[str] = []
    missing: list[str] = []
    for framework in frameworks:
        (available if framework_available(framework) else missing).append(framework)
    return available, missing


def install_hint(frameworks: Iterable[str]) -> str:
    """pip requirement line for the given catalog labels, for the 'how do I get this' note."""
    names = sorted({FRAMEWORK_IMPORTS[f] for f in frameworks if f in FRAMEWORK_IMPORTS})
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