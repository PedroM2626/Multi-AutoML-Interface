"""Shared task/category metadata for the Streamlit UI and helpers."""

from __future__ import annotations

from typing import Iterable


DATA_CATEGORIES = ["Tabular", "Computer Vision", "Multimodal"]

TASK_OPTIONS_BY_CATEGORY = {
    "Tabular": [
        "Classification",
        "Regression",
        "Multi-Label Classification",
        "Anomaly Detection",
        "Time Series Forecasting",
        "Ranking",
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
    ("Tabular", "Anomaly Detection"): ["PyCaret"],
    ("Tabular", "Time Series Forecasting"): ["AutoGluon", "FLAML", "PyCaret"],
    ("Tabular", "Ranking"): ["FLAML"],
    ("Computer Vision", "Image Classification"): ["AutoGluon", "AutoKeras"],
    ("Computer Vision", "Multi-Label Classification"): ["AutoGluon", "AutoKeras"],
    ("Computer Vision", "Object Detection"): ["AutoGluon"],
    ("Computer Vision", "Image Segmentation"): ["AutoGluon"],
    ("Multimodal", "Classification"): ["AutoGluon"],
    ("Multimodal", "Regression"): ["AutoGluon"],
}

DEFAULT_DATA_CATEGORY = "Tabular"


def get_task_options(data_category: str) -> list[str]:
    return list(TASK_OPTIONS_BY_CATEGORY.get(data_category, TASK_OPTIONS_BY_CATEGORY[DEFAULT_DATA_CATEGORY]))


def get_framework_options(data_category: str, task_type: str) -> list[str]:
    return list(TASK_FRAMEWORK_MAP.get((data_category, task_type), ["FLAML"]))


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