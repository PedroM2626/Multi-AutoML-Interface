from src.task_catalog import (
    DATA_CATEGORIES,
    TASK_FRAMEWORK_MAP,
    get_framework_options,
    get_task_options,
    infer_multimodal_columns,
    label_run_plan,
)


def test_computer_vision_multi_label_stays_one_experiment():
    """The bug this pins: the UI used to fan every multi-label selection out per column.

    A Computer Vision run with one label column is not a multi-label problem, so the engine
    refused it and every CV multi-label run from the interface failed twice - once per label -
    while the tabular fan-out is what the tabular engines actually want.
    """
    targets, is_multi = label_run_plan("Computer Vision", ["red", "circle"])
    assert is_multi is False
    assert targets == [["red", "circle"]]

    targets, is_multi = label_run_plan("Tabular", ["red", "circle"])
    assert is_multi is True
    assert targets == ["red", "circle"]

    targets, is_multi = label_run_plan("Tabular", "target")
    assert (targets, is_multi) == (["target"], False)


def test_the_ui_asks_the_catalog_how_to_split_a_multi_label_selection():
    """app.py used to loop the label columns itself, so a Computer Vision multi-label run became
    one run per label and every such run failed. The decision lives in label_run_plan; if the UI
    re-implements it, this test says so."""
    from pathlib import Path

    source = Path("app.py").read_text(encoding="utf-8")
    assert "label_run_plan(data_category, target)" in source
    assert "is_multi = len(target_cols)" not in source


def test_tabular_catalog_includes_expected_tasks_and_frameworks():
    assert DATA_CATEGORIES == ["Tabular", "Sequential", "Text", "Computer Vision", "Multimodal"]
    assert get_task_options("Tabular") == [
        "Classification",
        "Regression",
        "Multi-Label Classification",
        "Multi-Task Classification",
        "Anomaly Detection",
        "Clustering",
        "Forecast",
        "Ranking",
    ]
    assert get_framework_options("Tabular", "Classification") == ["AutoGluon", "FLAML", "H2O AutoML", "PyCaret", "Lale"]
    assert get_framework_options("Tabular", "Multi-Label Classification") == ["AutoGluon"]
    assert get_framework_options("Tabular", "Anomaly Detection") == ["PyCaret"]
    assert get_framework_options("Tabular", "Clustering") == ["PyCaret"]
    assert get_framework_options("Tabular", "Ranking") == ["FLAML"]


def test_rows_without_an_engine_path_are_gone():
    """Pinned here so they cannot come back as menu items that no engine implements:
    Semi-Supervised is a Classification checkbox, Sequential only differs from Tabular in the
    raw ordering its native time series path needs, and Text/Clustering had no text featurizer."""
    assert "Semi-Supervised Classification" not in get_task_options("Tabular")
    assert get_task_options("Sequential") == ["Forecast"]
    assert get_task_options("Text") == ["Classification", "Regression"]
    for pair in [("Tabular", "Semi-Supervised Classification"), ("Text", "Clustering"),
                 ("Sequential", "Classification"), ("Sequential", "Clustering")]:
        assert pair not in TASK_FRAMEWORK_MAP


def test_text_tasks_use_the_multimodal_engine_path():
    assert get_framework_options("Text", "Classification") == ["AutoGluon"]
    assert "HuggingFace" not in get_framework_options("Text", "Regression")


def test_multimodal_catalog_is_restricted_to_autogluon():
    assert get_task_options("Multimodal") == ["Classification", "Regression"]
    assert get_framework_options("Multimodal", "Classification") == ["AutoGluon"]


def test_infer_multimodal_columns_detects_text_and_image_paths():
    import pandas as pd

    df = pd.DataFrame(
        {
            "title": ["very long description about the product", "another long product description"],
            "image_path": ["/tmp/a.png", "/tmp/b.jpg"],
            "target": [0, 1],
        }
    )

    text_columns, image_columns = infer_multimodal_columns(df, "target")

    assert "title" in text_columns
    assert "image_path" in image_columns