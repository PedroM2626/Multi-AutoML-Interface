from src.task_catalog import DATA_CATEGORIES, get_framework_options, get_task_options, infer_multimodal_columns


def test_tabular_catalog_includes_expected_tasks_and_frameworks():
    assert DATA_CATEGORIES == ["Tabular", "Computer Vision", "Multimodal"]
    assert get_task_options("Tabular") == ["Classification", "Regression", "Time Series Forecasting", "Ranking"]
    assert get_framework_options("Tabular", "Classification") == ["AutoGluon", "FLAML", "H2O AutoML", "TPOT", "PyCaret", "Lale"]


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