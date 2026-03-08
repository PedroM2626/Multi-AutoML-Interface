import pytest
import pandas as pd
import os
import shutil
from src.autokeras_utils import run_autokeras_experiment
from src.modelsearch_utils import run_modelsearch_experiment

@pytest.fixture
def mock_cv_dataset(tmp_path):
    img_dir = tmp_path / "data_lake" / "images" / "mock_dataset"
    class_a = img_dir / "class_a"
    class_b = img_dir / "class_b"
    class_a.mkdir(parents=True, exist_ok=True)
    class_b.mkdir(parents=True, exist_ok=True)
    
    # Create fake images
    (class_a / "img1.png").touch()
    (class_b / "img2.png").touch()
    
    df = pd.DataFrame([{"Image_Directory": str(img_dir), "Type": "Image"}])
    return df, str(img_dir)

def test_autokeras_cv_failure_on_missing_dir():
    # Attempting to run without Image_Directory should raise ValueError
    df = pd.DataFrame([{"A": 1}])
    with pytest.raises(ValueError):
        run_autokeras_experiment(df, "label", "run1")

def test_modelsearch_cv_failure_on_missing_dir():
    df = pd.DataFrame([{"A": 1}])
    with pytest.raises(ValueError):
        run_modelsearch_experiment(df, "label", "run2")

def test_modelsearch_mock_run(mock_cv_dataset):
    df, _ = mock_cv_dataset
    res = run_modelsearch_experiment(df, "label", "run3", task_type="Computer Vision - Image Classification")
    assert res is not None
    assert res["type"] == "model_search"
    assert res["predictor"] is None
