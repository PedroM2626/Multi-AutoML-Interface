import pytest
import pandas as pd
import threading
from src.pycaret_utils import run_pycaret_experiment
import mlflow
import os

@pytest.fixture
def mock_classification_data():
    """Create a simple classification dataset."""
    df = pd.DataFrame({
        "feature_a": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0] * 5,
        "feature_b": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 5,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5
    })
    return df

def test_run_pycaret_experiment_classification(mock_classification_data, tmp_path, monkeypatch):
    """Test PyCaret experiment wrapper."""
    
    # Change MLflow URI to a temporary directory for tests
    mlflow_dir = tmp_path / "mlruns"
    os.makedirs(mlflow_dir, exist_ok=True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{mlflow_dir}")

    df = mock_classification_data
    
    # Run the experiment with a tiny configuration
    # Pycaret is aggressive, time_limit translates to n_iter in setup. Setting small bounds.
    result = run_pycaret_experiment(
        train_df=df,
        target_col="target",
        run_name="test_pycaret",
        log_queue=None,
        time_limit=10, # Translates to n_iter=1 internally for very fast checks
        val_df=None,
        stop_event=threading.Event()
    )
    
    # Asserts
    assert result is not None
    assert "error" not in result
    assert "predictor" in result
    assert "run_id" in result
    assert "metrics" in result
    
    # Basic structural check
    assert "accuracy" in result["metrics"]
    assert result["type"] == "pycaret"
