import pytest
import pandas as pd
import threading
from src.lale_utils import run_lale_experiment
import mlflow
import os

@pytest.fixture
def mock_classification_data():
    """Create a simple classification dataset."""
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    })
    return df

def test_run_lale_experiment_classification(mock_classification_data, tmp_path, monkeypatch):
    """Test Lale experiment wrapper with a fast minimal iteration run."""
    
    # Change MLflow URI to a temporary directory for tests
    mlflow_dir = tmp_path / "mlruns"
    os.makedirs(mlflow_dir, exist_ok=True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{mlflow_dir}")

    df = mock_classification_data
    
    # Run the experiment with a very short time budget (simulated by small iterations)
    # The lale_utils.py currently sets `max_evals=time_limit//10`
    result = run_lale_experiment(
        train_df=df,
        target_col="target",
        run_name="test_lale",
        log_queue=None,
        time_limit=10, # Very small budget, translates to max_evals=1
        cv_folds=2,
        val_df=df,
        stop_event=threading.Event()
    )
    
    # Asserts
    assert result is not None
    assert "error" not in result
    assert "predictor" in result
    assert "run_id" in result
    assert "metrics" in result
    
    # Ensure a metric is populated
    assert "f1_macro" in result["metrics"]
    
    # Basic structural check
    assert result["type"] == "lale"
