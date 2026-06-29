import pytest
import mlflow
import tempfile
import os
import shutil

@pytest.fixture(autouse=True)
def mlflow_test_setup_teardown():
    # End any active runs before test
    while mlflow.active_run():
        mlflow.end_run()
    
    # Clear environment variables
    env_vars = ["MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME"]
    old_envs = {v: os.environ.get(v) for v in env_vars}
    for v in env_vars:
        if v in os.environ:
            del os.environ[v]
            
    # Use a temporary directory for MLflow tracking in tests
    temp_dir = tempfile.mkdtemp()
    original_uri = mlflow.get_tracking_uri()
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"file:///{temp_dir}")
    mlflow.set_experiment("Default")
    
    yield
    
    # End any active runs after test
    while mlflow.active_run():
        mlflow.end_run()
    
    # Restore original URI
    mlflow.set_tracking_uri(original_uri)
    
    # Restore environment variables
    for v, val in old_envs.items():
        if val is not None:
            os.environ[v] = val
        elif v in os.environ:
            del os.environ[v]
            
    # Clean up temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)
