import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.orchestrator import UniversalAutoMLOrchestrator
from src.experiment_manager import ExperimentManager

def test_orchestrator_initialization():
    config = {"train_data": pd.DataFrame({"x": [1], "y": [2]}), "target": "y"}
    orchestrator = UniversalAutoMLOrchestrator("FLAML", config)
    assert orchestrator.framework == "FLAML"
    assert orchestrator.fw_key == "flaml"

@patch("src.orchestrator.UniversalAutoMLOrchestrator._get_train_function_and_kwargs")
def test_orchestrator_queue_experiment(mock_get_funcs):
    # Setup mock
    mock_train_fn = MagicMock()
    mock_get_funcs.return_value = (mock_train_fn, {"target": "y", "time_limit": 1})
    
    # Initialize
    config = {"target": "y", "time_limit": 1}
    orchestrator = UniversalAutoMLOrchestrator("FLAML", config)
    
    # Mock manager
    manager = ExperimentManager()
    
    # Queue
    entry = orchestrator.queue_experiment("test_run", exp_manager=manager)
    
    assert entry is not None
    assert entry.status in ("running", "completed", "failed")
    assert manager.get(entry.key) == entry
    
    # Clean up thread
    if entry.thread and entry.thread.is_alive():
        entry.stop_event.set()
