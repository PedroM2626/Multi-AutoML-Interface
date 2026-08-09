import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple

from src.experiment_manager import ExperimentEntry, get_or_create_manager
from src.training_worker import run_training_worker

logger = logging.getLogger(__name__)

class UniversalAutoMLOrchestrator:
    """
    Universal orchestrator for Multi-AutoML-Interface.
    Maps framework names to their training functions, builds parameters,
    and runs or queues training threads independently of any UI.
    """

    FRAMEWORK_MAPPINGS = {
        "AutoGluon": ("autogluon", "src.autogluon_utils", "train_model"),
        "AutoKeras": ("autokeras", "src.autokeras_utils", "run_autokeras_experiment"),
        "FLAML": ("flaml", "src.flaml_utils", "train_flaml_model"),
        "H2O AutoML": ("h2o", "src.h2o_utils", "train_h2o_model"),
        "PyCaret": ("pycaret", "src.pycaret_utils", "run_pycaret_experiment"),
        "Lale": ("lale", "src.lale_utils", "run_lale_experiment"),
        "TPOT": ("tpot", "src.tpot_utils", "train_tpot_model"),
        "HuggingFace": ("huggingface", "src.huggingface_utils", "run_huggingface_experiment")
    }

    def __init__(self, framework: str, config: Dict[str, Any]):
        if framework not in self.FRAMEWORK_MAPPINGS:
            raise ValueError(f"Unknown framework: {framework}. Available: {list(self.FRAMEWORK_MAPPINGS.keys())}")
        self.framework = framework
        self.config = config
        self.fw_key, self.module_path, self.func_name = self.FRAMEWORK_MAPPINGS[framework]

    def _get_train_function_and_kwargs(self) -> Tuple[Any, Dict[str, Any]]:
        import importlib
        module = importlib.import_module(self.module_path)
        train_fn = getattr(module, self.func_name)
        
        # Build kwargs from self.config
        kwargs = self.config.copy()
        return train_fn, kwargs

    def run_synchronously(self) -> Any:
        """
        Run the training directly in the current thread (synchronous).
        
        Returns:
            Any: The training result.
        """
        train_fn, kwargs = self._get_train_function_and_kwargs()
        logger.info(f"Running {self.framework} training synchronously...")
        return train_fn(**kwargs)

    def queue_experiment(self, run_name: str, exp_manager=None) -> ExperimentEntry:
        """
        Queue the experiment to run in a background thread.
        
        Args:
            run_name: Name of the run.
            exp_manager: Optional ExperimentManager instance.
            
        Returns:
            ExperimentEntry: The queued experiment entry.
        """
        if exp_manager is None:
            exp_manager = get_or_create_manager()
            
        train_fn, kwargs = self._get_train_function_and_kwargs()
        
        target_col = kwargs.get('target') or kwargs.get('target_col') or kwargs.get('target_column')
        key = f"{self.fw_key}_{target_col}_{int(time.time())}"
        
        # Create ExperimentEntry
        entry = ExperimentEntry(
            key=key,
            metadata={
                "framework": self.framework,
                "framework_key": self.fw_key,
                "run_name": run_name,
                "target": target_col,
                "dataset_path": kwargs.get("dataset_path"),
                "config_snapshot": {k: v for k, v in kwargs.items()
                                   if k not in ("train_data", "df", "valid_data", "val_df", "test_data", "test_df", "dataset_path")}
            }
        )
        
        # Set log_queue inside kwargs
        if "log_queue" in kwargs:
            kwargs["log_queue"] = entry.log_queue
            
        # Spawn thread
        thread = threading.Thread(
            target=run_training_worker,
            args=(entry, train_fn, kwargs),
            daemon=True
        )
        entry.thread = thread
        
        exp_manager.add(entry)
        thread.start()
        logger.info(f"Queued background thread for {self.framework} training, key: {key}")
        
        return entry
