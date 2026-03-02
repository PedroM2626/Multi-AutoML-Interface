"""
training_worker: thread entry point for every AutoML run.
Captures stdout/stderr, feeds log_queue, puts result into result_queue,
and respects the stop_event for graceful cancellation.
"""
import io
import logging
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr

from src.experiment_manager import ExperimentEntry


class _LogIO(io.StringIO):
    """StringIO that feeds every write into a queue."""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def write(self, s):
        if s.strip():
            self.log_queue.put(s.strip())
        return super().write(s)

    def flush(self):
        pass


class _QueueLogHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            pass


def run_training_worker(entry: ExperimentEntry, train_fn, kwargs: dict):
    """
    Thread target. Runs train_fn(**kwargs, stop_event=entry.stop_event),
    keeps the entry status updated, and puts the final result dict in result_queue.
    """
    # --- Logging capture setup ---
    handler = _QueueLogHandler(entry.log_queue)
    handler.setFormatter(logging.Formatter('%(message)s'))

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    for lib in ['flaml', 'autogluon', 'mlflow', 'h2o', 'tpot']:
        lib_logger = logging.getLogger(lib)
        lib_logger.addHandler(handler)
        lib_logger.setLevel(logging.INFO)

    log_io = _LogIO(entry.log_queue)

    entry.status = "running"
    entry.log_queue.put(f"[Worker] Starting training: {entry.metadata.get('run_name', entry.key)}")

    try:
        with redirect_stdout(log_io), redirect_stderr(log_io):
            # Inject stop_event into kwargs if the function accepts it
            try:
                import inspect
                sig = inspect.signature(train_fn)
                if 'stop_event' in sig.parameters:
                    kwargs['stop_event'] = entry.stop_event
            except Exception:
                pass

            result = train_fn(**kwargs)

        # result is expected to be a tuple or dict depending on the trainer
        # Normalise into a standard dict
        if isinstance(result, tuple):
            # Most trainers return (predictor, run_id)
            if len(result) == 2:
                predictor, run_id = result
                entry.result_queue.put({"success": True, "predictor": predictor, "run_id": run_id, "type": entry.metadata.get("framework_key", "unknown")})
            elif len(result) == 4:
                # TPOT: (tpot, pipeline, run_id, info)
                tpot, pipeline, run_id, info = result
                entry.result_queue.put({"success": True, "predictor": tpot, "pipeline": pipeline, "run_id": run_id, "info": info, "type": "tpot"})
            else:
                entry.result_queue.put({"success": True, "predictor": result[0], "run_id": result[-1], "type": entry.metadata.get("framework_key", "unknown")})
        elif isinstance(result, dict):
            entry.result_queue.put(result)
        else:
            entry.result_queue.put({"success": True, "predictor": result, "run_id": None, "type": entry.metadata.get("framework_key", "unknown")})

    except StopIteration:
        entry.log_queue.put("[Worker] Training cancelled by user request.")
        entry.result_queue.put({"success": False, "cancelled": True, "error": "Cancelled by user"})
    except Exception as e:
        err_tb = traceback.format_exc()
        entry.log_queue.put(f"[Worker] CRITICAL ERROR: {e}\n{err_tb}")
        entry.result_queue.put({"success": False, "error": str(e), "traceback": err_tb})
    finally:
        root_logger.removeHandler(handler)
        for lib in ['flaml', 'autogluon', 'mlflow', 'h2o', 'tpot']:
            logging.getLogger(lib).removeHandler(handler)
        entry.log_queue.put("[Worker] Thread finished.")
