"""
training_worker: thread entry point for every AutoML run.
Captures stdout/stderr, feeds log_queue, puts result into result_queue,
and respects the stop_event for graceful cancellation.

Log isolation strategy (definitive):
  - We attach a _QueueLogHandler to each relevant named library logger.
  - Each handler has a _ThreadFilter that only accepts log records whose
    record.thread matches the experiment thread's ID.
  - This means messages from Thread A never land in Thread B's queue,
    even though they share the same named logger objects.
  - propagate is set to False to prevent double-delivery via the root logger.
  - All are restored in the finally block.

Stdout/Stderr isolation:
  - redirect_stdout/redirect_stderr are process-global (they overwrite sys.stdout).
  - We use a _ThreadAwareIO wrapper instead: it checks threading.current_thread()
    on every write() call, so writes only reach the owning thread's queue.
"""
import io
import sys
import logging
import threading
import traceback
import queue

from src.experiment_manager import ExperimentEntry

_LIB_LOGGERS = [
    'flaml', 'autogluon', 'mlflow', 'h2o', 'tpot',
    'pycaret', 'lale', 'hyperopt', 'lightgbm', 'xgboost', 'catboost'
]


# ---------------------------------------------------------------------------
# Thread-aware stdout/stderr router (installed once, process-wide)
# ---------------------------------------------------------------------------
class _ThreadAwareIO(io.TextIOBase):
    """
    Drop-in replacement for sys.stdout / sys.stderr that routes each write()
    to the queue registered for the current thread, or falls back to the
    original stream.
    """
    def __init__(self, original_stream):
        super().__init__()
        self._original = original_stream
        self._lock = threading.Lock()
        self._thread_queues: dict[int, "queue.Queue"] = {}

    def register(self, thread_id: int, q):
        with self._lock:
            self._thread_queues[thread_id] = q

    def unregister(self, thread_id: int):
        with self._lock:
            self._thread_queues.pop(thread_id, None)

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                return 0
        tid = threading.current_thread().ident
        with self._lock:
            q = self._thread_queues.get(tid)
        if q is not None:
            if s.strip():
                # Filter out progress bar characters that fail on Windows cp1252
                # \u2588 is the full block character
                safe_s = s.replace('\u2588', '#').replace('\u258c', '|').replace('\u2584', '-')
                q.put(safe_s.strip())
        else:
            # Fall back to original stream for threads not registered
            try:
                self._original.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass

    @property
    def encoding(self):
        return getattr(self._original, 'encoding', 'utf-8') or 'utf-8'

    @property
    def errors(self):
        return getattr(self._original, 'errors', 'replace')


# Install thread-aware routers once for the entire process
_stdout_router = _ThreadAwareIO(sys.__stdout__)
_stderr_router = _ThreadAwareIO(sys.__stderr__)
sys.stdout = _stdout_router
sys.stderr = _stderr_router


# ---------------------------------------------------------------------------
# Per-thread log handler with thread filter
# ---------------------------------------------------------------------------
class _ThreadFilter(logging.Filter):
    """Only accepts log records emitted by a specific OS thread."""
    def __init__(self, thread_id: int):
        super().__init__()
        self._thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self._thread_id


class _QueueLogHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------
def run_training_worker(entry: ExperimentEntry, train_fn, kwargs: dict):
    """
    Thread target. Runs train_fn(**kwargs, stop_event=entry.stop_event),
    keeps the entry status updated, and puts the final result dict in
    result_queue.
    """
    thread_id = threading.current_thread().ident

    # --- Thread-aware stdout/stderr capture ---
    _stdout_router.register(thread_id, entry.log_queue)
    _stderr_router.register(thread_id, entry.log_queue)

    # --- Per-thread logging handler (with thread filter) ---
    handler = _QueueLogHandler(entry.log_queue)
    handler.setFormatter(logging.Formatter('%(message)s'))
    handler.addFilter(_ThreadFilter(thread_id))

    saved_propagate: dict[str, bool] = {}
    for lib in _LIB_LOGGERS:
        lib_logger = logging.getLogger(lib)
        saved_propagate[lib] = lib_logger.propagate
        lib_logger.propagate = False  # prevents root from seeing AND double-deliver
        lib_logger.addHandler(handler)
        if lib_logger.level == logging.NOTSET or lib_logger.level > logging.INFO:
            lib_logger.setLevel(logging.INFO)

    entry.status = "running"
    entry.log_queue.put(f"[Worker] Starting training: {entry.metadata.get('run_name', entry.key)}")

    try:
        # Inject stop_event and telemetry_queue into kwargs if the function accepts it
        try:
            import inspect
            sig = inspect.signature(train_fn)
            if 'stop_event' in sig.parameters:
                kwargs['stop_event'] = entry.stop_event
            if 'telemetry_queue' in sig.parameters:
                kwargs['telemetry_queue'] = entry.telemetry_queue
        except Exception:
            pass

        result = train_fn(**kwargs)

        # Normalise result into a standard dict
        if isinstance(result, tuple):
            if len(result) == 2:
                predictor, run_id = result
                entry.result_queue.put({
                    "success": True, "predictor": predictor, "run_id": run_id,
                    "type": entry.metadata.get("framework_key", "unknown")
                })
            elif len(result) == 4:
                tpot, pipeline, run_id, info = result
                entry.result_queue.put({
                    "success": True, "predictor": pipeline, "run_id": run_id, "info": info, "type": "tpot"
                })
            else:
                entry.result_queue.put({
                    "success": True, "predictor": result[0], "run_id": result[-1],
                    "type": entry.metadata.get("framework_key", "unknown")
                })
        elif isinstance(result, dict):
            entry.result_queue.put(result)
        else:
            entry.result_queue.put({
                "success": True, "predictor": result, "run_id": None,
                "type": entry.metadata.get("framework_key", "unknown")
            })

    except StopIteration:
        entry.log_queue.put("[Worker] Training cancelled by user request.")
        entry.result_queue.put({"success": False, "cancelled": True, "error": "Cancelled by user"})
    except Exception as e:
        err_tb = traceback.format_exc()
        entry.log_queue.put(f"[Worker] CRITICAL ERROR: {e}\n{err_tb}")
        entry.result_queue.put({"success": False, "error": str(e), "traceback": err_tb})
    finally:
        # Restore all lib loggers
        for lib in _LIB_LOGGERS:
            lib_logger = logging.getLogger(lib)
            lib_logger.removeHandler(handler)
            lib_logger.propagate = saved_propagate.get(lib, True)

        # Unregister stdout/stderr routing for this thread
        _stdout_router.unregister(thread_id)
        _stderr_router.unregister(thread_id)

        entry.log_queue.put("[Worker] Thread finished.")
