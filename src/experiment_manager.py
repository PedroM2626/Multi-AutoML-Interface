"""
ExperimentManager: central registry for all training runs.
Stored as a singleton in st.session_state['exp_manager'].
"""
import threading
import queue
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ExperimentEntry:
    key: str                          # unique slug: "autogluon_1712345678"
    metadata: dict                    # framework, run_name, config snapshot
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    log_queue: queue.Queue = field(default_factory=queue.Queue, repr=False)
    telemetry_queue: queue.Queue = field(default_factory=queue.Queue, repr=False)
    result_queue: queue.Queue = field(default_factory=queue.Queue, repr=False)
    status: str = "queued"            # queued | running | completed | failed | cancelled
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Optional[dict] = None     # {predictor, run_id, type, ...} or {error: str}
    all_logs: list = field(default_factory=list)
    latest_telemetry: dict = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)

    def elapsed_str(self) -> str:
        end = self.finished_at or time.time()
        secs = int(end - self.started_at)
        m, s = divmod(secs, 60)
        return f"{m}m {s:02d}s"

    def status_icon(self) -> str:
        return {
            "queued":    "⏳",
            "running":   "🟢",
            "completed": "✅",
            "failed":    "❌",
            "cancelled": "🚫",
        }.get(self.status, "❓")

    def drain_logs(self) -> bool:
        """Pull all pending log lines and telemetry into the entry."""
        new = False
        while not self.log_queue.empty():
            try:
                line = self.log_queue.get_nowait()
                self.all_logs.append(line)
                new = True
            except queue.Empty:
                break
        
        while not self.telemetry_queue.empty():
            try:
                data = self.telemetry_queue.get_nowait()
                if isinstance(data, dict):
                    self.latest_telemetry.update(data)
                    new = True
            except queue.Empty:
                break

        if new:
            self.last_update = time.time()
        return new

    def check_result(self):
        """Non-blocking check: pull result from queue if available."""
        if not self.result_queue.empty():
            try:
                res = self.result_queue.get_nowait()
                self.result = res
                if res.get("success"):
                    self.status = "completed"
                else:
                    self.status = "failed"
                self.finished_at = time.time()
                self.last_update = time.time()
            except queue.Empty:
                pass


class ExperimentManager:
    """In-process registry of all AutoML experiments."""

    def __init__(self):
        self._runs: dict[str, ExperimentEntry] = {}
        self._lock = threading.Lock()

    def add(self, entry: ExperimentEntry) -> str:
        with self._lock:
            self._runs[entry.key] = entry
        return entry.key

    def cancel(self, key: str):
        """Request graceful cancellation of a running experiment."""
        with self._lock:
            entry = self._runs.get(key)
            if entry and entry.status == "running":
                entry.stop_event.set()
                entry.status = "cancelled"
                entry.finished_at = time.time()
                entry.last_update = time.time()
                logger.info(f"Cancel requested for experiment: {key}")

    def delete(self, key: str):
        """Remove experiment from registry (only if not actively running)."""
        with self._lock:
            entry = self._runs.get(key)
            if entry and entry.status == "running":
                # Cancel first
                entry.stop_event.set()
                entry.status = "cancelled"
                entry.finished_at = time.time()
                entry.last_update = time.time()
            self._runs.pop(key, None)

    def get(self, key: str) -> Optional[ExperimentEntry]:
        with self._lock:
            return self._runs.get(key)

    def get_all(self) -> list[ExperimentEntry]:
        """Return all experiments newest-first."""
        with self._lock:
            entries = list(self._runs.values())
        return sorted(entries, key=lambda e: e.started_at, reverse=True)

    def has_running(self) -> bool:
        return any(e.status == "running" for e in self.get_all())

    def refresh_all(self):
        """Sync status/logs/results for all experiments."""
        for entry in self.get_all():
            entry.drain_logs()
            if entry.status in ("running", "queued"):
                entry.check_result()
                # Also check if thread died unexpectedly
                if getattr(entry, 'thread', None) is not None:
                    # Defensive check for is_alive
                    if not entry.thread.is_alive() and entry.status == "running":
                        if entry.result is None:
                            entry.status = "failed"
                            entry.result = {"success": False, "error": "Thread terminated unexpectedly"}
                        entry.finished_at = time.time()
                        entry.last_update = time.time()


def get_or_create_manager(session_state) -> ExperimentManager:
    """Get or create the singleton ExperimentManager from Streamlit session state."""
    if 'exp_manager' not in session_state or not isinstance(session_state.get('exp_manager'), ExperimentManager):
        session_state['exp_manager'] = ExperimentManager()
    return session_state['exp_manager']
