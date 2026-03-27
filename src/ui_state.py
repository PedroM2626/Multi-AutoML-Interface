import queue


SESSION_DEFAULTS: dict = {
    "df": None,
    "predictor": None,
    "model_type": None,
    "valid_df": None,
    "test_df": None,
    "active_df": None,
    "original_df": None,
    "target": None,
    "run_id": None,
    "dvc_hashes": {},
    "cv_folds": 0,
    "task_type": "Classification",
    "framework": "AutoGluon",
    "target_stats": {},
}


def init_session_state(session_state: dict) -> None:
    """Initialize Streamlit session defaults in a single place."""
    for key, value in SESSION_DEFAULTS.items():
        session_state.setdefault(key, value)

    if "log_queue" not in session_state:
        session_state["log_queue"] = queue.Queue()
