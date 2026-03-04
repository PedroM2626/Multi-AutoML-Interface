"""
pipeline_parser.py — infer which AutoML pipeline step is active from live logs.

Each framework has a sequence of steps. This module parses log lines
to determine which step is "done", which is "active", and which is "pending".
"""

from typing import Optional

# ── Step definitions per framework ───────────────────────────────────────────
# Each step has:
#   label       — displayed name
#   keywords    — log keywords that signal this step has STARTED or is active
#   done_kw     — log keywords that signal this step is DONE (optional)
#   description — tooltip / explainer text

_STEPS: dict[str, list[dict]] = {
    "autogluon": [
        {
            "label": "Data Preparation",
            "icon": "📊",
            "keywords": ["preprocessing", "converting", "fitting", "loading data", "train_data"],
            "done_kw": ["beginning automl", "fitting model:"],
            "description": "Validates and preprocesses the dataset. Handles missing values, categorical encoding and feature types.",
        },
        {
            "label": "Fitting Models",
            "icon": "🤖",
            "keywords": ["fitting model:", "training model for", "fitting with cpus"],
            "done_kw": ["weightedensemble", "autogluon training complete"],
            "description": "Trains each individual model (LightGBM, XGBoost, CatBoost, RF, etc.) within the time budget.",
        },
        {
            "label": "Stacking / Ensembling",
            "icon": "🏗️",
            "keywords": ["weightedensemble", "ensemble weights", "stacking"],
            "done_kw": ["autogluon training complete"],
            "description": "Combines the best models using weighted ensembling or multi-layer stacking.",
        },
        {
            "label": "Evaluation",
            "icon": "📏",
            "keywords": ["leaderboard", "best model:", "validation score", "score_val"],
            "done_kw": ["tabularpredictor saved", "best model logged"],
            "description": "Evaluates all models on the validation set and builds the final leaderboard.",
        },
        {
            "label": "MLflow Logging",
            "icon": "📝",
            "keywords": ["mlflow", "log_artifacts", "logged successfully", "artifacts logged"],
            "done_kw": ["thread finished"],
            "description": "Persists model artifacts, parameters, and metrics to MLflow for tracking and versioning.",
        },
    ],
    "flaml": [
        {
            "label": "Data Preparation",
            "icon": "📊",
            "keywords": ["data ready", "preprocessing", "starting flaml"],
            "done_kw": ["executing hyperparameter search"],
            "description": "Validates the dataset, detects feature types, and prepares inputs for FLAML's optimizer.",
        },
        {
            "label": "Hyperparameter Search",
            "icon": "🔍",
            "keywords": ["executing hyperparameter search", "automl.fit", "[flaml.automl", "trial", "best config"],
            "done_kw": ["search finished"],
            "description": "FLAML runs a cost-effective search over hyperparameter configurations using Bayesian optimization.",
        },
        {
            "label": "Best Config Selection",
            "icon": "🏆",
            "keywords": ["search finished", "best estimator", "best loss", "best final"],
            "done_kw": ["saving best model"],
            "description": "Identifies the best-performing estimator and its configuration from the search results.",
        },
        {
            "label": "Model Saving",
            "icon": "💾",
            "keywords": ["saving best model", "model_path", "artifact_path"],
            "done_kw": ["mlflow", "logged successfully"],
            "description": "Serializes the trained model to disk using pickle.",
        },
        {
            "label": "MLflow Logging",
            "icon": "📝",
            "keywords": ["mlflow", "log_artifact", "logged successfully"],
            "done_kw": ["thread finished"],
            "description": "Persists model artifacts, parameters, and metrics to MLflow for tracking and versioning.",
        },
    ],
    "h2o": [
        {
            "label": "H2O Cluster Init",
            "icon": "🌊",
            "keywords": ["h2o cluster initialized", "initializing h2o", "h2o init"],
            "done_kw": ["starting h2o automl"],
            "description": "Starts the local H2O Java cluster and allocates memory for distributed model training.",
        },
        {
            "label": "Data Preparation",
            "icon": "📊",
            "keywords": ["preparing data", "h2oframe", "feature engineering", "asfactor"],
            "done_kw": ["starting h2o automl training"],
            "description": "Converts Pandas DataFrames to H2O frames and applies type casting for features/targets.",
        },
        {
            "label": "AutoML Training",
            "icon": "🤖",
            "keywords": ["starting h2o automl training", "automl session", "training completed", "aml.train"],
            "done_kw": ["training completed in"],
            "description": "H2O trains multiple model families (GBM, XGBoost, GLM, DRF, DeepLearning) and their variants.",
        },
        {
            "label": "Leaderboard & Scoring",
            "icon": "📏",
            "keywords": ["top 5 models", "leaderboard", "best model score", "auc", "total_models_trained"],
            "done_kw": ["model saved at", "log model to mlflow"],
            "description": "Ranks all trained models and evaluates the leader on the validation/test set.",
        },
        {
            "label": "MLflow Logging",
            "icon": "📝",
            "keywords": ["mlflow", "log_artifacts", "logged successfully", "artifacts logged"],
            "done_kw": ["thread finished"],
            "description": "Persists model artifacts, parameters, and metrics to MLflow for tracking and versioning.",
        },
    ],
    "tpot": [
        {
            "label": "Data Preparation",
            "icon": "📊",
            "keywords": ["problem type:", "training data shape", "test data shape", "label encoder"],
            "done_kw": ["starting tpot training"],
            "description": "Applies feature engineering pipelines: TF-IDF for text, ordinal encoding, and standard scaling.",
        },
        {
            "label": "Pipeline Generation (GA)",
            "icon": "🧬",
            "keywords": ["starting tpot training", "generation:", "pipeline score:", "optimizing pipeline"],
            "done_kw": ["training completed"],
            "description": "TPOT uses a Genetic Algorithm to evolve and select the best scikit-learn pipeline configurations.",
        },
        {
            "label": "Pipeline Selection",
            "icon": "🏆",
            "keywords": ["training completed", "best pipeline", "fitted_pipeline_", "accuracy:", "f1_macro:"],
            "done_kw": ["pipeline exported"],
            "description": "Identifies the highest-scoring pipeline from the genetic search as the final model.",
        },
        {
            "label": "Export & Analysis",
            "icon": "📤",
            "keywords": ["pipeline exported", "export", "classification report"],
            "done_kw": ["mlflow"],
            "description": "Exports the best pipeline as a .py file and generates a classification/regression report.",
        },
        {
            "label": "MLflow Logging",
            "icon": "📝",
            "keywords": ["mlflow", "tpot automl model", "registered_model_name", "logged successfully"],
            "done_kw": ["thread finished"],
            "description": "Persists model artifacts, parameters, and metrics to MLflow for tracking and versioning.",
        },
    ],
}

# ── Public API ────────────────────────────────────────────────────────────────

def get_framework_steps(framework_key: str) -> list[dict]:
    """Return the step definitions for a given framework key."""
    return _STEPS.get(framework_key.lower(), [])


def infer_pipeline_steps(framework_key: str, logs: list[str], status: str) -> list[dict]:
    """
    Returns enriched step list with status attached:
      status = "done" | "active" | "pending"

    On completed/failed/cancelled runs, all matched steps are "done".
    """
    steps = get_framework_steps(framework_key)
    if not steps:
        return []

    log_blob = " ".join(logs).lower()

    if status == "completed":
        # Mark all steps done
        return [{"label": s["label"], "icon": s["icon"], "description": s["description"], "status": "done"} for s in steps]

    if status in ("failed", "cancelled"):
        # Mark up to the last-seen step as done, rest pending, mark last active as failed
        last_done_idx = -1
        for i, step in enumerate(steps):
            if any(kw in log_blob for kw in step["keywords"]):
                last_done_idx = i

        result = []
        for i, step in enumerate(steps):
            if i < last_done_idx:
                st_val = "done"
            elif i == last_done_idx:
                st_val = "failed" if status == "failed" else "cancelled"
            else:
                st_val = "pending"
            result.append({"label": step["label"], "icon": step["icon"], "description": step["description"], "status": st_val})
        return result

    # Running or queued: find the active step
    last_done_idx = -1
    for i, step in enumerate(steps):
        done_signals = step.get("done_kw", [])
        if any(kw in log_blob for kw in done_signals):
            last_done_idx = i

    # Active = first step after last_done
    active_idx = min(last_done_idx + 1, len(steps) - 1)

    result = []
    for i, step in enumerate(steps):
        if i <= last_done_idx:
            st_val = "done"
        elif i == active_idx and status == "running":
            st_val = "active"
        else:
            st_val = "pending"
        result.append({"label": step["label"], "icon": step["icon"], "description": step["description"], "status": st_val})

    return result


def extract_best_tpot_pipeline(logs: list[str]) -> Optional[str]:
    """Extract the TPOT best pipeline string from logs."""
    for line in reversed(logs):
        if "best pipeline:" in line.lower() or "fitted_pipeline_" in line.lower():
            return line.strip()
        if "pipeline(" in line.lower():
            return line.strip()
    return None


def extract_autogluon_leaderboard_text(logs: list[str]) -> Optional[str]:
    """Extract leaderboard table text from AutoGluon logs."""
    rows = []
    capture = False
    for line in logs:
        if "model" in line.lower() and "score_val" in line.lower():
            capture = True
        if capture:
            rows.append(line)
            if len(rows) > 15:
                break
    return "\n".join(rows) if rows else None
