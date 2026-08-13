# Multi-AutoML Interface — Complete Documentation

> Unified AutoML workbench built on Streamlit: train, compare, explain, and deploy models across 8 AutoML engines with full MLOps (MLflow + DVC), explainability (SHAP / saliency), and one-click deployment targets.

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation & Environments](#3-installation--environments)
4. [Task & Framework Support Matrix](#4-task--framework-support-matrix)
5. [User Guide](#5-user-guide)
6. [Framework Configuration Reference](#6-framework-configuration-reference)
7. [MLOps](#7-mlops)
8. [Deployment](#8-deployment)
9. [Testing & CI](#9-testing--ci)
10. [Project Structure](#10-project-structure)
11. [Troubleshooting](#11-troubleshooting)
12. [Contributing & Roadmap](#12-contributing--roadmap)

---

## 1. Overview

Multi-AutoML Interface is a single Streamlit application (`app.py`) that unifies multiple AutoML engines behind one consistent workflow: **upload → explore → configure → train (in background threads) → compare → predict → explain → export → deploy**.

### Core capabilities

| Area | What you get |
|---|---|
| AutoML engines | 8 engines behind a universal orchestrator (see below) |
| Data management | DVC-versioned local Data Lake (`data_lake/raw`, `data_lake/images`) with content hashing and graceful fallback when DVC is not installed |
| Experiment tracking | MLflow local file store (`mlruns/`) by default, optional remote server via `MLFLOW_TRACKING_URI`, Model Registry, auto-healing of corrupted stores, TTL cache |
| Concurrency | Multiple simultaneous training runs in background daemon threads with per-run log isolation, live telemetry, and graceful cancellation |
| Explainability (XAI) | SHAP waterfall/summary explanations for tabular models; occlusion saliency maps for Computer Vision models |
| Portability | ONNX export, Hugging Face Hub push, generated consumption code, auto-generated white-box Jupyter notebook (logged as MLflow artifact) |
| Serving | Generated FastAPI + Docker deployment packages (`deploy_<run_id>/`) |
| Distribution | Docker Compose, standalone Dockerfile, CV-specific Dockerfiles, Electron desktop app, Render, GitHub Actions CI |

### The 8 AutoML engines

| Engine | Integration module | Strengths in this project |
|---|---|---|
| AutoGluon | `src/autogluon_utils.py` | Tabular, Text, Time Series, Computer Vision (incl. Object Detection, Image Segmentation), Multimodal |
| FLAML | `src/flaml_utils.py` | Cost-effective hyperparameter search; classification, regression, forecast, ranking |
| H2O AutoML | `src/h2o_utils.py` | Distributed-style Java cluster training with native leaderboards (requires Java) |
| TPOT | `src/tpot_utils.py` | Genetic-algorithm pipeline search; exports the best pipeline as a `.py` file to `tpot_models/` |
| PyCaret | `src/pycaret_utils.py` | Broadest task coverage: semi-supervised, anomaly detection, clustering, time series |
| Lale | `src/lale_utils.py` | Hyperopt-based pipeline composition over scikit-learn operators |
| AutoKeras | `src/autokeras_utils.py` | Neural architecture search for image classification / multi-label CV tasks |
| HuggingFace | `src/huggingface_utils.py` | Hub integration (list/download/upload models) and experiment logging for text tasks |

### The 5 data categories

Defined in `src/task_catalog.py` (`DATA_CATEGORIES`):

1. **Tabular** — CSV/Excel with numeric, categorical, or text columns.
2. **Sequential** — time-ordered tabular data (forecast, anomaly detection, etc.).
3. **Text** — NLP-style classification/regression/clustering over text columns.
4. **Computer Vision** — image folders/ZIP uploads; labels inferred from directory structure.
5. **Multimodal** — mixed tabular + text + image-path columns (natively supported via AutoGluon).

### Deployment targets

- Docker Compose stack (Streamlit UI + MLflow UI)
- Standalone Docker image and two CV-specialized images
- Electron desktop application (Windows/macOS/Linux installers)
- Render cloud deployment (via `render.yaml`)
- Hugging Face Spaces (demo only)
- Generated per-model FastAPI + Docker packages (`deploy_<run_id>/`)

---

## 2. Architecture

### High-level diagram

```
                         ┌──────────────────────────────────────────────────────┐
                         │                  app.py (Streamlit UI)               │
                         │  Sidebar: 6 pages (src/navigation.py NAV_ITEMS)      │
                         │  Upload → EDA → Training → Experiments → Predict     │
                         │  → History(MLflow)                                   │
                         └──────┬───────────────────────────────┬───────────────┘
                                │ queue_experiment()            │ drain logs/results
                                ▼                               │ (st.fragment every 5s)
                 ┌──────────────────────────────┐               │
                 │ UniversalAutoMLOrchestrator  │               │
                 │ (src/orchestrator.py)        │               │
                 │ framework → train fn mapping │               │
                 └──────┬───────────────────────┘               │
                        │ spawns daemon thread                  │
                        ▼                                       │
                 ┌──────────────────────────────┐               │
                 │ run_training_worker          │               │
                 │ (src/training_worker.py)     │               │
                 │ thread-aware stdout/stderr,  │               │
                 │ per-thread log handlers,     │               │
                 │ stop_event, telemetry_queue  │               │
                 └──────┬───────────────────────┘               │
                        │ calls train_fn(**kwargs)              │
                        ▼                                       │
   ┌─────────────────────────────────────────────────────────┐  │
   │ Engine modules: autogluon_utils / flaml_utils /         │  │
   │ h2o_utils / tpot_utils / pycaret_utils / lale_utils /   │  │
   │ autokeras_utils / huggingface_utils                     │  │
   └──────┬──────────────────────────────────────────────────┘  │
          │ MLflow logging (file:///mlruns by default)          │
          ▼                                                     │
   ┌─────────────┐  ┌──────────────┐  ┌────────────────┐        │
   │ mlruns/     │  │ models/      │  │ tpot_models/   │        │
   │ (MLflow)    │  │ (local saves │  │ (TPOT .py +    │        │
   │             │  │  + ONNX)     │  │  info exports) │        │
   └─────────────┘  └──────────────┘  └────────────────┘        │
                                                                 │
   ┌──────────────────────────────────────────────────────────┐  │
   │ ExperimentManager singleton (src/experiment_manager.py)  │◀─┘
   │ st.session_state['exp_manager']                          │
   │ ExperimentEntry: status, log_queue, telemetry_queue,     │
   │ result_queue, stop_event, metadata, result               │
   └──────────────────────────────────────────────────────────┘

   Data plane: data_lake/raw + data_lake/images (+.dvc) ← src/data_utils.py (DVC CLI via subprocess)
```

### Threading model

The training pipeline is split across three modules:

1. **`src/orchestrator.py` — `UniversalAutoMLOrchestrator`**
   - Maps the 8 framework display names to `(framework_key, module, train_function)` via `FRAMEWORK_MAPPINGS` (e.g. `"H2O AutoML" → ("h2o", "src.h2o_utils", "train_h2o_model")`).
   - Lazily imports the engine module with `importlib` (engines never block app startup).
   - `queue_experiment(run_name, exp_manager)` builds an `ExperimentEntry` with a key like `autogluon_<target>_<timestamp>`, snapshots the config into `metadata`, spawns a **daemon `threading.Thread`** targeting `run_training_worker`, registers it in the manager, and starts it. Multiple experiments run concurrently.

2. **`src/training_worker.py` — `run_training_worker`**
   - Thread entry point for every run. Injects `stop_event` and `telemetry_queue` into the training function's kwargs **only if its signature accepts them** (checked via `inspect.signature`).
   - **Log isolation** — two mechanisms prevent log cross-contamination between concurrent runs:
     - `_ThreadAwareIO` is installed process-wide as `sys.stdout`/`sys.stderr`; each `write()` checks `threading.current_thread()` and routes output only to the owning thread's `log_queue` (also sanitizes progress-bar block characters for Windows cp1252).
     - A `_QueueLogHandler` with a `_ThreadFilter` (matches `record.thread`) is attached to the library loggers (`flaml`, `autogluon`, `mlflow`, `h2o`, `tpot`, `pycaret`, `lale`, `hyperopt`, `lightgbm`, `xgboost`, `catboost`); `propagate` is disabled to avoid double delivery and restored in `finally`.
   - Normalizes each engine's return value (tuple or dict) into a standard `{success, predictor, run_id, type}` message placed on `result_queue`.
   - After success, automatically generates a **white-box Jupyter notebook** via `src/notebook_generator.py` and logs it to the MLflow run as an artifact.
   - Cancellation: engines that honor `stop_event` stop cooperatively; a `StopIteration` raised by a worker is treated as a user cancellation.

3. **`src/experiment_manager.py` — `ExperimentManager`**
   - Thread-safe registry (lock-guarded dict) stored as a **singleton in `st.session_state['exp_manager']`** via `get_or_create_manager(session_state)`.
   - Each `ExperimentEntry` holds: `thread`, `stop_event`, `log_queue`, `telemetry_queue`, `result_queue`, `status` (`queued → running → completed | failed | cancelled`), timestamps, `result`, `all_logs`, `latest_telemetry`.
   - `cancel(key)` sets the `stop_event` for **graceful cancellation**; `delete(key)` cancels first if still running.
   - `refresh_all()` drains log/telemetry queues and checks results; it also detects dead threads (thread not alive + no result ⇒ marked `failed`).

**Live dashboard:** the Experiments page wraps its dashboard in a Streamlit fragment (`@st.fragment(run_every="5s")` when available — falls back to a plain function on older Streamlit via `_compat_fragment`), which re-renders status cards, pipeline visualizations, and color-coded logs every 5 seconds without rerunning the whole script.

### Storage layout

| Path | Purpose |
|---|---|
| `mlruns/` | MLflow local file-based tracking store (experiments, runs, artifacts). Auto-healed on startup. |
| `data_lake/raw/` | Versioned tabular datasets (CSV), named `<prefix>_<timestamp>.csv` with sibling `.dvc` files |
| `data_lake/images/` | Computer Vision datasets (`<dataset_name>_<timestamp>/` dirs) |
| `.dvc/` | DVC repository metadata (initialized lazily via `dvc init`) |
| `models/` | Local model saves, ONNX exports (e.g. `<run_name>.onnx`), HF downloads (`models/hf_downloads`) |
| `tpot_models/` | TPOT exports: `best_pipeline_<run_name>.py` and `model_info_<run_name>.txt` |
| `deploy_<id>/` | Generated FastAPI deployment packages (`main.py`, `requirements.txt`, `Dockerfile`, `README.md`). From the Experiments page the folder is `deploy_<experiment_key>`; from the History page it is `deploy_<run_id[:8]>`. |
| `temp/` | Temporary files (e.g. uploaded ONNX sessions) |

---

## 3. Installation & Environments

### Python version compatibility

| Runtime | Python | Notes |
|---|---|---|
| Full local app via `run.py` | **3.11 minimum (hard requirement for PyCaret/Lale)** | PyCaret and Lale require Python 3.11 with a matching scikit-learn. `run.py` re-launches itself on a Python 3.11 interpreter (`py -3.11`, `python3.11`, or `python`) when the current one is older, and refuses to start if none is found. Newer interpreters (3.12+) pass the check. |
| Core app (without PyCaret/Lale) | 3.12 | The Streamlit app itself, CI, and the base Docker image run on Python 3.12. |
| Docker base image | 3.12 | `Dockerfile` uses `python:3.12-slim`. |
| CI workflows | 3.12 | `actions/setup-python` with `python-version: "3.12"`. |
| Generated deployment packages | 3.11 | `deploy_<run_id>/Dockerfile` uses `python:3.11-slim`. |
| AutoGluon CV image | 3.10 | `Dockerfile.autogluon_cv` uses `python:3.10-slim` (torch 2.1.0 + mmcv 2.1.0 constraint). |

### Requirement files

| File | Role |
|---|---|
| `requirements.txt` | **Lightweight core stack**: Streamlit, pandas, numpy, scikit-learn, MLflow, FLAML, xgboost, matplotlib, FastAPI/Uvicorn, nbformat, pytest. Deliberately excludes the heavy engines so the base app and Docker image stay small. |
| `requirements-compiled.txt` | **Full pinned stack**, autogenerated with `pip-compile` for **Python 3.11** (`pip-compile --output-file=requirements-compiled.txt requirements.txt`), includes the framework transitive trees. Use for a reproducible full environment. |
| `requirements-dev.txt` | Developer quality gates: `ruff==0.13.1`, `pytest==8.4.2`. |

### Basic quick start

```bash
# 1. Clone the repository
git clone https://github.com/PedroM2626/Multi-AutoML-Interface.git
cd Multi-AutoML-Interface

# 2. Create a Python 3.11 virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux/macOS

# 3. Install the core stack (lightweight)
pip install -r requirements.txt

# 4. Launch (run.py enforces Python 3.11)
python run.py
# or directly:
py -3.11 -m streamlit run app.py
```

The UI opens at `http://localhost:8501`.

### Optional per-framework installs

All heavy engines are **optional**. They are imported lazily inside the engine modules and the orchestrator, so the app starts and degrades gracefully when a framework is missing (the failing run reports the import error; the rest of the app keeps working).

| Extra | Install command | Additional requirements |
|---|---|---|
| AutoGluon | `pip install autogluon` | Large install (PyTorch for CV/multimodal) |
| H2O AutoML | `pip install h2o` | **Java 11+** (JRE/JDK) must be on PATH |
| TPOT | `pip install tpot` | scikit-learn compatible version |
| PyCaret | `pip install pycaret` | Requires Python 3.11 |
| Lale | `pip install lale` | Requires Python 3.11 |
| AutoKeras | `pip install autokeras` | Requires TensorFlow |
| SHAP (XAI) | `pip install shap` | Needed for tabular explanations |
| Auto-EDA | `pip install ydata-profiling streamlit-pandas-profiling` | Powers the Data Exploration report |
| DVC | `pip install dvc` | Data-lake versioning (falls back to MD5 hashing when absent) |
| ONNX | `pip install onnx onnxruntime` | ONNX export/load |
| Hugging Face Hub | `pip install huggingface_hub` | Push/download models to/from the Hub |
| Deep Feature Synthesis | `pip install featuretools` | Optional DFS step in `src/processor.py` |
| DagsHub | `pip install dagshub` | Sidebar DagsHub integration |

### Non-Python prerequisites

- **Node.js 18+** — only needed for the Electron desktop app (`npm install`, `npm run dev` / `npm start`). CI builds with Node 20.
- **Java 11+** — only needed for H2O AutoML outside Docker (`src/h2o_utils.py` checks Java availability before training and raises an actionable error with Docker alternatives).

---

## 4. Task & Framework Support Matrix

> **Source of truth: `src/task_catalog.py` (`TASK_FRAMEWORK_MAP`) — update this table when the catalog changes.**

Legend: ✅ supported · — not supported · β beta

### Tabular

| Task | AutoGluon | FLAML | H2O AutoML | TPOT | PyCaret | Lale | AutoKeras | HuggingFace |
|---|---|---|---|---|---|---|---|---|
| Classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| Regression | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| Multi-Label Classification | ✅ | — | — | — | — | — | — | — |
| Multi-Task Classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| Semi-Supervised Classification | — | — | — | — | ✅ | — | — | — |
| Anomaly Detection | — | — | — | — | ✅ | — | — | — |
| Clustering | — | — | — | — | ✅ | — | — | — |
| Forecast | ✅ | ✅ | — | — | ✅ | — | — | — |
| Ranking | — | ✅ | — | — | — | — | — | — |

### Sequential

| Task | AutoGluon | FLAML | H2O AutoML | TPOT | PyCaret | Lale | AutoKeras | HuggingFace |
|---|---|---|---|---|---|---|---|---|
| Classification | ✅ | — | — | — | ✅ | — | — | — |
| Regression | ✅ | — | — | — | ✅ | — | — | — |
| Forecast | ✅ | ✅ | — | — | ✅ | — | — | — |
| Anomaly Detection | — | — | — | — | ✅ | — | — | — |
| Clustering | — | — | — | — | ✅ | — | — | — |

### Text

| Task | AutoGluon | FLAML | H2O AutoML | TPOT | PyCaret | Lale | AutoKeras | HuggingFace |
|---|---|---|---|---|---|---|---|---|
| Classification | ✅ | ✅ | — | — | ✅ | — | — | ✅ |
| Regression | ✅ | ✅ | — | — | ✅ | — | — | ✅ |
| Clustering | — | — | — | — | ✅ | — | — | — |

### Computer Vision

| Task | AutoGluon | FLAML | H2O AutoML | TPOT | PyCaret | Lale | AutoKeras | HuggingFace |
|---|---|---|---|---|---|---|---|---|
| Image Classification | ✅ | — | — | — | — | — | ✅ | — |
| Multi-Label Classification | ✅ | — | — | — | — | — | ✅ | — |
| Object Detection | ✅ β | — | — | — | — | — | — | — |
| Image Segmentation | ✅ β | — | — | — | — | — | — | — |

### Multimodal

| Task | AutoGluon | FLAML | H2O AutoML | TPOT | PyCaret | Lale | AutoKeras | HuggingFace |
|---|---|---|---|---|---|---|---|---|
| Classification | ✅ | — | — | — | — | — | — | — |
| Regression | ✅ | — | — | — | — | — | — | — |

Notes:

- Object Detection and Image Segmentation (AutoGluon) are **beta**.
- Multimodal training is natively supported only through AutoGluon in this interface (the UI warns if another framework is selected).
- If a `(category, task)` pair is missing from the map, `get_framework_options()` falls back to `["FLAML"]`.

---

## 5. User Guide

The sidebar (`src/navigation.py`, `NAV_ITEMS`) exposes 6 pages:

| Sidebar item | Internal page | Purpose |
|---|---|---|
| 🏠 Overview | Data Upload | Upload datasets into the DVC Data Lake + quick preview |
| 🗄️ Data | Data Exploration | Auto-EDA profiling reports |
| ⚙️ AutoML | Training | Configure and launch AutoML experiments |
| 🧪 Experiments | Experiments | Live dashboard of background runs (logs, metrics, exports) |
| 📦 Registry & Deploy | Prediction | Load models, batch/manual prediction, XAI |
| 📈 Monitoring | History (MLflow) | Run comparison, Model Registry, deployment packages |

The sidebar also hosts the optional **DagsHub integration** panel (see [MLOps](#7-mlops)).

### End-to-end flow

#### Step 1 — Upload data (🏠 Overview)

Two tabs:

- **Tabular, Text & Sequential (CSV/Excel):** upload `.csv`/`.xlsx`/`.xls`, optionally check *"This file has no header row"* (columns become `col_0, col_1, …`), choose a file prefix, then **Process & Save**. The file lands in `data_lake/raw/`, is registered with DVC (`dvc init` / `dvc add` via subprocess, with MD5 fallback), and its content hash is shown.
- **Computer Vision Data (Images/ZIP):** upload multiple PNG/JPG files or a single ZIP; images are extracted to `data_lake/images/<dataset_name>_<timestamp>/` and DVC-tracked.

Below the uploaders, a **Preview & Profiling** section shows dataset overview cards (rows, columns, missing %, memory) and tabs for preview, missing values, data types, and per-column distributions.

#### Step 2 — Auto-EDA (🗄️ Data)

Select a Data Lake dataset and click **Generate Auto-EDA Report**. This uses `ydata-profiling` + `streamlit-pandas-profiling` — both **optional dependencies**; the page shows an explicit error if they are not installed. A health alert warns when overall missing values exceed 5%. Large datasets (> 10,000 rows) automatically use the minimal profiling mode.

#### Step 3 — Configure training (⚙️ AutoML)

1. **Dataset selection:** pick Training (required), Validation (optional), Test/Holdout (optional) from the Data Lake. Each loaded file's DVC hash is recorded. An expander exposes per-file *no header row* options.
2. **Splitting & validation strategy:** choose **Random**, **Manual** (a column containing `train`/`val`/`test` markers), or **Chronological** (sort by a date column). Configure test percentage, then validation as simple holdout or **Cross-Validation** (K folds, 2–10).
3. **AutoML configuration:** select **Data Category** (5 categories), then **Task Type** (filtered per category), then **Framework** (filtered per category+task from the matrix). Target selection adapts:
   - CV tasks: target fixed to `label` (directory-structure labels).
   - Anomaly Detection / Clustering: no target (unsupervised).
   - Multi-Label / Multi-Task: multi-select of ≥ 2 target columns.
4. **Data characteristics flags (Tabular):** 📅 *Contains Temporal / Time Series Data* (date column + forecast horizon), 📝 *Contains Text / NLP Data* (text column selection), and *Self-Training / Semi-Supervised* for classification. For Multimodal, text and image-path columns are suggested automatically (`infer_multimodal_columns`).
5. **Global parallelism:** an ⚡ *Parallelism (n_jobs)* expander — Auto (all cores, `n_jobs=-1`) or Manual slider up to your CPU count. Applies to FLAML, TPOT, and PyCaret.
6. **Framework configuration block** — see [Section 6](#6-framework-configuration-reference).
7. **Launch options:** *Enable Strict CV (Data Leakage Prevention)* (default on) and an optional **Deep Feature Synthesis (DFS)** expander (depth 1–3, requires `featuretools`).

Clicking **🚀 Start Training** runs the optional preprocessing (`src/processor.py`) and queues one background experiment per target via the orchestrator. You can immediately start another training.

#### Step 4 — Monitor / cancel (🧪 Experiments)

The dashboard auto-refreshes every 5 seconds (Streamlit fragment) while runs are active. Per experiment you get:

- Status cards (Running / Completed / Failed / Cancelled), status badges, elapsed timer.
- **⛔ Cancel** (graceful stop via `stop_event`), **🗑️ Delete**, **🔮 Predict** (load into session), **📋 Register** (push to MLflow Model Registry).
- **📦 Export to ONNX** and **🚀 Push to Hugging Face** buttons for completed runs.
- A **Training Pipeline** visualizer (steps inferred from logs by `src/pipeline_parser.py`) and tabs: 📋 Logs (color-coded), 📈 Metrics (from MLflow), 🔬 Pipeline Inspector (AutoGluon leaderboard, FLAML best estimator/config, H2O leaderboard, TPOT best pipeline, or live telemetry while running), 🔍 MLflow (params/metrics/artifacts/metadata), 💻 Code & Deploy.

#### Step 5 — Compare runs (📈 Monitoring)

Select an experiment node, then multi-select runs to compare metrics side by side (bar chart), register a model in the MLflow Registry under a chosen name, or generate a FastAPI deployment package for any selected run. Cache statistics show the 5-minute TTL cycle.

#### Step 6 — Predict (📦 Registry & Deploy)

Model sources: **Current session model**, **Load from MLflow runs** (framework + Run ID), or **Load from ONNX / Hugging Face** (local `.onnx` upload, or download a file from a Hub repo — ONNX files are loaded into an inference session).

Input modes:

- **Batch Prediction (CSV/Excel):** upload a file, execute, download `predictions.csv`.
- **Real-time Prediction (Manual Entry):** widgets are generated per feature from the training data schema.

#### Step 7 — Explain (XAI)

- **Tabular (SHAP):** for single-target tabular classification/regression, *🧠 Explain Prediction (SHAP)* builds background data from the training set (max 100 samples) and renders a waterfall/summary plot. Requires the optional `shap` package.
- **Computer Vision:** *👁️ Explain AI Decision (Saliency Map)* computes a model-agnostic **occlusion saliency map** (sliding 30×30 black window, step 15) highlighting the image regions most important for the prediction.

#### Step 8 — Export

- **ONNX export** (Experiments page) — exports to `models/<run_name>.onnx` and logs it to MLflow; needs `onnx`/`onnxruntime`.
- **Hugging Face Hub push** — repository ID + token; uploads the local model or the MLflow artifact via `HuggingFaceService.upload_model`.
- **Consumption code** — a ready-to-run Python snippet per framework (`src/code_gen_utils.generate_consumption_code`: autogluon, flaml, h2o, tpot, pycaret, lale), downloadable as `consume_model.py`.
- **White-box notebook** — every successful run automatically generates a reproducible Jupyter notebook (`src/notebook_generator.py`), logged as an MLflow artifact.

#### Step 9 — FastAPI deployment package

From the Experiments page (💻 Code & Deploy tab) or History page, generate a package in `deploy_<id>/` containing `main.py` (FastAPI), `requirements.txt`, `Dockerfile`, and `README.md`. See [Section 8](#8-deployment) for how to run it.

#### Step 10 — Maintenance & cleanup (🧪 Experiments → Maintenance expander / 📈 Monitoring sidebar)

- **🧹 Clear Local Models** — empties the `models/` folder (safe if runs are synced to MLflow).
- **🔥 Reset MLflow (mlruns)** — deletes the local `mlruns/` store (destructive).
- **Disk monitoring** — free/used GB indicator with color thresholds.
- History page sidebar: **Hard Reset MLflow** (repairs tracking by removing `mlruns/`) and **Clear Python MLflow Cache** (in-memory TTL cache only).

---

## 6. Framework Configuration Reference

All parameters below are verified against the configuration blocks in `app.py` and the training signatures in `src/*_utils.py`.

### Common to all frameworks

| Parameter | UI control | Default | Notes |
|---|---|---|---|
| Seed | Number input | `42` | Reproducibility seed passed to engines |
| n_jobs | ⚡ Parallelism expander (Auto / Manual slider, max = CPU count) | Auto ⇒ `-1` | Applied to FLAML, TPOT, PyCaret |
| Strict CV | Checkbox | `True` | Bypasses stateful global transformations to prevent leakage |
| DFS depth | Expander slider 1–3 | `1` (disabled by default) | Requires `featuretools` |

### AutoGluon (`src/autogluon_utils.py` → `train_model`)

| Parameter | UI control | Default / Range | Notes |
|---|---|---|---|
| Time limit | *Enable Time Limit* checkbox + slider | 60 s, range 30–3600; disabled ⇒ `None` (train all models) | Seconds |
| Presets | Selectbox | `medium_quality` | Options: `medium_quality`, `best_quality`, `high_quality`, `good_quality`, `optimize_for_deployment` |
| cv_folds | Global split section | 0 (engine default) | From the Cross-Validation strategy |
| task_type / data_category | Auto | From page selections | Routes Tabular vs Multimodal vs CV code paths |
| multimodal text/image columns | Multiselect (Multimodal category) | Heuristic suggestions | Used by `MultiModalPredictor` |

### FLAML (`src/flaml_utils.py` → `train_flaml_model`)

| Parameter | UI control | Default / Range | Notes |
|---|---|---|---|
| Time budget | *Enable Time Limit* + slider | 60 s, range 30–3600; disabled ⇒ `None` | Seconds |
| Task | Auto-synced from task type | — | `classification` / `regression` / `ts_forecast` (Forecast) / `rank` (Ranking) |
| Metric | Selectbox (context-aware) | `auto` | Binary: `auto, accuracy, roc_auc, f1, log_loss`; Multiclass: `auto, accuracy, macro_f1, micro_f1, roc_auc_ovr, roc_auc_ovo, log_loss`; Regression: `auto, rmse, mae, r2, mape` |
| Estimators | Multiselect | `['lgbm', 'rf']` | Options: `lgbm, rf, catboost, xgboost, extra_tree, lrl1, lrl2`; empty ⇒ `'auto'` |
| n_jobs | Global parallelism | `-1` | Passed through |

### H2O AutoML (`src/h2o_utils.py` → `train_h2o_model`)

> Requires Java 11+. The module verifies Java before training and the cluster starts with `h2o.init(max_mem_size="4G", nthreads=-1)`.

| Parameter | UI control | Default / Range | Notes |
|---|---|---|---|
| Max runtime | *Enable Time Limit* + slider | 300 s, range 60–3600; disabled ⇒ `0` (until max models) | Seconds |
| Max models | Slider | 10, range 5–50 | |
| nfolds | Slider | 3, range 2–10 | Overridden by the global CV folds when set |
| Balance classes | Checkbox | `True` | |
| Exclude algorithms | Multiselect | `[]` | Options: `DeepLearning, GLM, GBM, DRF, XGBoost, GLRM` |
| Sort metric | Internal | `None` | Engine default |

### TPOT (`src/tpot_utils.py` → `train_tpot_model`)

| Parameter | UI control | Default / Range | Notes |
|---|---|---|---|
| Generations | Slider | 5, range 1–20 | Genetic evolution generations |
| Population size | Slider | 20, range 10–100 | |
| CV folds | Slider | 5, range 2–10 | Overridden by the global CV folds when set |
| Max time | *Enable Time Limit* + slider | 30 min, range 5–120; disabled ⇒ `None` | Minutes |
| Max time per evaluation | Slider | 5 min, range 1–20 | Minutes |
| Verbosity | Slider | 2, range 0–3 | |
| Configuration | Selectbox (Advanced expander) | `TPOT light` | Options: `TPOT light, TPOT MDR, TPOT sparse, TPOT NN` |
| TF-IDF max features | Number input (Advanced) | 500, range 100–10000 | Text feature dimensions |
| N-gram max size | Slider (Advanced) | 2, range 1–3 | Yields `tfidf_ngram_range=(1, n)` |
| Scoring | Selectbox (Advanced) | Task-dependent | Classification: `accuracy, balanced_accuracy, f1_macro, f1_micro, f1_weighted, roc_auc_ovr, roc_auc_ovo, precision_macro, recall_macro`; Regression: `neg_mean_squared_error, neg_root_mean_squared_error, neg_mean_absolute_error, r2, explained_variance` |
| n_jobs | Global parallelism | `-1` | |

Outputs: best pipeline exported as `tpot_models/best_pipeline_<run_name>.py` plus a `model_info_<run_name>.txt`.

### PyCaret (`src/pycaret_utils.py` → `run_pycaret_experiment`)

> Requires Python 3.11.

| Parameter | UI control | Default / Range | Notes |
|---|---|---|---|
| Time limit (tuning iterator limit) | *Enable Tuning Iterator Limit* + slider | 300 s, range 60–1200; disabled ⇒ `None` | Pseudo-time limit impacting `n_iter` |
| Forecast horizon (`fh`) | Number input (Time Series tasks) | 12 | Only for Time Series Forecasting |
| Seasonal period | Number input (Time Series tasks) | 12 | e.g. 12 for monthly, 7 for daily |
| task_type | Auto | From page selection | Routes classification/regression/time-series/anomaly/clustering setup |
| n_jobs | Global parallelism | `-1` | |

### Lale (`src/lale_utils.py` → `run_lale_experiment`)

> Requires Python 3.11.

| Parameter | UI control | Default / Range | Notes |
|---|---|---|---|
| Tune limit | *Enable Tune Limit* + slider | 120 s, range 60–600; disabled ⇒ `None` | Caps the internal Hyperopt search |
| task_type | Auto | From page selection | |

### AutoKeras (`src/autokeras_utils.py` → `run_autokeras_experiment`)

| Parameter | Source | Default | Notes |
|---|---|---|---|
| Time limit | Launch config | 60 s | Seconds of NAS search |
| task_type | Page selection | `Computer Vision - Image Classification` | Image Classification / Multi-Label Classification |
| valid_data | Split section | Optional | |

Requires TensorFlow. The module raises a clear `ImportError` when TensorFlow/AutoKeras is missing.

### HuggingFace (`src/huggingface_utils.py` → `run_huggingface_experiment`)

| Parameter | Source | Default | Notes |
|---|---|---|---|
| time_limit | Launch config | 60 s | |
| task_type | Page selection | `Classification` | Text classification/regression |
| Authentication | Token input or `HUGGINGFACE_TOKEN` env var | — | `HuggingFaceService` degrades gracefully if `huggingface_hub` is not installed |

---

## 7. MLOps

### MLflow tracking

- **Default backend: local file store.** On startup the app runs `heal_mlruns()` and `safe_set_experiment("Multi_AutoML_Project")` (`src/mlflow_utils.py`), which configures the tracking URI to `file:///<project_root>/mlruns`. No external server is required.
- **Per-framework experiments:** engine modules log into dedicated experiments (e.g. `H2O_Experiments`, `HuggingFace_Experiments`) while the UI uses `Multi_AutoML_Project`.
- **Auto-heal:** `heal_mlruns()` removes numeric experiment directories that are missing `meta.yaml` (a common cause of MLflow `MissingConfigException` crashes) and recreates `mlruns/.trash`. `safe_set_experiment()` retries once after healing if that error occurs.
- **Optional remote server:** set `MLFLOW_TRACKING_URI` (e.g. `http://localhost:5000` or a managed endpoint). Docker Compose does this automatically, pointing the app at the bundled MLflow server container.
- **Model Registry:** register any completed run via `mlflow.register_model("runs:/{run_id}/model", name)` — available from the Experiments page (📋 Register) and the History page (registration form).
- **TTL cache:** `src/mlflow_cache.py` caches run/experiment queries for **5 minutes (300 s TTL)** and keeps an `lru_cache`-backed experiment list, with manual cache-clear buttons on the History page.
- **What gets logged:** parameters, metrics, models, and artifacts — including the auto-generated white-box notebook and ONNX exports.

### DVC data lake

- `src/data_utils.py` drives DVC through the **CLI (`subprocess`)**: `dvc init` (lazily) and `dvc add` for every uploaded dataset or image directory.
- The DVC content hash (`md5` from the generated `.dvc` file) is surfaced in the UI and recorded with training runs; if DVC is missing or fails, the app **falls back to plain MD5 hashing** and keeps working.
- Tabular files live in `data_lake/raw/`; image datasets in `data_lake/images/`. The Data Lake is shared by all frameworks and by the prediction page.

### DagsHub integration

The sidebar (all pages) offers an optional **DagsHub Integration** panel:

1. Enable DagsHub, then enter **Username**, **Repository Name**, and **Access Token**.
2. On connect, the app sets `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` and calls `dagshub.init(repo_owner, repo_name, mlflow=True)`.
3. Requires the optional `dagshub` package; the UI shows an actionable error if it is missing.

### Environment variables

| Variable | Where used | Purpose |
|---|---|---|
| `HUGGINGFACE_TOKEN` | `src/huggingface_utils.py` | Fallback Hugging Face Hub token when none is typed in the UI |
| `MLFLOW_TRACKING_URI` | `docker-compose.yml`, `Dockerfile.autogluon_cv`, `Dockerfile.autokeras_cv` | Redirects MLflow to a remote/local server instead of `file:///mlruns` |
| `MLFLOW_TRACKING_USERNAME` | `app.py` (DagsHub connect) | Credentials for authenticated MLflow backends (set programmatically) |
| `MLFLOW_TRACKING_PASSWORD` | `app.py` (DagsHub connect) | Credentials for authenticated MLflow backends (set programmatically) |
| `STREAMLIT_SERVER_PORT` | `Dockerfile`, `docker-compose.yml`, `render.yaml` | Streamlit port (default `8501`) |
| `STREAMLIT_SERVER_ADDRESS` | `Dockerfile`, `docker-compose.yml`, `render.yaml` | Bind address (`0.0.0.0` in containers) |
| `PORT` | `Dockerfile` | Container port default (`8501`) |
| `JAVA_HOME` | `Dockerfile` | Points to the installed JDK for H2O |
| `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`, `PIP_NO_CACHE_DIR` | `Dockerfile` | Container hygiene defaults |
| `OMP_NUM_THREADS` | `Dockerfile.autogluon_cv` (set to `2`) | Prevents thread-locking in small containers |
| `GH_TOKEN` | `.github/workflows/build-electron.yml` (macOS job) | electron-builder publishing token (from `secrets.GITHUB_TOKEN`) |
| `MLFLOW_ALLOW_FILE_STORE` | `tests/conftest.py` (test-only) | Allows file-based MLflow stores during tests |

> Note: H2O cluster memory is not controlled by an environment variable in this codebase — it is fixed in `src/h2o_utils.py` via `h2o.init(max_mem_size="4G", nthreads=-1)` (2G when loading models for prediction).

---

## 8. Deployment

### Docker Compose (recommended local stack)

`docker-compose.yml` defines **two services**:

| Service | Image / Build | Published port | Notes |
|---|---|---|---|
| `autogluon-ui` | Builds the project `Dockerfile` | **8501** | Streamlit app; gets `MLFLOW_TRACKING_URI=http://mlflow-ui:5000`; mounts the project as `/app`; waits for MLflow health |
| `mlflow-ui` | `ghcr.io/mlflow/mlflow:v2.11.1` | **5000** | `mlflow server` with `./mlruns` mounted as backend store and artifact root |

```bash
docker compose up --build
# UI: http://localhost:8501  ·  MLflow: http://localhost:5000
```

> **Important:** the base image installs only `requirements.txt` (the lightweight core stack). Heavy frameworks (AutoGluon, H2O, TPOT, PyCaret, Lale, AutoKeras…) are **not** inside this image — extend the Dockerfile or install extras in the mounted volume if you need them inside containers.

### Standalone Dockerfile

- Base `python:3.12-slim`, installs `build-essential`, `libgomp1`, `libgl1`, `default-jre` + `default-jdk` (so H2O works), `curl`, `git`.
- `EXPOSE 8501` (Streamlit) and `5000` (MLflow), healthcheck on `/_stcore/health`.

```bash
docker build -t multi-automl-interface .
docker run -p 8501:8501 multi-automl-interface
```

### CV-specialized images

| Dockerfile | Base | Contents |
|---|---|---|
| `Dockerfile.autogluon_cv` | `python:3.10-slim` | CPU torch 2.1.0 + `mmcv==2.1.0` via OpenMIM, filtered requirements install with fallback to `autogluon==1.1.1 pandas scikit-learn streamlit mlflow`; `EXPOSE 8501`, `OMP_NUM_THREADS=2` |
| `Dockerfile.autokeras_cv` | `tensorflow/tensorflow:2.15.0` | Installs `autokeras` + `keras-tuner`, filtered requirements; `EXPOSE 8501` |

```bash
docker build -f Dockerfile.autogluon_cv -t multi-automl-agcv .
docker build -f Dockerfile.autokeras_cv -t multi-automl-akcv .
```

### Electron desktop app

Requires **Node.js 18+** (CI uses Node 20). `electron/main.js` spawns `python -m streamlit run app.py` headless on `127.0.0.1:8501` and loads it in a BrowserWindow (with retry and an `error_loading.html` fallback).

| Script | Effect |
|---|---|
| `npm install` | Install dev deps; `postinstall` runs `electron-builder install-app-deps` |
| `npm start` | Run Electron (expects Streamlit already running) |
| `npm run dev` | Runs Streamlit and Electron together (`concurrently` + `wait-on http://localhost:8501`) |
| `npm run streamlit` | Streamlit only (`--server.port 8501`) |
| `npm run build-win` | Windows installer — **NSIS** (guided install, desktop/start-menu shortcuts) |
| `npm run build-mac` | macOS **DMG** |
| `npm run build-linux` | Linux **AppImage** |

Installers are written to `dist/` (product name "Multi-AutoML Desktop", app ID `com.multi-automl.desktop`, `asar: false`).

### Render

`render.yaml` declares a `web` service named `multi-automl` (free plan) that builds the repository `Dockerfile` from the `main` branch of `https://github.com/PedroM2626/Multi-AutoML-Interface`, setting `STREAMLIT_SERVER_PORT=8501` and `STREAMLIT_SERVER_ADDRESS=0.0.0.0`. Deploy by connecting the repo in Render (Blueprint) — Render builds the Docker image for you.

### Hugging Face Spaces

A public demo lives at `https://huggingface.co/spaces/PedroM2626/Multi-AutoML-Interface`. **Demo only:** it is provided for quick visualization/testing — real experiments and reproducible workloads should run locally or in your own containers.

### Running a generated `deploy_<run_id>/` FastAPI package

Generated by `src/code_gen_utils.generate_api_deployment` (supported engines: autogluon, flaml, h2o, tpot, pycaret, lale). Package contents:

| File | Contents |
|---|---|
| `main.py` | FastAPI app: `GET /` → health check (`{"status": "running", "model": ..., "run_id": ...}`); `POST /predict` → accepts `{"data": [rows...]}` or a single row object, returns `{"predictions": [...]}`. Loads the model from MLflow artifacts at startup. |
| `requirements.txt` | `fastapi==0.104.1`, `uvicorn==0.24.0`, `pydantic==2.5.2`, `pandas==2.1.4`, `mlflow==2.9.2` + engine-specific pins (e.g. `tpot==0.12.2`, `pycaret==3.3.0`, `lale==0.9.1`…) |
| `Dockerfile` | `python:3.11-slim` with build essentials + JRE, `EXPOSE 8000`, runs `uvicorn main:app --host 0.0.0.0 --port 8000` |
| `README.md` | Run instructions |

```bash
# Local
cd deploy_<run_id>
pip install -r requirements.txt
python main.py                  # serves on http://localhost:8000

# Docker
docker build -t ml-api:<tag> .
docker run -p 8000:8000 ml-api:<tag>

# Example request
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"data": [{"feature1": 1.5, "feature2": "value"}]}'
```

> The generated service downloads its model from MLflow at startup, so the same MLflow store (or `MLFLOW_TRACKING_URI`) used during training must be reachable.

---

## 9. Testing & CI

### Test inventory (`tests/`)

| Group | Files |
|---|---|
| Core regression flows | `test_regression_flows.py` |
| GUI / interface | `test_streamlit_gui.py`, `test_interface_simulation.py` |
| Orchestrator & catalog | `test_orchestrator.py`, `test_task_catalog.py` |
| TPOT | `test_tpot_integration.py`, `test_tpot_large_data.py`, `test_tpot_nan_fix.py`, `test_tpot_sparse_fix.py`, `test_tpot_timeout_fix.py` |
| H2O | `test_h2o_integration.py`, `test_h2o_simulation.py`, `test_h2o_docker_simulation.py` |
| PyCaret / Lale | `test_pycaret_utils.py`, `test_lale_utils.py` |
| AutoGluon / CV | `test_autogluon_dispatch.py`, `test_cv_utils.py`, `test_empty_leaderboard.py` |
| External integrations | `test_external_integrations.py` |
| Simulation scripts (manual/dev) | `simulate_training.py`, `simulate_pycaret.py`, `simulate_lale.py` |
| Shared fixtures | `conftest.py` (isolated temporary `mlruns` store per test session, `MLFLOW_ALLOW_FILE_STORE=true`) |

### Running tests locally

```bash
# Quick gates (same as PR CI)
pytest -q tests/test_regression_flows.py tests/test_streamlit_gui.py

# Full suite (requires the corresponding optional frameworks)
pytest -q tests
```

### CI workflows

**`.github/workflows/ci.yml`** — triggers: push to `main`/`master`, pull requests, daily schedule (`cron: 0 3 * * *`), manual dispatch. Python 3.12, pip caching.

| Job | When | Steps |
|---|---|---|
| `quick-pr` | Every push/PR | Install `requirements-dev.txt` + force-reinstall runtime pins (`numpy==2.5.0 pandas==2.3.3 scikit-learn==1.9.0 mlflow==3.14.0 flaml==2.6.0 streamlit==1.58.0`) → `ruff check .` → `python -m compileall app.py run.py src tests` → `pytest -q tests/test_regression_flows.py tests/test_streamlit_gui.py` |
| `nightly-complete` | Schedule or manual dispatch | Same validation gates first, then optional full `requirements.txt` install and a best-effort `pytest -q tests` (both steps `continue-on-error`) |

**`.github/workflows/build-electron.yml`** — triggers: push to `main`/`master`, manual dispatch. 3-OS matrix (`windows-latest`, `macos-latest`, `ubuntu-latest`) with Node 20; runs `npm run build-win` / `build-mac` / `build-linux` and uploads `dist/*.exe`, `dist/*.dmg`, `dist/*.AppImage` artifacts (retention: 7 days).

### Developer quality gates

```bash
pip install -r requirements-dev.txt   # ruff==0.13.1, pytest==8.4.2
ruff check .                          # lint
python -m compileall app.py run.py src tests   # syntax gate
```

---

## 10. Project Structure

```
Multi-AutoML-Interface/
├── app.py                        # Streamlit application (~2,300 lines): UI, pages, config forms
├── run.py                        # Launcher that requires Python 3.11+ (re-execs with py -3.11 if the current interpreter is older)
├── src/                          # 25 modules (below)
│   ├── __init__.py               # Package marker
│   ├── autogluon_utils.py        # AutoGluon training (tabular/CV/multimodal), leaderboard, MLflow logging
│   ├── autokeras_utils.py        # AutoKeras NAS experiments for image tasks
│   ├── code_gen_utils.py         # Consumption-code snippets + FastAPI/Docker deployment generator
│   ├── data_utils.py             # File loading, DVC data lake, image/ZIP ingestion, hashing
│   ├── experiment_manager.py     # ExperimentEntry/ExperimentManager singleton (threads, queues, statuses)
│   ├── flaml_utils.py            # FLAML training and metric evaluation
│   ├── h2o_utils.py              # Java check, H2O cluster init, H2O AutoML training/loading
│   ├── huggingface_utils.py      # HF Hub service (list/upload/download) + experiment logging
│   ├── lale_utils.py             # Lale/Hyperopt pipeline experiments
│   ├── log_utils.py              # Queue-based logging setup and stdout redirection helpers
│   ├── mlflow_cache.py           # 5-minute TTL cache for MLflow queries
│   ├── mlflow_utils.py           # mlruns auto-heal + safe experiment configuration
│   ├── navigation.py             # Sidebar page registry (NAV_ITEMS) and state sync
│   ├── notebook_generator.py     # White-box reproducible Jupyter notebook generator
│   ├── onnx_utils.py             # ONNX export and InferenceSession loading
│   ├── orchestrator.py           # UniversalAutoMLOrchestrator: framework dispatch + thread queuing
│   ├── pipeline_parser.py        # Infers pipeline steps from logs; extracts TPOT best pipeline
│   ├── prediction_service.py     # Unified model loading and prediction across frameworks
│   ├── processor.py              # AutoMLDataProcessor: temporal/NLP/DFS feature engineering, strict CV
│   ├── pycaret_utils.py          # PyCaret experiments (cls/reg/ts/anomaly/clustering)
│   ├── task_catalog.py           # DATA_CATEGORIES, TASK_OPTIONS_BY_CATEGORY, TASK_FRAMEWORK_MAP
│   ├── tpot_utils.py             # TPOT genetic search, pipeline export to tpot_models/
│   ├── training_worker.py        # Thread entry point, log isolation, result normalization
│   ├── ui_state.py               # Streamlit session-state initialization
│   └── xai_utils.py              # SHAP explanations + occlusion saliency maps
├── tests/                        # Pytest suite + simulate_*.py scripts (see Section 9)
├── electron/                     # Desktop shell: main.js, preload.js, renderer.js, assets/icon.png
├── mlruns/                       # MLflow local tracking store (generated)
├── data_lake/                    # DVC-versioned datasets: raw/ + images/ (generated)
├── models/                       # Local model saves + ONNX exports (generated)
├── tpot_models/                  # TPOT pipeline/info exports (generated)
├── docs/                         # This documentation
├── .github/workflows/            # ci.yml (quick-pr + nightly-complete), build-electron.yml
├── Dockerfile                    # Core image (python:3.12-slim + Java)
├── Dockerfile.autogluon_cv       # AutoGluon CV image (python:3.10-slim, torch 2.1, mmcv)
├── Dockerfile.autokeras_cv       # AutoKeras CV image (tensorflow:2.15.0)
├── docker-compose.yml            # autogluon-ui:8501 + mlflow-ui:5000
├── render.yaml                   # Render cloud blueprint
├── package.json                  # Electron scripts + electron-builder targets
├── requirements.txt              # Lightweight core stack
├── requirements-compiled.txt     # Full pinned stack (pip-compile, Python 3.11)
├── requirements-dev.txt          # ruff + pytest
├── pyproject.toml                # Python project config
└── error_loading.html            # Electron fallback page when Streamlit is unreachable
```

---

## 11. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| **`RuntimeError: Java is not installed`** when starting H2O | H2O requires Java 11+. Install a JRE/JDK and set `JAVA_HOME`, run via Docker (the base image includes `default-jre`/`default-jdk`), or use AutoGluon/FLAML instead. The error message in `src/h2o_utils.py` lists all three options. |
| **Port conflict on 8501** (Streamlit) | Another Streamlit instance is running (or Electron's dev mode). Stop it, or start on another port: `streamlit run app.py --server.port 8502`. In Docker, remap: `-p 8502:8501`. |
| **Port conflict on 5000** (MLflow) | Change the compose mapping: `"5001:5000"` and update `MLFLOW_TRACKING_URI` accordingly. |
| **MLflow `MissingConfigException` / missing `meta.yaml`** | Corrupted experiment folders. The app auto-heals at startup (`heal_mlruns` removes numeric dirs without `meta.yaml`); you can also use *Hard Reset MLflow* on the History page or delete `mlruns/` manually. |
| **Stale run lists in History** | Results are cached for 5 minutes (`MLflowCache`). Use *Clear Python MLflow Cache* on the History page. |
| **`ModuleNotFoundError` for an engine** (autogluon, h2o, tpot, pycaret, lale, autokeras…) | Engines are optional and imported lazily — install the missing extra (Section 3). The failing run shows the import error in its log panel; the rest of the app keeps working (graceful degradation). |
| **SHAP explanation unavailable** | `pip install shap` (optional). Explanations are limited to single-target tabular classification/regression. |
| **Auto-EDA fails** | `pip install ydata-profiling streamlit-pandas-profiling` (optional). |
| **DVC messages ("DVC is not installed or not in PATH")** | Install `dvc` or ignore — the app falls back to MD5 hashing and remains functional. |
| **ONNX export error** | `pip install onnx onnxruntime`; a Data Lake dataset is required for shape inference. |
| **Memory errors during training** | Reduce `n_jobs` (Parallelism expander → Manual), shrink time limits/populations, or lower CV folds. H2O's cluster cap is fixed at `max_mem_size="4G"` in `src/h2o_utils.py` — edit that value to change it. DFS at depth ≥ 2 can consume massive RAM; keep depth at 1. |
| **Electron window shows the error page** | Streamlit did not start within ~20 retries. Ensure Python + Streamlit are installed and port 8501 is free; use `npm run dev` to watch both processes. |
| **Electron build fails** | Requires Node 18+ (CI uses 20). Delete `node_modules`/`dist` and rerun `npm install`, then `npm run build-win|mac|linux`. macOS builds use `GH_TOKEN`. |
| **Windows PowerShell: `&&` not recognized** | PowerShell v5 does not support `&&` as a statement separator — chain commands with `;` instead (e.g. `pip install dvc; dvc init`). |
| **Python 3.12 + PyCaret/Lale errors** | PyCaret and Lale require Python 3.11. Launch with `python run.py` / `py -3.11 -m streamlit run app.py` (run.py re-launches itself on the right interpreter automatically). |

---

## 12. Contributing & Roadmap

### Contributing

1. Fork the repository and create a feature branch.
2. Install dev tooling: `pip install -r requirements-dev.txt`.
3. Before opening a PR, run the quality gates (they mirror CI):
   ```bash
   ruff check .
   python -m compileall app.py run.py src tests
   pytest -q tests/test_regression_flows.py tests/test_streamlit_gui.py
   ```
4. Keep new task/framework support synchronized between `src/task_catalog.py` and the matrix in [Section 4](#4-task--framework-support-matrix).
5. Follow the existing engine-module pattern (`src/<engine>_utils.py` + orchestrator mapping) when adding integrations, and honor `stop_event`/`log_queue`/`telemetry_queue` in training signatures.

### Roadmap

- **Auto-sklearn** integration as an additional AutoML engine.
- **Advanced visualizations** for experiment comparison and telemetry.
- **Batch processing queue** for scheduling multiple datasets/tasks end-to-end.

---

*This documentation was generated from the source code. If behavior differs, the code is authoritative — in particular `src/task_catalog.py` for the support matrix and `src/code_gen_utils.py` for deployment packages.*
