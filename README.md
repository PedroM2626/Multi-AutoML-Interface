# Multi-AutoML Interface

![Version](https://img.shields.io/badge/version-5.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/PedroM2626/Multi-AutoML-Interface)
![License](https://img.shields.io/badge/license-MIT-green)

📘 Full documentation: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)

**A unified interface for experimenting with AutoML, allowing you to compare multiple frameworks (AutoGluon, FLAML, H2O AutoML, TPOT, PyCaret, Lale, AutoKeras, HuggingFace) with integrated MLOps via MLflow.**

---

**Important:** The linked Hugging Face Spaces demo is provided for testing and visualization only — this project is intended to be run locally for real experiments and production use. See the Quick Start section below to run the application on your machine.

## 🆕 What's New (Recent)

- **White-box notebook generation**: every training run can be exported as a reproducible Jupyter notebook (`src/notebook_generator.py`).
- **HuggingFace experiment integration**: fine-tune transformer models for text tasks directly from the UI (`src/huggingface_utils.py`).
- **Temporal & text preprocessing**: tabular datasets support "Contains Temporal Data" (chronological splits, lag/rolling features) and "Contains Text / NLP Data" (automatic TF-IDF vectorization).
- **Forecast task type**: dedicated Forecast task replacing the old Time Series task, wired across the supported frameworks.
- **Multi-Task Classification**: predict multiple targets concurrently; **Semi-Supervised Classification** via `SelfTrainingClassifier` over unlabeled samples (`-1`/`NaN`).
- User-selectable parallelism (`n_jobs`), headerless CSV/Excel uploads, and Streamlit caching performance improvements.

## 🎯 **Overview**

The Multi-AutoML Interface is a web/desktop application that simplifies the use of AutoML frameworks, enabling:

- **Side-by-side comparison** of 8 different AutoML engines
- **Integrated MLOps** with tracking via MLflow (local file-based store out of the box) and DVC data versioning
- **Unified interface** for training, evaluation, and prediction across 5 data categories: Tabular, Sequential, Text, Computer Vision, and Multimodal
- **Flexible deployment** (web, Docker, desktop, Hugging Face Spaces)
- **Deployment-package generation**: one-click FastAPI + Docker serving package for any trained model
- **Detailed metrics and logging**

---

## ✨ **Key Features**

### 🤖 **Supported AutoML Frameworks:**
- **AutoGluon** (Amazon) - Tabular, multimodal, and computer-vision AutoML
- **FLAML** (Microsoft) - Fast and efficient economical AutoML
- **H2O AutoML** (H2O.ai) - Robust and comprehensive enterprise AutoML
- **TPOT** - Genetic-algorithm pipeline optimization
- **PyCaret** - End-to-end low-code ML platform
- **Lale** (IBM) - Scikit-Learn compatible topology search with Hyperopt
- **AutoKeras** - AutoML for deep learning based on Keras
- **HuggingFace** - Transformer fine-tuning for text tasks

### 📊 **Integrated MLOps & Dashboard:**
- **Explainable AI (XAI)**: SHAP for tabular data and Saliency Maps (Occlusion) for Computer Vision.
- **Auto-EDA & Data Health**: profiling via `ydata-profiling`.
- **Live Experiments Dashboard**: monitor concurrent training runs with real-time logs and metrics (Streamlit Fragments).
- **Multi-Concurrent Training**: launch all 8 engines simultaneously via background training workers, with graceful cancellation.
- **Complete MLflow tracking**: metrics, parameters, and artifacts in a local `mlruns/` store.
- **Automatic Code & Notebook Generation**: Python consumption snippets and reproducible notebooks per run.
- **One-Click API Deployment**: generate a complete FastAPI + Docker package for any model (`src/code_gen_utils.py`).
- **ONNX Integration**: model export/import via ONNX (`src/onnx_utils.py`).
- **Advanced Prediction**: batch processing via file upload or manual entry form.
- **Unified ML Task Selector**: choose the data category first, then a task type valid for that family; only supporting engines are shown.

### 🖥️ **Multi-Deploy:**
- **Web interface** (Streamlit), **Docker container** (Compose with MLflow server), **Desktop app** (Electron), **Hugging Face Spaces** (live demo), **Render** (`render.yaml`).

Note: The Hugging Face Spaces entry above links to a demo deployment provided for quick preview and visualization. For reproducible experiments and real workloads, run the project locally using the Quick Start instructions.

---

## 🏗️ **Architecture**

```
┌─────────────────────────┐      ┌──────────────────────────────┐
│        Frontend         │      │        src/ backend          │
│                         │      │                              │
│ • Streamlit UI (app.py) │─────►│ • orchestrator.py            │
│ • Electron wrapper      │      │ • training_worker.py         │
│                         │      │ • experiment_manager.py      │
└─────────────────────────┘      └──────────────┬───────────────┘
                                                ▼
                     ┌─────────────────────────────────────────┐
                     │               ML Engines                │
                     │ AutoGluon • FLAML • H2O AutoML • TPOT   │
                     │ PyCaret • Lale • AutoKeras • HuggingFace│
                     └──────────────────┬──────────────────────┘
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
┌──────────────────────────┐ ┌────────────────────┐ ┌──────────────────────────┐
│    Experiment Store      │ │  Data Versioning   │ │    Deployment Targets    │
│ • MLflow local (mlruns/) │ │ • DVC (data_lake/) │ │ • Docker • Render        │
│ • Artifacts • Notebooks  │ │                    │ │ • HF Spaces • Generated  │
│                          │ │                    │ │   FastAPI packages       │
└──────────────────────────┘ └────────────────────┘ └──────────────────────────┘
```

Note: FastAPI is not the application backend — the app is a Streamlit application. FastAPI appears only inside the generated per-model deployment packages.

---

## 🚀 **Quick Start**

### 📋 **Prerequisites:**
- **Python 3.11** — minimum for the full framework stack (`run.py` requires Python 3.11+ and re-launches on 3.11 when needed: PyCaret and Lale require Python 3.11). The CI/Docker core path uses Python 3.12 for the lightweight stack.
- **Node.js 18+** (for the Electron desktop app; CI builds with Node 20)
- **Java 11+** (only for H2O AutoML)
- **Git**

### 🔧 **Installation:**

```bash
# 1. Clone
git clone https://github.com/PedroM2626/Multi-AutoML-Interface.git
cd Multi-AutoML-Interface

# 2. Create and activate a Python 3.11 virtual environment
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install the lightweight core stack
pip install -r requirements.txt
```

`requirements.txt` installs only the core stack (Streamlit, MLflow, FLAML, FastAPI, scikit-learn, XGBoost, and supporting libraries). `requirements-compiled.txt` provides the full pinned stack compiled with pip-compile on Python 3.11.

#### **Optional framework backends:**

The heavy AutoML frameworks are **lazy-imported** and degrade gracefully when not installed — the app runs with any subset. Install what you need: `autogluon`, `h2o` (requires Java 11+), `tpot`, `pycaret`, `lale`, `autokeras`, `transformers`/`datasets`/`huggingface_hub`, `shap` (XAI), `ydata-profiling` (Auto-EDA), `dvc` (data versioning), `onnxruntime` (ONNX export) — e.g. `pip install autogluon pycaret`.

#### **Run the Application:**
```bash
# Recommended: auto-selects a Python 3.11 interpreter and starts Streamlit
python run.py

# Or directly (inside a Python 3.11 environment)
streamlit run app.py
```

MLflow needs no setup: tracking is **local and file-based** (`mlruns/`) out of the box. An MLflow tracking server is entirely optional (see Docker section). Alternative run modes: `npm install && npm run dev` (desktop app, Node.js 18+) or `docker-compose up` (Streamlit app + MLflow server).

---

## 📖 **User Guide**

### 🎯 **Basic Workflow:**

#### **1. Data Upload & Exploration:**
- CSV/Excel uploads (train mandatory; validation/test optional), automatic type detection
- **Auto-EDA**: profiling reports via `ydata-profiling`
- **Automatic Data Lake**: processed data is copied to `data_lake/` and versioned with DVC

#### **2. Experiment Configuration:**
- **Data Category + Task Type**: choose one of the 5 categories — Tabular, Sequential, Text, Computer Vision, Multimodal — then a compatible task type.
- **Framework Agnostic**: AutoGluon, FLAML, H2O AutoML, TPOT, PyCaret, Lale, AutoKeras, HuggingFace.
- **ONNX Integration**: universal model export/import; **HF Hub**: publish models with one click.
- **Advanced parameters**: seed, time limits, folds, TF-IDF feature caps, CV, forecasting horizon, etc.

#### **Task Type Support Matrix (Current Implementation)**

Generated from `TASK_FRAMEWORK_MAP` in `src/task_catalog.py`.

Legend: ✅ = implemented, ⚠️ = partial/beta path, ❌ = not implemented.

| Data Category | Task Type | AutoGluon | FLAML | H2O AutoML | TPOT | PyCaret | Lale | AutoKeras | HuggingFace |
|---|---|---|---|---|---|---|---|---|---|
| Tabular | Classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Tabular | Regression | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Tabular | Multi-Label Classification | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Tabular | Multi-Task Classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Tabular | Semi-Supervised Classification | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Tabular | Anomaly Detection | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Tabular | Clustering | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Tabular | Forecast | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Tabular | Ranking | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Sequential | Classification | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Sequential | Regression | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Sequential | Forecast | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Sequential | Anomaly Detection | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Sequential | Clustering | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Text | Classification | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Text | Regression | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Text | Clustering | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Computer Vision | Image Classification | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Computer Vision | Multi-Label Classification | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Computer Vision | Object Detection | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Computer Vision | Image Segmentation | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multimodal | Classification | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multimodal | Regression | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

Notes:
- CV Object Detection and Image Segmentation are exposed through AutoGluon but should be treated as beta until broader test coverage is added.
- Tabular Anomaly Detection and Clustering run through PyCaret's unsupervised modules (no target column required).
- For framework-native capabilities beyond this matrix, see [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

#### **3. Training, Results & Prediction:**
- **Experiments Tab**: live dashboard with real-time logs; training runs in background workers; cancel at any time.
- **Results**: comparative leaderboards, side-by-side run comparison, model registry, and one-click FastAPI deployment-package generation.
- **Prediction**: batch inference via CSV/Excel upload or a dynamic manual entry form.
- **Explainability**: "Explain Prediction (SHAP)" for tabular, "Explain AI Decision (Saliency Map)" for CV, plus generated consumption code.
- **Maintenance**: integrated cleanup for `models/` and `mlruns`, disk-space warnings, automatic MLflow sync of artifacts.

---

## 🐳 **Deploy with Docker**

```bash
# Build image
docker build -t multi-automl:latest .

# Start all services (Streamlit app + MLflow server)
docker-compose up -d

# Logs / Stop
docker-compose logs -f
docker-compose down
```

**Ports:** `8501` Streamlit UI, `5000` MLflow UI. H2O AutoML runs in-process inside the app container — no separate H2O cluster port is exposed.

---

## 🖥️ **Desktop App (Electron)**

```bash
# Install Node dependencies (Node.js 18+)
npm install

# Development mode (starts Streamlit on 8501, then opens Electron)
npm run dev

# Production builds
npm run build-win    # Windows (NSIS installer)
npm run build-mac    # macOS (DMG)
npm run build-linux  # Linux (AppImage)
```

The Electron build is also produced automatically by the `Build Desktop App` workflow in `.github/workflows/build-electron.yml` (Windows, macOS, Linux on Node 20).

---

## 📊 **Framework Comparison (Qualitative)**

| Framework | Typical Strengths | Typical Trade-offs |
|---|---|---|
| **AutoGluon** | Strong out-of-the-box accuracy; broadest task coverage (tabular, text, CV, multimodal) | Heavier install and memory footprint |
| **FLAML** | Very fast, economical search; lightweight | Smaller model zoo |
| **H2O AutoML** | Mature enterprise tabular AutoML | Requires Java; JVM memory overhead |
| **TPOT** | Interpretable exported pipelines (genetic search) | Slow search for large budgets |
| **PyCaret** | Widest task surface in this project (anomaly, clustering, semi-supervised) | Requires Python 3.11 |
| **Lale** | sklearn-compatible topology search | Classification/regression focus |
| **AutoKeras** | Deep-learning CV AutoML | GPU/TF stack required |
| **HuggingFace** | Transformer fine-tuning for text | GPU recommended; slower training |

No hardcoded benchmark numbers are published: results depend strongly on dataset, budget, and hardware. Use the in-app leaderboard to compare engines on your own data.

---

## 🔧 **Troubleshooting**

- **"Java not found" (H2O)**: set `JAVA_HOME` to a Java 11+ installation (e.g. `set JAVA_HOME="C:\Program Files\Java\jdk-11"` on Windows, `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk` on Linux).
- **"Python 3.11 not found" (run.py)**: PyCaret and Lale require Python 3.11. Install it and retry, or run `py -3.11 -m streamlit run app.py` directly.
- **"Port already in use"**: start on another port — `streamlit run app.py --server.port 8502`.
- **MLflow errors / missing `mlruns`**: the store is auto-healed at startup (malformed experiment folders are cleaned and recreated). If problems persist, remove the offending folder under `mlruns/` and restart.

---

## 🧪 **Testing**

```bash
# Install dev tooling (ruff, pytest, ...)
pip install -r requirements-dev.txt

# Lint (critical rules) + syntax check
ruff check .
python -m compileall app.py run.py src tests

# Quick suite (mirrors the CI quick-pr job)
pytest -q tests/test_regression_flows.py tests/test_streamlit_gui.py

# Full suite
pytest -q tests
```

**CI (`.github/workflows/ci.yml`):**
- **quick-pr**: on push/PR — ruff lint, compile check, and the quick regression suite.
- **nightly-complete**: scheduled/dispatched — quick gates plus a best-effort full `pytest -q tests` run with the optional integration stack.
- **Build Desktop App** (`.github/workflows/build-electron.yml`): builds the Electron installers for Windows, macOS, and Linux.

---

## 📁 **Project Structure**

```
Multi-AutoML-Interface/
├── 📁 src/                         # Main source code
│   ├── 📄 autogluon_utils.py       # AutoGluon integration
│   ├── 📄 autokeras_utils.py       # AutoKeras integration
│   ├── 📄 code_gen_utils.py        # Consumption code + FastAPI deployment packages
│   ├── 📄 data_utils.py            # Data processing & DVC integration
│   ├── 📄 experiment_manager.py    # Experiment lifecycle management
│   ├── 📄 flaml_utils.py           # FLAML integration
│   ├── 📄 h2o_utils.py             # H2O AutoML integration
│   ├── 📄 huggingface_utils.py     # HuggingFace experiment integration
│   ├── 📄 lale_utils.py            # Lale integration
│   ├── 📄 log_utils.py             # Logging utilities
│   ├── 📄 mlflow_cache.py          # MLflow query caching
│   ├── 📄 mlflow_utils.py          # MLflow helpers and mlruns auto-healing
│   ├── 📄 navigation.py            # UI navigation helpers
│   ├── 📄 notebook_generator.py    # White-box notebook generation
│   ├── 📄 onnx_utils.py            # ONNX export/import
│   ├── 📄 orchestrator.py          # Framework dispatch orchestrator
│   ├── 📄 pipeline_parser.py       # Pipeline parsing helpers
│   ├── 📄 prediction_service.py    # Prediction service
│   ├── 📄 processor.py             # Data preprocessing pipeline
│   ├── 📄 pycaret_utils.py         # PyCaret integration
│   ├── 📄 task_catalog.py          # Data categories & task/framework map
│   ├── 📄 tpot_utils.py            # TPOT integration
│   ├── 📄 training_worker.py       # Background training workers
│   ├── 📄 ui_state.py              # Streamlit session-state management
│   └── 📄 xai_utils.py             # SHAP and Saliency Map integration
├── 📁 tests/                       # Automated tests (regression, integrations, simulations)
├── 📁 electron/                    # Desktop app (main.js, preload.js, renderer.js)
├── 📁 docs/                        # Extended documentation
├── 📁 data_lake/                   # DVC-versioned dataset lake
├── 📁 .github/workflows/           # CI: ci.yml, build-electron.yml
├── 📁 deploy_[run_id]/             # Generated FastAPI deployment packages (at runtime)
├── 📄 app.py                       # Streamlit application entry
├── 📄 run.py                       # Launcher (requires Python 3.11+, re-launches on 3.11 when needed)
├── 📄 pyproject.toml               # Project metadata & tooling config
├── 📄 requirements.txt             # Lightweight core dependencies
├── 📄 requirements-dev.txt         # Dev tooling (ruff, pytest)
├── 📄 requirements-compiled.txt    # Full pinned stack (pip-compile, Python 3.11)
├── 📄 render.yaml                  # Render deployment config
├── 🐳 Dockerfile                   # Docker configuration
├── 🐳 Dockerfile.autogluon_cv      # AutoGluon computer-vision image (Python 3.10, torch + mmcv)
├── 🐳 Dockerfile.autokeras_cv      # AutoKeras computer-vision image (TensorFlow)
├── 🐳 docker-compose.yml           # Streamlit app + MLflow server
└── 📄 package.json                 # Electron desktop app config
```

---

## 🤝 **Contributing**

1. **Fork and clone** the repository, then create a branch: `git checkout -b feature/new-feature`.
2. **Develop**: follow existing code style, add tests, document changes.
3. **Run the quality gates**: `ruff check .`, `python -m compileall app.py run.py src tests`, `pytest -q tests/test_regression_flows.py tests/test_streamlit_gui.py`.
4. **Commit and push** using Conventional Commits (`feat:`, `fix:`, ...), then open a Pull Request describing changes and linking related issues.

Guidelines: PEP 8 (enforced via ruff), Conventional Commits, clear Markdown in English.

---

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Credits and Acknowledgements**

### 🤖 **Frameworks:**
- **AutoGluon** - Amazon Web Services
- **FLAML** - Microsoft Research
- **H2O AutoML** - H2O.ai
- **TPOT** - TPOT contributors
- **PyCaret** - PyCaret contributors
- **Lale** - IBM
- **AutoKeras** - AutoKeras contributors
- **HuggingFace** - Hugging Face

### 🛠️ **Technologies:**
- **Streamlit** - Web interface
- **MLflow** - Experiment tracking
- **Electron** - Desktop app
- **Docker** - Containerization

---

## 🗺️ **Future Roadmap**

### 🚀 **Upcoming Features**
- [ ] **Auto-sklearn** (meta-learning)
- [ ] **Advanced visualizations** (3D clusters, interactive ROC)
- [ ] **Batch processing queue** (Distributed training)

---

### 🌐 **Live Demo:**
[Hugging Face Spaces - Multi-AutoML Interface](https://huggingface.co/spaces/PedroM2626/Multi-AutoML-Interface) — demo only (visualization/testing). Run locally for real experiments.

---

*Developed by Pedro Morato Lahoz*
