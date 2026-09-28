# Changelog

All notable changes to Multi-AutoML Interface are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Release tags are `vMAJOR.MINOR.PATCH` and must match `version` in `package.json`;
pushing such a tag runs the `Release Desktop App` workflow, which builds the
Windows/macOS/Linux installers and attaches them to the GitHub Release.

## 5.0.0 - 2026-09-28

First tagged release. It publishes the tree at the `5.0.0` version bump (`2de6d83`) plus
the security and correctness fixes listed under _Fixed_, which is why the release date is
later than that commit.

### Added

- **White-box notebook generation**: every AutoML run now exports a runnable Jupyter
  notebook reproducing its preprocessing and model (`src/notebook_generator.py`),
  logged as an artifact on the MLflow run. Ported from `automlops-studio`
  (`bdd9b7a`), with dynamic metric support (`d3113f8`), notebook structure and MLflow
  logging (`18176bb`) and dataset-path wiring (`a36a4c1`).
- **HuggingFace as an experiment backend**: transformer fine-tuning for text tasks and
  Hub push/pull from the UI (`src/huggingface_utils.py`, `f83c877`).
- **Deep Feature Synthesis** as an opt-in preprocessing stage (`d791a21`).
- **Strict cross-validation** mode and explicit security warnings on unsafe
  deserialization paths (`3f1da5c`).
- **Multimodal, clustering, multi-label and anomaly-detection tasks**, plus a task
  catalog that filters frameworks by compatibility (`d15f139`, `f3f7b2d`, `19ff39a`,
  `c688618`, `bed8f0f`).
- **Universal orchestrators**: framework dispatch decoupled from the Streamlit layer
  (`src/orchestrator.py`, `src/processor.py`, `src/training_worker.py`) (`76f260f`).
- **Desktop resilience**: Streamlit load retries and an error page in the Electron
  shell (`e1f913b`).
- **CI**: lint/compile/regression gate on every push and PR, nightly full suite
  (`b42fc84`, `dbb93bf`), and a three-OS Electron build workflow (`build-electron.yml`).
- MIT license (`be9910d`).

### Changed

- Base runtime moved to Python 3.12 in the container images and CI (`c46b08e`).
- Training flow gained validation checks, error handling and reworked data processing
  (`a4324f3`).
- `xgboost` and `nbformat` became explicit runtime dependencies (`f83c877`).

### Fixed

- **H2O prediction was broken for every dataset**: `prepare_data_for_h2o` indexed the
  target column unconditionally while `predict_with_h2o` passes a placeholder target,
  so each prediction raised `KeyError` (`src/h2o_utils.py`).
- **Generated notebooks could not run**: they called `AutoMLDataProcessor.fit_transform`
  and `.transform` with the wrong arity, and instantiated the framework name as a model
  class. The notebook now uses the real processor API and loads the winning model from
  its MLflow run instead of re-fitting it (`src/notebook_generator.py`,
  `src/training_worker.py`).
- **Leaderboard cleanup crash** in the H2O path when the leaderboard could not be
  converted to CSV (`src/h2o_utils.py`).
- **Path traversal from user-controlled names**: run names and data-lake prefixes/names
  are reduced to a single safe path component before being joined into filesystem
  paths, and the destructive `models/<run>` cleanup now asserts it stays inside
  `models/` (`src/data_utils.py`, `app.py`, `src/autogluon_utils.py`,
  `src/autokeras_utils.py`).
- **Zip-slip on CV dataset upload**: archive members that would extract outside the
  target directory are now rejected (`src/data_utils.py`).
- **Electron external-link handling**: `setWindowOpenHandler` passed any URL — including
  `file://` and custom schemes — straight to `shell.openExternal`, and nothing constrained
  top-frame navigation. Both are now restricted to `http(s)` and to the local app origin
  (`electron/main.js`). The obsolete `new-window` handler (removed from Electron) was
  dropped.
- **Electron About dialog and docs link** reported `v1.0.0` and a placeholder repository
  URL; the version now comes from `app.getVersion()` and the link points at this repo.
- **Dependency CVEs** in the pinned stack (OSV): GitPython `3.1.50 → 3.1.62`
  (incl. CVE-2026-78676, CRITICAL), mlflow/mlflow-tracing `3.14.0 → 3.15.0`
  (CVE-2026-64849, CRITICAL), pillow `12.2.0 → 12.3.0`, aiohttp `3.14.1 → 3.14.3`,
  cryptography `48.0.1 → 49.0.0` (mlflow 3.15 caps cryptography at `<50`).
- **Build tooling pinned**: `requirements-dev.txt` is now tracked (`.gitignore` excluded
  every `*.txt*`), so CI installs the pinned ruff/pytest instead of falling back to
  whatever the index serves; the pins were aligned with `requirements.txt`.

### Known limitations

- Installers are not code-signed or notarized, so Windows SmartScreen and macOS
  Gatekeeper warn on first launch.
- The Electron shell starts the system Python and expects the app dependencies to be
  installed already; it does not bundle an interpreter.
- Two advisories remain unpatched by design: CVE-2026-71211 (MLflow AI Gateway SSRF —
  no fixed release, and this app does not use the gateway) and CVE-2026-69247
  (cryptography PKCS#7 decryption — blocked by mlflow's `cryptography<50`, and this app
  performs no PKCS#7 decryption).
- `.dvc/config` ships without a DVC remote, so `data_lake/*.dvc` pointers only resolve
  after each user configures their own storage.
- `electron/renderer.js` and the `window.electron` block in `electron/preload.js` are
  dead code kept for a future native-desktop layer.

## Before 5.0.0

No version before 5.0.0 was ever tagged or published. The earlier numbering was informal:
four snapshot commits on 2026-02-25/26 labelled `first version` (`825699f`: app skeleton,
AutoGluon utilities, Docker files), `second version` (`a09010f`: FLAML), `third version`
(`34ea15a`: H2O plus the simulation test suite) and `fourth version` (`f6fe252`: TPOT).
Later numbering (and the current `5.0.0`) only ever existed in the README badge and
`package.json`, so no retroactive tags are published for them.

