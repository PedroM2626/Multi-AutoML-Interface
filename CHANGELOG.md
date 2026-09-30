# Changelog

All notable changes to Multi-AutoML Interface are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Release tags are `vMAJOR.MINOR.PATCH` and must match `version` in `package.json`;
pushing such a tag runs the `Release Desktop App` workflow, which builds the
Windows/macOS/Linux installers and attaches them to the GitHub Release.

## [Unreleased]

### Fixed

- **An H2O run no longer leaves a Java cluster running for the rest of the process.** The client
  keeps one connection per process, so `train_h2o_model` used to hand the live `H2OAutoML` to the
  session and never shut the cluster down: that is what let post-training prediction work, and it
  also meant 2-4 GB of heap parked in a shared, multi-session server, plus the risk of one run's
  `cleanup_h2o()` killing the cluster another run was still training on. `h2o_cluster()` now holds
  a lock, starts a **private cluster on a free loopback port**, and shuts it down on the way out -
  including when the body raises or `initialize_h2o()` itself fails, which used to strand the lock.
  The free port matters: without one the client *adopts* whatever cluster is listening on 54321,
  which is how a second session ended up shutting down the first one's JVM.
  Training returns `H2OSessionModel(run_id)`; `predict_with_h2o` reopens a cluster, reloads the
  model with `fetch_h2o_model`, and materialises the result as numpy before releasing. Verified
  live on the 3.11 interpreter: after train, after a leaderboard read and after prediction, `jps -l`
  lists no `H2OApp`.
- **The H2O model reload path had never worked.** `fetch_h2o_model` only matched a `*.zip`
  artifact, but `h2o.save_model` writes an archive named after the model id **with no extension**,
  so every reload raised `H2O model not found in artifacts.` It was invisible because training had
  always handed the live object to the caller. Both layouts are accepted now, `.zip` first.
- **The H2O Inspector reads its leaderboard from the run's artifacts.** The first version of the
  reloaded-cluster Inspector called `model.leaderboard` on the model `h2o.load_model` returns; that
  attribute does not exist. Measured on h2o 3.46 with a real run: `hasattr` is `False` for
  `leaderboard`, `leader`, `best_model` and `all_models`, and reading it raises
  `AttributeError: type object 'ModelBase' has no attribute 'leaderboard'` - a reloaded model is one
  estimator, not the AutoML object. `h2o_run_leaderboard(run_id)` now parses the
  `h2o_leaderboard_<run>.csv` the run logs (writer and reader share one constant, and a test fails
  if the training path stops using it), which also means opening that expander costs no JVM. A run
  that trained no model says so instead of showing an empty table.

### Added

- **`tests/test_h2o_cluster_lifecycle.py`** - 16 tests against a fake `h2o` module, so the whole
  lifecycle is covered without Java: one cluster per nested operation, release when the body
  raises, release when the cluster never started, exclusivity across threads (a waiting operation
  starts no JVM and cannot shut one down), cancellation while queued, handle-not-object returned by
  training, extensionless and `.zip` artifact reloads, and predictions materialised before the
  cluster goes away. `h2o_cluster` is a class rather than `@contextmanager` because PEP 479 turns a
  `StopIteration` raised in a generator body into `RuntimeError`, which `training_worker` would
  stop recognising as a cancellation - the test for that is what found it.

## 5.5.0 - 2026-09-30

### Added

- **`requirements-all.txt`: every catalog engine in one interpreter.** Compiled for **Python 3.11**
  from `requirements-all.in` with
  `uv pip compile requirements-all.in --python-version 3.11 -o requirements-all.txt` (284 packages:
  FLAML, AutoGluon tabular + multimodal, PyCaret, Lale, TPOT, H2O, SHAP, ONNX, MLflow, Streamlit).
  3.11 is not a preference - `pycaret 3.3.2` raises `RuntimeError: Pycaret only supports python
  3.9, 3.10, 3.11` while importing on 3.12, and its pins drag numpy/pandas/scipy/matplotlib/
  scikit-learn back with it. `tests/test_engine_matrix.py` now trains **and scores** every row of
  the task catalog with the engine that row names, skipping whatever the interpreter lacks; the
  nightly `engine-matrix` job runs it on a 3.11 runner with CPU-only torch.
- **`RUNTIME_REQUIREMENTS` in `scripts/prepare_python_runtime.js`**, so a local build can bundle a
  different lock. The released installers keep `requirements.txt`: measured, the core stack is
  872 MB of site-packages and the all-engine stack is about 1.7 GB (torch alone is 465 MB), and
  GitHub caps a release asset at 2 GiB. The all-engine lock also cannot be made CVE-clean: PyCaret
  pins scikit-learn 1.4.2 (CVE-2024-5206 is only fixed in 1.5.0), and both TPOT (`stopit`) and
  `autogluon.multimodal` (`data.templates`) import `pkg_resources`, which setuptools >= 81 no
  longer ships, while CVE-2026-59890 is only fixed in 83.0.0. AutoGluon's tabular learner also
  opens a pandas option that only exists from 2.2, which PyCaret's `pandas<2.2` pin forbids - so
  no single interpreter runs the whole catalog, and `tests/test_engine_matrix.py` records that
  pair as an expected failure rather than hiding it. `requirements-all.txt` is the Windows lock -
  this set has no portable one, because `pywin32` carries no marker and PyPI's Linux torch is the
  CUDA build - so the nightly job installs from `requirements-all.in` with the CPU index, and each
  platform regenerates its own lock.
- **Computer Vision Multi-Label Classification is offered again, with an input that can express
  it.** The CV upload now also takes an annotations CSV - an `image` column naming files inside the
  dataset plus one 0/1 column per label - stores it as `annotations.csv` in the dataset folder, and
  `load_data` returns that table instead of the one-row directory stub. `train_model` resolves the
  image paths and fits one `MultiModalPredictor` per label column behind the existing
  `MultiLabelAutoGluonPredictor` wrapper - asking the predictor for `problem_type="multilabel"`
  asserts inside `fit()`, and its error lists every type it does support. Folder names
  hold exactly one class per image, so the row now says so instead of training a single-label model
  behind a multi-label label.

### Fixed

- **The first prediction after an AutoGluon training failed.** `train_model` copies the model into
  the MLflow run and then deletes `models/<run_name>/` to save disk, but returned the predictor it
  had just fitted - an object that reads `models/<run_name>/models/*/model.pkl` from exactly that
  folder. The app keeps it in session state, so scoring the next row raised
  `FileNotFoundError: ... LightGBMXT/model.pkl` even though the run had succeeded. After the
  cleanup the run's artifact is now reloaded and that object is what the session gets, which also
  exercises the MLflow load path on every AutoGluon run. Found by
  `tests/test_engine_matrix.py`, which predicts with the returned predictor.
- **AutoGluon's tabular rows in the all-engine interpreter now say what is wrong.** Its learner
  opens `pd.option_context("future.no_silent_downcasting")`, a key that only exists from pandas 2.2,
  and PyCaret's pin holds pandas below it: the run used to die with a bare `OptionError` after the
  data had already been processed. `train_model` refuses it up front and names both the engine's
  need and the pin that blocks it.
- **Every AutoGluon vision, text and multimodal run could crash the interpreter that also had
  PyCaret.** scikit-learn <=1.4 wheels vendor `sklearn/.libs/vcomp140.dll` and map it from
  `sklearn/_distributor_init.py`; after that, torch's `c10.dll` fails its DllMain with
  `OSError [WinError 1114]` and the process dies. `preload_torch_before_sklearn()`
  (`src/task_catalog.py`) runs at the top of `app.py` and in `tests/conftest.py`, and only when
  that DLL is actually vendored, so a modern interpreter pays nothing for it.
- **A Lale run hung forever with the CPU idle.** `max_eval_time` makes Lale's hyperopt spawn one
  `multiprocessing.Process` per trial through the Windows spawn start method, which re-imports the
  parent's `__main__` - the Streamlit CLI inside this app - and blocks in
  `multiprocessing.reduction.dump`. `max_opt_time` is worse still: it answers a timeout with
  `sys.exit(0)` in the search thread. The Lale budget now bounds `max_evals` only.
- **The packaging pipeline could no longer build the runtime.** Three separate things: `npm audit`
  now lists twelve high-severity advisories against the axios that `wait-on` pulls (1.18.0 ->
  1.20.0 with `npm audit fix`); the workflows pinned Node 20 while `@electron/get` 5.1.0 declares
  `>= 22.12`; and the copied tree lost its interpreter - `fs.cpSync` leaves `bin/python3` and
  `lib/libpython3.12.so` as symlinks, some absolute into the staging directory the script then
  deletes, so `existsSync` and even a version probe succeed and the next spawn answers ENOENT.
  Windows kept passing because `python.exe` is a plain file, which made the first diagnosis
  (a newer uv refusing `pip install --system` on the copied tree, so the payload now goes in with
  the bundled interpreter's own pip, and the workflows read `uv==` from `requirements.txt`) look
  right until the same ENOENT came back from `ensurepip`. Links under `bin/` and `lib/` are
  materialized now, the interpreter is verified before staging is removed, and the
  uncompressed-size report can no longer fail a release.
- **The shipped lock carried advisories again.** urllib3 2.7.0 is now flagged by CVE-2026-97687,
  -97688 and -97689 (fixed in 2.8.0), and the `setuptools<81` line added for TPOT's `stopit`
  import brought CVE-2026-59890 into every installer even though TPOT is not part of this stack.
  Both moved up; `pip-audit -r requirements.txt` reports no known vulnerabilities.
- **`run.py` moved the app onto an interpreter that had none of its dependencies.** It re-launched
  itself on any Python 3.11 it could find, while the shipped stack is 3.12 - so the app died on
  import errors instead of starting. It now starts on the current interpreter and prints which
  catalog engines that interpreter cannot import, with the command that adds them.

## 5.4.0 - 2026-09-30

### Fixed

- **AutoGluon threw away every text, multimodal and computer-vision result.** The reporting step
  called `predictor.leaderboard(...)`, which `MultiModalPredictor` does not implement, so the run
  died *after* training had completed and logged nothing. That path now evaluates the fitted model
  with `evaluate()`, logs the numeric metrics it returns, and skips the ONNX attempt (the
  multimodal predictor has no `export_onnx`). Verified end to end on CPU: Text classification
  (216 s), Multimodal classification (298 s) and CV image classification (128 s) each produced an
  MLflow run with metrics.
- **The macOS x64 disk image shipped an interpreter its target machines cannot execute.**
  `release.yml` built `--mac --x64 --arm64` while `prepare_python_runtime.js` installs the CPython
  of the runner's own architecture, so both images carried the same arm64 interpreter. macOS is
  built arm64-only now, and the packaging smoke test reads the bundled interpreter with `lipo` and
  fails when the image directory declares a different architecture - verified green on a real
  Apple Silicon runner.

- **TPOT could not reach training at all.** `detect_problem_type` tested the whole column on every
  loop step (`all(y % 1 == 0 for val in ...)`) and pandas raised "The truth value of a Series is
  ambiguous" the moment a numeric target arrived; it now checks `(values % 1 == 0).all()`.
  The estimator is also built from the signature of whichever TPOT is installed, because
  generations/population_size/scoring/verbosity/config_dict were dropped from the 1.x estimator
  and raised `TypeError` deep inside the search, after the UI had reported the run as started -
  the ignored knobs are logged instead of silently swallowed. `setuptools==80.9.0` is pinned:
  tpot -> stopit -> `import pkg_resources`, which setuptools >= 81 no longer ships, so TPOT could
  not even be imported in a fresh interpreter.
- **The availability check asked about the engine, not the module a row needs.** With
  `autogluon.tabular` installed and no `autogluon.multimodal`, the vision/text/multimodal rows were
  still offered and died inside the engine; availability is now resolved per
  (engine, data category), cached per module, and the "how do I install this" hint names the
  extra (`pip install autogluon.multimodal`) instead of the base package.

### Changed

- **The catalog is 14 pairs across 5 engines.** Object Detection and Image Segmentation are no
  longer offered: the CV upload infers labels from the directory structure, so there is no COCO
  box or mask annotation for the engine to read, and AutoGluon's detection pipeline also needs
  mmcv with PyTorch <=2.1. `train_model` still honours those problem types for a caller that
  brings an annotated frame. TPOT is no longer offered either - `pip install tpot` gives 1.1.0,
  which raises `TypeError: TPOTEstimator.__init__() got an unexpected keyword argument 'scoring'`
  from inside its own `fit` template, while 0.12.2 trains correctly against scikit-learn 1.4 but
  fails on this project's scikit-learn 1.9 with "Expected an estimator instance ... got estimator
  class instead". `src/tpot_utils.py` and its orchestrator entry stay for an environment that
  pins its own scikit-learn.
- Computer Vision offers Image Classification only. Multi-Label was removed with the same
  argument as detection: an image lives in exactly one class folder, so there is no multi-hot
  target to learn, and AutoGluon was quietly getting a plain multi-class problem while the UI
  said multi-label.
- AutoKeras leaves the catalog too. `pip install autokeras` gives 3.0.0 against keras 3.x, whose
  classification head rejects the single-unit output ("Received an invalid value for `units`,
  expected a positive integer. Received: units=1"), and multi-label fails on target shape; it
  needs a `keras<3` environment the project does not pin. Both CV rows stay available through
  AutoGluon, which was trained end to end on synthetic images.
- The support matrices list only engines some row can actually run, so TPOT no longer has a column
  of promises the catalog does not keep, and the docs stop counting the Hugging Face Hub as an
  eighth engine.

## 5.3.0 - 2026-09-29

### Added

- **ONNX export and SHAP explanations now ship in the installers.** They were never in
  `requirements.txt`, so the desktop app - which installs exactly that file - could not run the
  *🧠 Explain Prediction* or *📦 Export to ONNX* buttons at all. The lock now carries
  `onnx`, `onnxruntime`, `skl2onnx`, `onnxconverter-common` and `shap` (plus `numba`, `llvmlite`,
  `slicer`, `tqdm`, `flatbuffers`, `ml-dtypes`), with `shap` excluded on Intel macOS, where the
  `numba` version it allows cannot take the `numpy==2.5.0` pin. Windows and Linux resolve; the
  packaging workflow verifies the macOS build.

### Fixed

- **`export_to_onnx` never wrote a model.** It called `to_onnx(model, input_sample[:1], ...)`,
  which makes skl2onnx treat every column as a separate input, so even a plain
  `RandomForestClassifier` raised `InvalidInputLengthException`; the export now names one
  `FloatTensor` of the sample's width and the artifact loads and predicts through onnxruntime.
  The Experiments button also handed over FLAML's `AutoML` wrapper where the engine had passed the
  inner estimator - the wrapper is unwrapped now. Boosted-tree learners (`lgbm`, `xgboost`,
  `catboost`) genuinely have no converter in skl2onnx, so that case raises a message naming the
  estimator instead of a warning logged inside a training thread nobody reads; FLAML's default
  learner is `lgbm`, which is why the feature looked like it worked and did nothing.
- **The tabular SHAP path could not be imported without OpenCV.** `src/xai_utils.py` had
  `import cv2` at module scope while only the saliency-map function uses it (and imports it
  there), so *Explain Prediction* failed on any interpreter without opencv-python.

## 5.2.1 - 2026-09-29

### Fixed

- **Three of PyCaret's five catalog rows could not start.** Verified by installing
  `pycaret==3.3.2` in an isolated Python 3.11 interpreter and running
  `run_pycaret_experiment` for each task type:
  - Anomaly Detection and Clustering raised `TypeError: setup() got an unexpected keyword
    argument 'fold'` - the unsupervised setups have no cross-validation folds. `fold` is now
    passed only to the supervised ones, and those rows produce `IForest` and `KMeans`.
  - Forecast raised `ValueError: Estimator naive Not Available` as soon as the frame carried
    any column besides the target, because PyCaret's time series module is univariate and
    keeps only the pmdarima family available. The estimator list now follows the frame
    (`_ts_include_models`), and the date column the UI selects is moved into the index
    instead of being read as an exogenous feature. Both shapes - raw ordering under
    Sequential, lag features under Tabular - train to an `EnsembleForecaster`.
- **Two PyCaret trainings in one process never finished.** The functional API keeps a single
  process-global experiment, and this module also ended whichever MLflow run was active; with
  two sessions training at once both threads were stuck for minutes, while each run alone
  takes seconds. Concurrent MLflow runs were tested and are fine, so the engine is serialized:
  `run_pycaret_experiment` now queues behind a lock and a queued run can still be cancelled.
- **The availability guard broke tests that stub the engine module.** Two dispatch tests built a
  `FLAML` orchestrator with a fake module and hit the new "install it with: pip install flaml"
  check on interpreters without FLAML; they now declare the engine present, and a test covers
  the guard itself.

## 5.2.0 - 2026-09-29

### Fixed

- **Most catalog rows pointed at engines the interpreter did not have.** The bundled runtime
  installs `requirements.txt`, whose only AutoML engine is FLAML (plus LightGBM and XGBoost),
  yet the framework selector offered AutoGluon, PyCaret, Lale, TPOT, H2O and AutoKeras for most
  of the 23 `(category, task)` rows; the background thread then died on `No module named
  'autogluon'`, far from the widget that caused it. Both the training selector and the
  model-source selector now list only engines that can be imported and print the `pip install`
  line for the rest, and the orchestrator raises the same message before starting a thread.
- **FLAML Forecast and Ranking could not train.** `ts_forecast` asserts a forecast `period`
  before the search starts, so every Forecast run with FLAML failed on the first iteration;
  Forecast now passes the date column as `time_col` and the horizon as `period` (Sequential uses
  that native path, Tabular keeps the processor's lag features and trains as regression).
  Ranking handed LightGBM float relevance grades and rows in arbitrary order; it now sorts by a
  new *Query / Group Column* input and casts integer grades, and Ranking lists only the boosting
  learners because the sklearn forests reject the `group` argument the ranker forwards. Missing
  inputs raise a readable `ValueError` instead of failing inside the learner. Both were run end
  to end against the bundled interpreter, and `tests/test_flaml_task_paths.py` keeps them covered.
- **Rows that no engine implemented.** `Semi-Supervised Classification` was a task row while the
  real feature is the Classification checkbox that wraps the model in `SelfTrainingClassifier`;
  Text/Clustering had no text featurizer; four Sequential rows dispatched exactly like their
  Tabular twins. Hugging Face logged parameters and returned a successful run id without
  training anything, and its "models" could not be loaded back by the prediction service, so
  `run_huggingface_experiment` is gone - the Hub push/pull service stays.
- **Forecast models were restored through the wrong PyCaret module.** The catalog calls the task
  `Forecast`, but `prediction_service` and the generated code still compared the older
  `"Time Series Forecasting"`, so a time-series artifact was loaded with
  `pycaret.classification.load_model`.
- **The data lake offered Git LFS pointer files as datasets.** Several `data_lake/raw/*.csv` are
  committed through LFS and were never pulled, so pandas read the 130-byte pointer as a
  one-column table and the Training page proposed `version
  https://git-lfs.github.com/spec/v1` as a data column. Loading one now says to run
  `git lfs pull`.

### Changed

- Text tasks train through AutoGluon's multimodal predictor with the columns you mark as text,
  the same path Multimodal already used, instead of a tabular predictor that treated the text as
  one categorical feature.
- `Sequential` is now one row (Forecast): the category exists to hand the raw time ordering to an
  engine's native time series task, which is also why AutoGluon is not offered there - its
  tabular predictor cannot forecast a future step from same-row features.

### Added

- **The documented support matrices are checked against the catalog.** `README.md` and
  `docs/DOCUMENTATION.md` restate `TASK_FRAMEWORK_MAP`, and had drifted (rows for engines with no
  code path, the Forecast rename). `tests/test_doc_matrix_sync.py` parses both files and compares
  them pair by pair; it is dependency-free, so it runs in the PR gate.
- **The dispatch contract is read from `app.py`, not transcribed.** The engine-kwargs test kept a
  hand-written key list that had already drifted for PyCaret and Lale; the tests now parse the
  dispatch chain with `ast`.

## 5.1.0 - 2026-09-28

### Fixed

- **A flaky ONNX test, caught by the new nightly gate.** The export fixture drew its
  target from an unseeded `np.random.randint(0, 2, 10)`, which can be a single class;
  `LogisticRegression` then refuses to fit. It passed on Windows by luck and failed on the
  first Linux nightly where the full suite is a real gate. The feature matrix is seeded and
  the target is balanced by construction.
- **The desktop app no longer needs Python installed by the user.** `scripts/prepare_python_runtime.js`
  downloads a standalone CPython 3.12 with `uv` and installs `requirements.txt` into it;
  electron-builder ships that tree as `resources/runtime`, and `electron/main.js` starts the
  bundled interpreter through `runtime/runtime-manifest.json`, falling back to the system
  Python only in a source checkout. Verified by packaging the app and launching it: the
  window renders, `/_stcore/health` answers, and the relocated interpreter imports
  streamlit/mlflow/flaml/pandas/sklearn.
- **Runs, models and the data lake were written next to the program files.** The app now
  works in a per-user workspace (Electron `userData`, e.g.
  `%APPDATA%\multi-automl-desktop\workspace`), which a normal user can write to; Program
  Files is not. `safe_set_experiment` resolves `mlruns/` against the working directory
  instead of the source tree so the change takes effect, and `PYTHONPATH` keeps `src/`
  importable from the new cwd.
- **Smoke builds were self-signing every bundled executable.** Without credentials
  electron-builder generated its own certificate and signed hundreds of files inside the
  runtime, which is slow and produces signatures nobody trusts. The packaging workflow now
  builds unpacked directories with signing explicitly off and asserts the packaged layout
  (`resources/runtime/...`, `resources/app/app.py`) instead of uploading 1.2 GB per OS.

### Added

- **Signing is wired up, and verified.** `release.yml` signs Windows installers from
  `WIN_CSC_LINK`/`WIN_CSC_KEY_PASSWORD` and macOS from `MAC_CSC_LINK`/`MAC_CSC_KEY_PASSWORD`
  plus `APPLE_ID`/`APPLE_APP_SPECIFIC_PASSWORD`/`APPLE_TEAM_ID`, because electron-builder
  reads those from the environment. A build that had credentials but produced an unsigned
  artifact now fails, signature reports are uploaded as artifacts, and the release notes
  state which case applied. With no credentials the build stays unsigned and says so.
  Azure Artifact Signing is documented as an alternative but is not wired: it needs an
  explicit `win.sign` configuration block, and passing it on the command line
  (`-c.win.sign.type=azure`) is rejected by electron-builder 26's schema - as is
  `-c.win.sign=false`, which is what broke the first packaging runs.
- `npm run runtime` builds just the bundled interpreter, and the packaging scripts run it
  before electron-builder, so `npm run build-win` produces a working installer in one step.

## 5.0.2 - 2026-09-28

### Fixed

- **The desktop "Abrir MLflow" menu opened a port nothing was listening on.** The desktop
  app records runs in the local `./mlruns` file store, so `http://localhost:5000` only
  works when the MLflow container is running. The menu now opens `MLFLOW_TRACKING_URI`
  when it points at an http(s) server, and otherwise explains where the runs are and how
  to start the UI.

### Fixed

- **FLAML training crashed with the app's own default settings.** `estimator_list`
  defaults to `['lgbm', 'rf']`: LightGBM is not in `requirements.txt`, so the search died
  inside FLAML with `TypeError: 'NoneType' object is not callable`, and the telemetry
  callback FLAML forwards to every learner took six arguments while LightGBM calls it with
  one `CallbackEnv` - with mixed lists sklearn then raised
  `BaseForest.fit() got an unexpected keyword argument 'callbacks'`. LightGBM is now a
  declared dependency, the callback matches the `CallbackEnv` contract, it is registered
  only for learners that accept it, and a missing learner package is named before the
  search instead of failing deep inside cross-validation. Verified end to end in a clean
  environment: train -> MLflow run -> pickle -> trusted reload -> predictions -> notebook.
- **Three more vulnerable pins**, found by the `pip-audit` gate added in 5.0.1: anyio
  `4.14.1 -> 4.14.2`, pyasn1 `0.6.3 -> 0.6.4`, sqlparse `0.5.5 -> 0.6.0`. `pip-audit
  --strict` over `requirements.txt` now reports no known vulnerabilities, and it is that
  gate - not the earlier spot check behind the 5.0.1 note - that found them.

### Corrected

- The 5.0.1 entry claimed OSV reported no applicable vulnerability for any pin in
  `requirements.txt`. That was checked against a subset of ~40 packages; the full
  resolved audit found the three above. The published 5.0.1 release notes were edited to
  drop the overstatement.

## 5.0.1 - 2026-09-28

Second release, published after an audit of the codebase and of the packaged app. It
fixes the local-first MLflow default, closes the multi-session exposures, repairs
several runtime defects and leaves the end-of-life Electron 28 shell.

### Fixed

- **The app recorded no MLflow runs.** `safe_set_experiment` failed on every startup
  because MLflow 3 refuses a file-based tracking store unless
  `MLFLOW_ALLOW_FILE_STORE` is set; the test suite set it, so the breakage was
  invisible in CI. The desktop app now logs the experiment setup successfully, and a
  configured `MLFLOW_TRACKING_URI` is honoured instead of being overwritten with the
  local path (a shared server or database store was silently ignored before).
- **`python run.py` listened on every interface.** Streamlit binds `0.0.0.0` when no
  address is given, so the local launcher exposed the app, and the code execution of
  the Python behind it, to the whole network. It now binds `127.0.0.1` unless
  `--server.address` or `STREAMLIT_SERVER_ADDRESS` is supplied, and the DagsHub
  credential gate treats an unset address as shared.
- **Cross-session credential leak.** The DagsHub panel wrote a visitor's username and
  token into process-global `os.environ` and never cleared them; in multi-session mode
  another user's run would authenticate with them. Per-user tokens are accepted only
  when the server is bound to loopback.
- **Untrusted model loading (CWE-502).** All six MLflow flavors are restored with
  pickle/joblib, and the run id came from a free-text field against a tracking URI that
  the sidebar can repoint. Loading now requires an explicit confirmation in the UI and
  rejects run ids containing path characters.
- **CORS was disabled everywhere.** Both containers and the Electron launcher passed
  `--server.enableCORS=false`; with the app reachable from other origins, any page that
  could reach the port could read and post to it. Streamlit's defaults now stand, and
  Compose publishes 8501/5000 on loopback only.
- **Leaked host repository into containers.** Compose bind-mounted `.:./app`, which
  also exposed `.git` and let the container overwrite source; it now mounts `data_lake/`
  and `mlruns/` only. Its MLflow server image (`v2.11.1`) was also two majors behind the
  pinned client and is now version-matched.
- **Threads that never stopped.** The H2O cancellation watcher and telemetry loop only
  exited when training returned, so a failed run left them polling inside the shared
  process; they are released from a `finally`. Two concurrent FLAML runs wrote the same
  `flaml.log`, now named per run.
- **Requests that could hang forever.** `dvc init` and `dvc add` ran without timeouts and
  so could block a session indefinitely; they now bound at 120 and 900 seconds, and the
  interpreter probe in `run.py` at 10.
- **Run history destroyed by the auto-healer.** `heal_mlruns` deleted any numeric
  `mlruns/` directory lacking `meta.yaml`, which under multi-session is an experiment
  being written right now. It quarantines to `mlruns/.trash` instead and skips anything
  touched within the last hour.
- **`shutil.rmtree` on a path built from user input, ZIP extraction without member
  checks, and 14 bare `except:` clauses** that swallowed `KeyboardInterrupt`.
- **`queue_experiment()` crashed when called without a manager**, because it fell back
  to `get_or_create_manager()` without the session state that function requires; the
  manager is now an explicit argument, so the orchestrator cannot silently share one
  across sessions.
- **`run.py` accepted any interpreter newer than 3.11** while the frameworks need 3.11.
  It now re-launches on 3.11 whenever available, warns and continues on a newer
  interpreter, and hard-fails only on older ones.
- **Generated notebooks could not be written** in the installed app: the exporter wrote
  into the working directory, which is inside Program Files there. They now land under
  the system temp directory.
- **Progress bars disappeared in a terminal.** The stdout/stderr router used to capture
  per-run logs inherited `io.TextIOBase`, whose `isatty()` always answers False and whose
  `fileno()` raises - so H2O, FLAML and tqdm disabled their bars even in a real terminal,
  and anything probing the descriptor failed. Both now delegate to the underlying stream
  and degrade cleanly when there is none.
- **A cancelled run dropped its result.** `refresh_all` only polled entries that were
  running or queued, so the payload a cancelled worker still delivered was never read:
  `entry.result` stayed empty and the UI reported "Unknown" instead of the real outcome.
  Cancelled runs are polled too, and a late result no longer relabels the row as
  completed or failed.
- **Desktop shell:** external links (`file://`, custom schemes) were passed straight to
  `shell.openExternal` with no navigation guard; Electron moves from the unsupported
  28.3.3 to 44.4.5 with `electron-builder` 26.15.3; the preload assigned
  `window.electron` in its own isolated world where no page could read it, now exposed
  through `contextBridge`; `npm ci` replaces `npm install` so the lockfile is respected.
- **Dependency advisories:** mlflow and mlflow-tracing to 3.16.1 and cryptography to
  50.0.1, which closes the two advisories 5.0.0 had to leave open (CVE-2026-69247,
  CVE-2026-71211). OSV reports no applicable vulnerability for any pin in
  The unused `skops` pin was dropped. (Spot-checked against a subset of packages at the
  time; the full audit added in this release then found three more, see 5.0.2.)

### Added

- **CI gates that mean something:** the nightly full suite is now authoritative when the
  dependency stack installs (it was `continue-on-error`), `pip-audit --strict` runs over
  `requirements.txt`, `npm audit --audit-level=high` runs before packaging, and both
  Python and JS installers now build from lockfiles. `pytest` invocations pass
  `-o addopts=""` so the pass/skip summary is not swallowed by a double `-q`.
- **Multi-session deployment notes** in `docs/DOCUMENTATION.md`, plus troubleshooting
  entries for the loopback default and the artifact-trust confirmation.
- A missing DVC remote is now reported after an upload, because the `.dvc` pointer will
  not resolve on another machine.

### Changed

- `build-electron.yml` no longer runs the 3-OS matrix on every push to `main`; it builds
  on packaging changes in pull requests and on manual dispatch, since `release.yml`
  already builds and publishes on tags.

### Known limitations

- The app still has no authentication or per-user quota of its own: an internet-facing
  deployment must terminate TLS and authentication in a reverse proxy, and sessions
  continue to share `mlruns/`, `models/` and the data lake in one working directory.
- `electron/renderer.js` is still not wired into the window; enabling it would overlay a
  custom header on the Streamlit UI, which is a design decision rather than a bug fix.
- 23 `use_container_width` calls in `app.py` emit Streamlit deprecation warnings past
  their announced removal date. They cannot be replaced mechanically: `st.pyplot` has no
  `width` argument, so each widget needs its own judgement.
- Installers remain unsigned and unnotarized, and still require Python plus
  `requirements.txt` on the target machine.

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

