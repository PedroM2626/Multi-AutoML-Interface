# Changelog

All notable changes to Multi-AutoML Interface are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Release tags are `vMAJOR.MINOR.PATCH` and must match `version` in `package.json`;
pushing such a tag runs the `Release Desktop App` workflow, which builds the
Windows/macOS/Linux installers and attaches them to the GitHub Release.

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
- **Requests that could hang forever.** `dvc init`/`dvc add` and the Java probe ran
  without timeouts. They now bound at 120/900/5 seconds.
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
- **Desktop shell:** external links (`file://`, custom schemes) were passed straight to
  `shell.openExternal` with no navigation guard; Electron moves from the unsupported
  28.3.3 to 44.4.5 with `electron-builder` 26.15.3; the preload assigned
  `window.electron` in its own isolated world where no page could read it, now exposed
  through `contextBridge`; `npm ci` replaces `npm install` so the lockfile is respected.
- **Dependency advisories:** mlflow and mlflow-tracing to 3.16.1 and cryptography to
  50.0.1, which closes the two advisories 5.0.0 had to leave open (CVE-2026-69247,
  CVE-2026-71211). OSV reports no applicable vulnerability for any pin in
  `requirements.txt` and `npm audit` reports none for the desktop toolchain.
  The unused `skops` pin was dropped.

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

