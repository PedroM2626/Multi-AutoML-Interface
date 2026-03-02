import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import queue
from sklearn.model_selection import train_test_split

# Development Cache Optimization (optional via URL ?dev=true)
dev_mode = st.query_params.get("dev", "false").lower() == "true"
if dev_mode:
    st.sidebar.info("🛠️ Dev Mode: Reload active")
    modules_to_reload = [
        'src.autogluon_utils',
        'src.flaml_utils', 
        'src.h2o_utils',
        'src.tpot_utils',
        'src.mlflow_cache'
    ]
    for module in modules_to_reload:
        if module in sys.modules:
            importlib.reload(sys.modules[module])

# Functions with cache for Performance
@st.cache_data(show_spinner="Loading data...")
def cached_load_data(file_path_or_obj):
    return load_data(file_path_or_obj)

@st.cache_data
def cached_get_data_summary(df):
    return get_data_summary(df)

@st.cache_data(ttl=60) # 1 Minute Cache for file list
def cached_get_data_lake_files():
    return get_data_lake_files()

from src.data_utils import load_data, get_data_summary, save_to_data_lake, init_dvc, get_data_lake_files, get_dvc_hash
from src.autogluon_utils import train_model as train_autogluon, load_model_from_mlflow as load_autogluon
from src.flaml_utils import train_flaml_model, load_flaml_model
from src.h2o_utils import train_h2o_model, load_h2o_model
from src.tpot_utils import train_tpot_model, load_tpot_model
from src.log_utils import setup_logging_to_queue, StdoutRedirector
from src.mlflow_utils import heal_mlruns
from src.mlflow_cache import mlflow_cache, get_cached_experiment_list
from src.experiment_manager import get_or_create_manager, ExperimentEntry
from src.training_worker import run_training_worker
import mlflow
import time
import threading

st.set_page_config(page_title="AutoML Visual Interface", layout="wide")

# Heal MLflow cache on startup
heal_mlruns()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'predictor' not in st.session_state:
    st.session_state['predictor'] = None
if 'model_type' not in st.session_state:
    st.session_state['model_type'] = None
if 'log_queue' not in st.session_state:
    st.session_state['log_queue'] = queue.Queue()

# Initialise the experiment manager singleton
exp_manager = get_or_create_manager(st.session_state)

st.title("🚀 AutoML Multi-Framework Interface")

# Initialize MLflow experiment and tracking
try:
    from src.mlflow_utils import safe_set_experiment
    safe_set_experiment("Multi_AutoML_Project")
except Exception as e:
    st.error(f"Error initializing MLflow: {e}")

# Sidebar - Stats & Status
st.sidebar.title("📊 Training Status")

# Badge for running experiments (cached for 5s to avoid script-wide slowdown)
curr_time = time.time()
if '_last_count_time' not in st.session_state or curr_time - st.session_state['_last_count_time'] > 5:
    st.session_state['_running_count'] = sum(1 for e in exp_manager.get_all() if e.status == "running")
    st.session_state['_last_count_time'] = curr_time

_running_count = st.session_state['_running_count']
_running_label = f" 🟢 {_running_count}" if _running_count else ""
menu = st.sidebar.selectbox("Menu", ["Data Upload", "Training", f"Experiments{_running_label}", "Prediction", "History (MLflow)"])
menu = menu.split(" 🟢")[0]  # Normalize label so page logic still matches

st.sidebar.markdown("---")
st.sidebar.header("🔗 DagsHub Integration (Optional)")
use_dagshub = st.sidebar.checkbox("Enable DagsHub")

if use_dagshub:
    dagshub_user = st.sidebar.text_input("DagsHub Username")
    dagshub_repo = st.sidebar.text_input("Repository Name")
    dagshub_token = st.sidebar.text_input("Access Token (DagsHub)", type="password")
    
    if st.sidebar.button("Connect to DagsHub"):
        if dagshub_user and dagshub_repo and dagshub_token:
            try:
                import dagshub
                import os
                os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
                os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
                dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
                st.sidebar.success("Successfully connected to DagsHub!")
            except ImportError:
                st.sidebar.error("dagshub library not found. Add 'dagshub' to requirements.txt and install it.")
            except Exception as e:
                st.sidebar.error(f"Connection error: {e}")
        else:
            st.sidebar.warning("Please fill all DagsHub fields.")
st.sidebar.markdown("---")

if menu == "Data Upload":
    st.header("📂 Data Upload and Data Lake")
    
    st.markdown("Upload new files to the Data Lake. They'll become available on the Training and Prediction tabs.")
    uploaded_file = st.file_uploader("New CSV/Excel File", type=["csv", "xlsx", "xls"])
    filename_prefix = st.text_input("Data Lake file prefix", value="dataset")
        
    if st.button("Process and Save to Data Lake"):
        if uploaded_file is not None:
            try:
                with st.spinner("Initializing Data Lake and processing data..."):
                    init_dvc()
                    df = cached_load_data(uploaded_file)
                    t_path, t_tag, t_hash = save_to_data_lake(df, filename_prefix)
                    st.cache_data.clear() # Clear cache because new data was injected
                    st.success(f"File loaded and versioned in the Data Lake with DVC! Generated Hash: {t_hash}")
                    
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                st.subheader("Data Summary")
                summary = cached_get_data_summary(df)
                s_col1, s_col2 = st.columns(2)
                s_col1.metric("Rows", summary['rows'])
                s_col2.metric("Columns", summary['columns'])
                
                st.write("Data Types and Missing Values:")
                summary_df = pd.DataFrame({
                    "Type": summary['dtypes'],
                    "Missing": summary['missing_values']
                })
                st.table(summary_df)
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            st.error("No file selected!")

elif menu == "Training":
    st.header("🧠 Model Training")
    
    available_files = cached_get_data_lake_files()
    
    if not available_files:
        st.warning("No datasets found in Data Lake. Please add them in the 'Data Upload' tab first.")
        st.stop()
        
    st.subheader("1. Data Lake Dataset Selection")
    
    # UI mapping filenames
    file_options = ["None"] + [os.path.basename(f) for f in available_files]
    file_paths_map = {os.path.basename(f): f for f in available_files}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        train_file_selection = st.selectbox("Training (Required)", file_options[1:])
    with col2:
        valid_file_selection = st.selectbox("Validation (Optional)", file_options)
    with col3:
        test_file_selection = st.selectbox("Test/Holdout (Optional)", file_options)
        
    if train_file_selection:
        try:
            # Load Train
            train_path = file_paths_map[train_file_selection]
            df = cached_load_data(train_path)
            
            # Fetch Hash
            t_hash_full, t_hash_short = get_dvc_hash(train_path)
            dvc_hashes = {"dvc_train_hash": t_hash_short}
            
            # Load Valid
            valid_df = None
            if valid_file_selection != "None":
                valid_path = file_paths_map[valid_file_selection]
                valid_df = cached_load_data(valid_path)
                v_hash_full, v_hash_short = get_dvc_hash(valid_path)
                dvc_hashes["dvc_valid_hash"] = v_hash_short
                
            # Load Test
            test_df = None
            if test_file_selection != "None":
                test_path = file_paths_map[test_file_selection]
                test_df = cached_load_data(test_path)
                te_hash_full, te_hash_short = get_dvc_hash(test_path)
                dvc_hashes["dvc_test_hash"] = te_hash_short
                
            # Store globally
            st.session_state['df'] = df
            st.session_state['valid_df'] = valid_df
            st.session_state['test_df'] = test_df
            st.session_state['dvc_hashes'] = dvc_hashes
            
        except Exception as e:
            st.error(f"Error loading datasets from Data Lake: {e}")
            
    st.markdown("---")
    st.subheader("2. Data Splitting and Validation Strategy")
    
    cv_folds = 0
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        valid_df_session = st.session_state.get('valid_df', None)
        test_df_session = st.session_state.get('test_df', None)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Final Test Set**")
            if test_df_session is None:
                test_size_pct = st.slider("Percentage extracted for Test (%)", 0, 50, 15, 5, help="Size of the test set retained for final model evaluation.")
            else:
                st.success("Test-set provided through a dedicated Data Lake file.")
                test_size_pct = 0
                
        with col2:
            st.markdown("**Internal Validation Strategy**")
            if valid_df_session is None:
                val_strategy = st.radio("Method", ["Simple Holdout", "Cross-Validation"], horizontal=True, help="Holdout will physically split the Dataset. CV instructs engines to use Folds.")
                if val_strategy == "Simple Holdout":
                    val_size_pct = st.slider("Percentage extracted for Validation (%)", 0, 50, 20, 5)
                else:
                    cv_folds = st.slider("Number of Folds (K)", 2, 10, 5)
                    val_size_pct = 0
            else:
                st.success("Validation-set provided via file in Data Lake.")
                val_size_pct = 0
                
        # Apply Splits if needed and store on UI refresh safely
        # We need a pristine copy or just track the original df length to not shrink infinitely on UI refreshes
        # We'll use the current st.session_state['df'] as base, but this requires we cache original on selection.
        if 'original_df' not in st.session_state or len(st.session_state['original_df']) != len(df) and ('has_split' not in st.session_state):
             # Keep track of original selection payload
             st.session_state['original_df'] = df.copy()
             
        base_df = st.session_state['original_df'].copy()
        
        if test_size_pct > 0:
            base_df, fresh_test_df = train_test_split(base_df, test_size=(test_size_pct/100.0), random_state=42)
            test_df_session = fresh_test_df
            st.session_state['test_df'] = test_df_session
            
        if val_size_pct > 0:
            if len(base_df) > 100: # Safe margin
                base_df, fresh_val_df = train_test_split(base_df, test_size=(val_size_pct/100.0), random_state=42)
                valid_df_session = fresh_val_df
                st.session_state['valid_df'] = valid_df_session
                
        # Update current working df
        df = base_df
        st.session_state['active_df'] = df
        st.session_state['cv_folds'] = cv_folds

    st.markdown("---")
    st.subheader("3. AutoML Configuration")
    
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        valid_df = st.session_state.get('valid_df', None)
        test_df = st.session_state.get('test_df', None)
        
        columns = df.columns.tolist()
        
        framework = st.selectbox("Select AutoML Framework", ["AutoGluon", "FLAML", "H2O AutoML", "TPOT"])
        target = st.selectbox("Select Target Column", columns, index=columns.index(st.session_state.get('target', columns[0])) if st.session_state.get('target') in columns else 0)
        st.session_state['target'] = target
        run_name = st.text_input("Run Name", value=f"{framework.lower()}_run_{int(time.time())}")
        
        # Datasets info
        st.info(f"Active Datasets - Training: {len(df)} rows | Validation: {'N/A' if valid_df is None else str(len(valid_df)) + ' rows'} | Test: {'N/A' if test_df is None else str(len(test_df)) + ' rows'}")
        
        # Framework specific options
        st.subheader(f"{framework} Configurations")
        
        # Common framework options
        seed = st.number_input("Seed (reproducibility)", value=42, min_value=0, max_value=9999)
        
        # Init vars
        time_limit = time_budget = max_runtime_secs = 60
        presets = task = metric = estimator_list = None
        nfolds = balance_classes = sort_metric = exclude_algos = None
        
        if framework == "AutoGluon":
            time_limit = st.slider("Time limit (seconds)", 30, 3600, 60)
            presets = st.selectbox("Presets", ['medium_quality', 'best_quality', 'high_quality', 'good_quality', 'optimize_for_deployment'])
        elif framework == "FLAML":
            time_budget = st.slider("Time budget (seconds)", 30, 3600, 60)
            task = st.selectbox("Task", ['classification', 'regression', 'ts_forecast', 'rank'])
            
            # Smart metric selection for FLAML
            num_classes = df[target].nunique()
            if task == 'classification':
                if num_classes > 2:
                    st.warning(f"Multiclass problem detected ({num_classes} classes).")
                    metric_options = ['auto', 'accuracy', 'macro_f1', 'micro_f1', 'roc_auc_ovr', 'roc_auc_ovo', 'log_loss']
                else:
                    metric_options = ['auto', 'accuracy', 'roc_auc', 'f1', 'log_loss']
            elif task == 'regression':
                metric_options = ['auto', 'rmse', 'mae', 'r2', 'mape']
            else:
                metric_options = ['auto']
                
            metric = st.selectbox("Metric", metric_options)
            estimators = st.multiselect("Estimators", ['lgbm', 'rf', 'catboost', 'xgboost', 'extra_tree', 'lrl1', 'lrl2'], default=['lgbm', 'rf'])
            estimator_list = estimators if estimators else 'auto'
        elif framework == "H2O AutoML":
            st.warning("⚠️ H2O AutoML requires Java. If Java is not installed, use AutoGluon or FLAML.")
            st.info("💡 To run H2O without Java installed locally, run via Docker.")
            
            max_runtime_secs = st.slider("Max runtime (seconds)", 60, 3600, 300)
            max_models = st.slider("Max number of models", 5, 50, 10)
            if cv_folds == 0:
                nfolds = st.slider("CV folds (H2O Native)", 2, 10, 3)
            else:
                nfolds = cv_folds
                st.info(f"H2O native folds logic is overriden by the global CV configuration ({cv_folds} folds).")
                
            balance_classes = st.checkbox("Balance classes", value=True)
            
            exclude_options = ['DeepLearning', 'GLM', 'GBM', 'DRF', 'XGBoost', 'GLRM']
            exclude_algos = st.multiselect("Exclude Algorithms", exclude_options, help="Algorithms to exclude from AutoML")
        elif framework == "TPOT":
            st.info("🧬 TPOT uses genetic algorithms to optimize machine learning pipelines.")
            st.warning("⚠️ TPOT can be slower, but often finds highly optimal pipelines.")
            
            generations = st.slider("Generations", 1, 20, 5, help="Number of generations for genetic evolution")
            population_size = st.slider("Population Size", 10, 100, 20, help="Population size in each generation")
            if cv_folds == 0:
                cv = st.slider("Cross Validation Folds (TPOT)", 2, 10, 5)
            else:
                cv = cv_folds
                st.info(f"TPOT CV folds override by global CV settings ({cv_folds} folds).")
            max_time_mins = st.slider("Max time (minutes)", 5, 120, 30, help="Maximum training time in minutes")
            max_eval_time_mins = st.slider("Max time per evaluation (minutes)", 1, 20, 5, help="Maximum time per pipeline evaluation")
            verbosity = st.slider("Log verbosity level", 0, 3, 2, help="TPOT feedback verbosity")
            n_jobs = st.slider("Parallel jobs", -1, 8, -1, help="Number of parallel processes (-1 to use all)")
            
            # Advanced TPOT Options
            with st.expander("⚙️ Advanced TPOT Options"):
                config_dict = st.selectbox("TPOT Configuration", [
                    'TPOT light', 'TPOT MDR', 'TPOT sparse', 'TPOT NN'
                ], help="Predefined TPOT configuration for different types of problems")
                
                tfidf_max_features = st.number_input("Text features max dimensions (TF-IDF)", min_value=100, max_value=10000, value=500, step=100)
                ngram_max = st.slider("Max text N-Gram size", 1, 3, 2, help="If 2, evaluates unigrams and bigrams. If 3, unigrams, bigrams, and trigrams.")
                tfidf_ngram_range = (1, ngram_max)
                
                # Auto problem detection
                problem_type = 'classification' if df[target].nunique() <= 20 or df[target].dtype == 'object' else 'regression'
                st.info(f"🎯 Problem type detected: **{problem_type}**")
                
                # Metrics based on problem type
                if problem_type == 'classification':
                    scoring_options = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'precision_macro', 'recall_macro']
                else:
                    scoring_options = ['neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2', 'explained_variance']
                
                scoring = st.selectbox("Optimization Metric", scoring_options, help="Metric used to optimize the pipelines")

        st.markdown("---")
        st.subheader("4. Launch Experiment")

        if st.button("🚀 Start Training", type="primary"):
            import time as _t
            _key = f"{framework.lower()}_{int(_t.time())}"

            # Build kwargs dict for the trainer
            if framework == "AutoGluon":
                _fn = train_autogluon
                _kwargs = dict(train_data=df, target=target, run_name=run_name,
                               valid_data=valid_df, test_data=test_df,
                               time_limit=time_limit, presets=presets, seed=seed, cv_folds=cv_folds)
                _fw_key = "autogluon"
            elif framework == "FLAML":
                _fn = train_flaml_model
                _kwargs = dict(train_data=df, target=target, run_name=run_name,
                               valid_data=valid_df, test_data=test_df,
                               time_budget=time_budget, task=task, metric=metric,
                               estimator_list=estimator_list, seed=seed, cv_folds=cv_folds)
                _fw_key = "flaml"
            elif framework == "H2O AutoML":
                _fn = train_h2o_model
                _kwargs = dict(train_data=df, target=target, run_name=run_name,
                               valid_data=valid_df, test_data=test_df,
                               max_runtime_secs=max_runtime_secs, max_models=max_models,
                               nfolds=nfolds, balance_classes=balance_classes,
                               seed=seed, sort_metric=sort_metric, exclude_algos=exclude_algos)
                _fw_key = "h2o"
            else:  # TPOT
                _fn = train_tpot_model
                _kwargs = dict(df=df, target_column=target, run_name=run_name,
                               valid_data=valid_df, test_data=test_df,
                               generations=generations, population_size=population_size,
                               cv=cv, scoring=scoring, max_time_mins=max_time_mins,
                               max_eval_time_mins=max_eval_time_mins,
                               random_state=seed, verbosity=verbosity, n_jobs=n_jobs,
                               config_dict=config_dict, tfidf_max_features=tfidf_max_features,
                               tfidf_ngram_range=tfidf_ngram_range)
                _fw_key = "tpot"

            _entry = ExperimentEntry(
                key=_key,
                metadata={
                    "framework": framework,
                    "framework_key": _fw_key,
                    "run_name": run_name,
                    "target": target,
                    "config_snapshot": {k: v for k, v in _kwargs.items()
                                         if k not in ("train_data", "df", "valid_data",
                                                       "valid_df", "test_data", "test_df")}
                }
            )

            _t_obj = threading.Thread(
                target=run_training_worker,
                args=(_entry, _fn, _kwargs),
                daemon=True
            )
            _entry.thread = _t_obj
            exp_manager.add(_entry)
            _t_obj.start()

            st.success(f"🚀 Experiment **{run_name}** queued! Navigate to the **Experiments** tab to monitor progress.")
            st.info("You can start another training right away or switch tabs — training runs in the background.")
    else:
        st.warning("Please upload or select Data Lake training sets first.")

elif menu == "Experiments":
    st.header("🧪 Experiments Dashboard")
    
    # Helper for cached MLflow data
    def get_run_data_cached(run_id):
        cache_key = f"ml_run_{run_id}"
        if cache_key not in st.session_state or time.time() - st.session_state[f"{cache_key}_time"] > 30:
            try:
                import mlflow
                data = mlflow.get_run(run_id)
                st.session_state[cache_key] = data
                st.session_state[f"{cache_key}_time"] = time.time()
                return data
            except Exception:
                return st.session_state.get(cache_key)
        return st.session_state.get(cache_key)

    def get_artifacts_cached(run_id):
        cache_key = f"ml_arts_{run_id}"
        if cache_key not in st.session_state or time.time() - st.session_state[f"{cache_key}_time"] > 60:
            try:
                import mlflow
                arts = mlflow.MlflowClient().list_artifacts(run_id)
                st.session_state[cache_key] = arts
                st.session_state[f"{cache_key}_time"] = time.time()
                return arts
            except Exception:
                return st.session_state.get(cache_key)
        return st.session_state.get(cache_key)

    # Wrap the entire dashboard in a fragment for non-blocking auto-refresh
    @st.fragment(run_every="3s")
    def render_experiment_dashboard():
        # Refresh all experiment statuses inside the fragment
        exp_manager.refresh_all()
        all_exps = exp_manager.get_all()

        if not all_exps:
            st.info("No experiments launched yet. Go to the **Training** tab to start one.")
            return

        # Summary Metrics
        n_running   = sum(1 for e in all_exps if e.status == "running")
        n_completed = sum(1 for e in all_exps if e.status == "completed")
        n_failed    = sum(1 for e in all_exps if e.status == "failed")
        n_cancelled = sum(1 for e in all_exps if e.status == "cancelled")

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("🟢 Running",   n_running)
        s_col2.metric("✅ Completed", n_completed)
        s_col3.metric("❌ Failed",    n_failed)
        s_col4.metric("🚫 Cancelled", n_cancelled)

        # Maintenance Section
        with st.expander("🛠️ Maintenance & Storage Management"):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                if st.button("🧹 Clear Local Models Folder", use_container_width=True, help="Deletes all folders inside /models. Safe if runs were synced to MLflow."):
                    try:
                        import shutil
                        if os.path.exists("models"):
                            shutil.rmtree("models")
                            os.makedirs("models")
                            st.success("Local models cleared!")
                        else:
                            st.info("Models folder is already empty.")
                    except Exception as me:
                        st.error(f"Cleanup error: {me}")
            
            with m_col2:
                if st.button("🔥 Reset Local MLflow (mlruns)", use_container_width=True, help="DANGER: Deletes the local mlruns folder. All local experiment history will be lost."):
                    try:
                        import shutil
                        if os.path.exists("mlruns"):
                            shutil.rmtree("mlruns")
                            st.success("Local MLflow history reset!")
                        else:
                            st.info("mlruns folder not found.")
                    except Exception as re:
                        st.error(f"Reset error: {re}")
            
            # Disk space check (Simplified)
            try:
                import shutil
                total, used, free = shutil.disk_usage(".")
                free_gb = free // (2**30)
                if free_gb < 2:
                    st.warning(f"⚠️ Low Disk Space: Only {free_gb} GB remaining. Please clear models or mlruns.")
                else:
                    st.caption(f"Disk Space: {free_gb} GB free.")
            except:
                pass

        st.markdown("---")

        for entry in all_exps:
            fw   = entry.metadata.get("framework", "Unknown")
            rname = entry.metadata.get("run_name", entry.key)
            icon  = entry.status_icon()
            elapsed = entry.elapsed_str()
            
            # Run_id from result (may not be available yet)
            run_id = None
            if entry.result and entry.result.get("run_id"):
                run_id = entry.result["run_id"]

            title = f"{icon} **{rname}** — {fw} — {elapsed}"
            subtitle = f"[{entry.status.upper()}]"
            
            # Show a pulsing heart emoji if it was updated recently to show it's alive
            if entry.status == "running" and time.time() - getattr(entry, 'last_update', 0) < 5:
                subtitle += " — 💓 Active Log"
            
            with st.expander(f"{title} {subtitle}", expanded=(entry.status == "running")):
                # --- Action buttons ---
                act_col1, act_col2, act_col3, act_col4 = st.columns([2, 1, 1, 1])

                with act_col1:
                    if run_id:
                        st.code(f"Run ID: {run_id}", language=None)
                    else:
                        st.caption(f"Key: {entry.key}")

                with act_col2:
                    if entry.status == "running":
                        if st.button("⛔ Cancel", key=f"cancel_{entry.key}"):
                            exp_manager.cancel(entry.key)
                            st.rerun()

                with act_col3:
                    if entry.status in ("completed", "cancelled", "failed"):
                        if st.button("🗑️ Delete", key=f"delete_{entry.key}"):
                            exp_manager.delete(entry.key)
                            st.rerun()

                with act_col4:
                    if entry.status == "completed" and entry.result and entry.result.get("predictor"):
                        if st.button("🔮 Load to Predict", key=f"load_{entry.key}"):
                            st.session_state['predictor']  = entry.result["predictor"]
                            st.session_state['model_type'] = entry.result.get("type", "unknown")
                            st.session_state['run_id']     = entry.result.get("run_id")
                            st.success("Model loaded! Switch to the Prediction tab.")

                # --- Tabs inside the card ---
                tab_logs, tab_metrics, tab_mlflow, tab_code = st.tabs([
                    "📋 Logs", "📈 Metrics", "🔍 MLflow Details", "💻 Consumption Code"
                ])

                with tab_logs:
                    log_text = "\n".join(entry.all_logs[-60:]) if entry.all_logs else "(No logs yet)"
                    st.code(log_text, language=None)

                with tab_metrics:
                    if entry.status == "completed" and run_id:
                        try:
                            run_data = get_run_data_cached(run_id)
                            if run_data:
                                metrics = run_data.data.metrics
                                if metrics:
                                    st.dataframe(pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()]))
                                    import matplotlib.pyplot as _plt
                                    fig, ax = _plt.subplots(figsize=(8, max(2, len(metrics) * 0.4)))
                                    ax.barh(list(metrics.keys()), list(metrics.values()))
                                    ax.set_title("Metrics")
                                    st.pyplot(fig)
                                    _plt.close(fig)
                                else:
                                    st.info("No metrics logged to MLflow yet.")
                            else:
                                st.info("Run data from MLflow is spinning up...")
                        except Exception as me:
                            st.warning(f"Could not load metrics: {me}")
                    elif entry.status == "running":
                        st.info("Training in progress — metrics will appear here.")
                    else:
                        st.info("No metrics available.")

                with tab_mlflow:
                    if run_id:
                        try:
                            run_data = get_run_data_cached(run_id)
                            if run_data:
                                st.subheader("⚙️ Parameters")
                                if run_data.data.params:
                                    st.dataframe(pd.DataFrame([{"Parameter": k, "Value": v} for k, v in run_data.data.params.items()]))
                                
                                st.subheader("📊 Metrics")
                                if run_data.data.metrics:
                                    st.dataframe(pd.DataFrame([{"Metric": k, "Value": v} for k, v in run_data.data.metrics.items()]))

                                st.subheader("📦 Artifacts")
                                try:
                                    artifacts = get_artifacts_cached(run_id)
                                    if artifacts:
                                        for art in artifacts:
                                            st.write(f"• `{art.path}` ({art.file_size or 'dir'} bytes)")
                                    else:
                                        st.info("No artifacts logged yet.")
                                except Exception as ae:
                                    st.warning(f"Could not list artifacts: {ae}")
                            else:
                                st.info("MLflow data is being fetched...")
                        except Exception as re:
                            st.warning(f"Could not load MLflow details: {re}")
                    else:
                        st.info("MLflow Run ID not available yet.")

                with tab_code:
                    if run_id:
                        try:
                            from src.code_gen_utils import generate_consumption_code, generate_api_deployment
                            fw_key = entry.metadata.get("framework_key", "unknown")
                            target_col = entry.metadata.get("target", "target")
                            code_snippet = generate_consumption_code(fw_key, run_id, target_col)
                            st.code(code_snippet, language="python")
                            
                            st.markdown("---")
                            deploy_dir = f"deploy_{entry.key}"
                            if st.button(f"🚀 Generate FastAPI Deployment", key=f"deploy_{entry.key}"):
                                generate_api_deployment(fw_key, run_id, target_col, output_dir=deploy_dir)
                                st.success(f"✅ Deployment package at `{deploy_dir}/`")
                        except Exception as ce:
                            st.warning(f"Could not generate code: {ce}")
                    else:
                        st.info("Consumption code will appear here after training completes.")

                # Framework Results
                if entry.status == "completed" and entry.result and entry.result.get("success"):
                    st.markdown("---")
                    st.subheader("🏆 Training Results")
                    fw_type = entry.result.get("type", "")
                    predictor = entry.result.get("predictor")

                    if fw_type == "autogluon" and predictor:
                        try:
                            lb = predictor.leaderboard(silent=True)
                            st.dataframe(lb)
                        except Exception as e:
                            st.warning(f"Results error: {e}")
                    elif fw_type == "flaml" and predictor:
                        st.metric("Best Estimator", predictor.best_estimator)
                        st.json(predictor.best_config)
                    elif fw_type == "h2o" and predictor:
                        if predictor.leader:
                            st.success(f"Best model: **{predictor.leader.model_id}**")
                            st.dataframe(predictor.leaderboard.as_data_frame())

                elif entry.status == "failed":
                    st.error(f"❌ Training failed: {entry.result.get('error', 'Unknown error') if entry.result else 'Unknown error'}")

    # Invoke the fragment
    render_experiment_dashboard()


elif menu == "Prediction":
    st.header("🔮 Prediction")
    
    load_option = st.radio("Choose the model source", ["Current session model", "Load from MLflow runs"])
    
    if load_option == "Load from MLflow runs":
        col1, col2 = st.columns(2)
        m_type = col1.selectbox("Model Framework", ["AutoGluon", "FLAML", "H2O AutoML", "TPOT"])
        run_id_input = col2.text_input("Run ID")
        
        if st.button("Load Model"):
            try:
                if m_type == "AutoGluon":
                    st.session_state['predictor'] = load_autogluon(run_id_input)
                    st.session_state['model_type'] = "autogluon"
                elif m_type == "FLAML":
                    st.session_state['predictor'] = load_flaml_model(run_id_input)
                    st.session_state['model_type'] = "flaml"
                elif m_type == "H2O AutoML":
                    st.session_state['predictor'] = load_h2o_model(run_id_input)
                    st.session_state['model_type'] = "h2o"
                elif m_type == "TPOT":
                    st.session_state['predictor'] = load_tpot_model(run_id_input)
                    st.session_state['model_type'] = "tpot"
                
                st.session_state['run_id'] = run_id_input
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Loading error: {e}")

    if st.session_state['predictor'] is not None:
        predictor = st.session_state['predictor']
        m_type = st.session_state['model_type']
        run_id = st.session_state.get('run_id', 'N/A')
        
        st.info(f"Active model: {m_type}")
        
        with st.expander("💻 View Model Consumption Code"):
            try:
                from src.code_gen_utils import generate_consumption_code
                code_sample = generate_consumption_code(m_type, run_id, "target")
                st.code(code_sample, language="python")
            except Exception as e:
                st.warning(f"Could not generate code sample: {e}")
        
        input_mode = st.radio("Input Mode", ["File Upload", "Manual Input"], horizontal=True)
        
        if input_mode == "File Upload":
            predict_file = st.file_uploader("Upload prediction dataset", type=["csv", "xlsx", "xls"])
            if predict_file is not None:
                predict_df = load_data(predict_file)
                st.dataframe(predict_df.head())
                execute_pred = st.button("Execute Prediction")
        else:
            st.subheader("📝 Manual Entry")
            # Try to get features from session state DF first
            features = []
            if 'df' in st.session_state and st.session_state['df'] is not None:
                # Assuming all columns except target are features
                target_col = st.session_state.get('target', None)
                features = [c for c in st.session_state['df'].columns if c != target_col]
            else:
                st.warning("Feature list unknown (Training data not in session). Please upload a file once to identify features, or use File Upload.")
                features = []
            
            if features:
                manual_data = {}
                cols = st.columns(min(len(features), 3))
                for i, feat in enumerate(features):
                    with cols[i % 3]:
                        # Basic guess of type based on training data
                        dtype = st.session_state['df'][feat].dtype
                        if pd.api.types.is_numeric_dtype(dtype):
                            manual_data[feat] = st.number_input(feat, value=float(st.session_state['df'][feat].median()))
                        else:
                            options = st.session_state['df'][feat].unique().tolist()
                            manual_data[feat] = st.selectbox(feat, options)
                
                predict_df = pd.DataFrame([manual_data])
                execute_pred = st.button("Confirm Manual Prediction")
            else:
                execute_pred = False

        if execute_pred:
                try:
                    # Validate predictor payload
                    if predictor is None:
                        st.error("No model is loaded. Please train or load a model first.")
                        st.stop()
                    
                    if m_type == "autogluon":
                        predictions = predictor.predict(predict_df)
                    elif m_type == "h2o":
                        from src.h2o_utils import predict_with_h2o
                        predictions = predict_with_h2o(predictor, predict_df)
                    else: # flaml
                        predictions = predictor.predict(predict_df)
                    
                    result_df = predict_df.copy()
                    result_df['Predictions'] = predictions
                    
                    st.success("Predictions concluded!")
                    st.dataframe(result_df)
                    
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

elif menu == "History (MLflow)":
    st.header("📊 Experiments History")
    
    # Button to clean corrupted MLflow metadata
    if st.sidebar.button("Hard Reset MLflow (Repair MLRuns tracking)"):
        import shutil
        if os.path.exists("mlruns"):
            # Instead of deleting everything, we could try to find the malformed ones
            # but deleting is safer for a local "repair"
            shutil.rmtree("mlruns")
            st.sidebar.success("Cache cleared! Please restart your training processes.")
            st.rerun()

    # Soft cache clear
    if st.sidebar.button("Clear Python MLflow Cache"):
        mlflow_cache.clear_cache()
        st.sidebar.success("Cache cleared!")
        st.rerun()

    # Cached experiment list
    experiment_list = get_cached_experiment_list()
    exp_name = st.selectbox("Select Experiment Node", experiment_list)
    
    try:
        # Request cached runs
        runs = mlflow_cache.get_cached_all_runs(exp_name)
        
        if not runs.empty:
            # Clean up columns for better display
            display_runs = runs.copy()
            
            st.subheader("🏁 Run Selection & Comparison")
            
            # Allow selecting runs for comparison
            selected_run_ids = st.multiselect("Select runs to compare", runs['run_id'].tolist(), help="Select multiple runs to see a metric comparison chart.")
            
            if selected_run_ids:
                comparison_df = runs[runs['run_id'].isin(selected_run_ids)]
                
                # Identify metric columns
                metric_cols = [col for col in comparison_df.columns if col.startswith('metrics.')]
                
                if metric_cols:
                    st.write("### 📈 Metric Comparison")
                    # Prepare data for plotting
                    plot_data = comparison_df.set_index('run_id')[metric_cols]
                    # Remove 'metrics.' prefix for cleaner labels
                    plot_data.columns = [c.replace('metrics.', '') for c in plot_data.columns]
                    
                    st.bar_chart(plot_data)
                else:
                    st.info("No metrics found for the selected runs.")
                
                # Model Registration
                st.subheader("📑 Model Registration")
                reg_col1, reg_col2 = st.columns([2, 1])
                with reg_col1:
                    model_to_reg = st.selectbox("Select run to register", selected_run_ids)
                with reg_col2:
                    reg_name = st.text_input("Registration Name", value="best_model")
                
                if st.button("Register Model in MLflow Registry"):
                    try:
                        # Extract the actual run object or just use ID
                        model_uri = f"runs:/{model_to_reg}/model"
                        reg_info = mlflow.register_model(model_uri, reg_name)
                        st.success(f"Successfully registered model '{reg_name}' (Version {reg_info.version})")
                    except Exception as e:
                        st.error(f"Registration error: {e}")
                
                # Model API Deployment Generator
                st.subheader("🚀 One-Click API Deployment")
                api_col1, api_col2 = st.columns([2, 1])
                with api_col1:
                    model_to_deploy = st.selectbox("Select run to deploy as API", selected_run_ids)
                
                if st.button("Generate FastAPI Deployment Package"):
                    try:
                        from src.code_gen_utils import generate_api_deployment
                        
                        # Find the model_type and target for this run
                        run_info = runs[runs['run_id'] == model_to_deploy].iloc[0]
                        run_model_type = run_info.get('params.model_type', 'unknown')
                        run_target = run_info.get('params.target', 'target')
                        
                        deploy_dir = f"deploy_{model_to_deploy[:8]}"
                        
                        generate_api_deployment(run_model_type, model_to_deploy, run_target, output_dir=deploy_dir)
                        st.success(f"✅ Deployment package generated successfully in folder: `{deploy_dir}/`")
                        with st.expander("Show instructions"):
                            st.write("1. Open your terminal in the generated folder.")
                            st.code(f"cd {deploy_dir}", language="bash")
                            st.write("2. Build and run via Docker (Recommended):")
                            st.code(f"docker build -t ml-api:{model_to_deploy[:8]} .\ndocker run -p 8000:8000 ml-api:{model_to_deploy[:8]}", language="bash")
                    except Exception as e:
                        st.error(f"Failed to generate API deployment: {e}")

            st.markdown("---")
            st.subheader("📋 All Runs Data")
            st.dataframe(runs)
            
            # Cache statistics insight
            with st.expander("📊 Cache Statistics"):
                st.write(f"Experiment: {exp_name}")
                st.write(f"Total runs: {len(runs)}")
                st.write(f"Cache TTL cycle: 5 minutes")
        else:
            st.write("No recorded runs found for this experiment tracking node.")
    except Exception as e:
        st.error(f"Error reading MLflow cache: {e}")
        st.warning("This is commonly caused by corrupted trailing database traces or manually deleted runs folders. Use the Hard Reset button to fix locally.")
