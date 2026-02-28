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
import mlflow
import time

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

st.title("🚀 AutoML Multi-Framework Interface")

# Sidebar navigation
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Menu", ["Data Upload", "Training", "Prediction", "History (MLflow)"])

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
        target = st.selectbox("Select Target Column", columns)
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

        if st.button("Start Training"):
            st.subheader("📺 Real-time Monitoring")
            
            col_logs, col_chart = st.columns([1, 1])
            
            with col_logs:
                st.write("📋 Training Logs")
                log_placeholder = st.empty()
            
            with col_chart:
                st.write("📈 Performance Evolution")
                chart_placeholder = st.empty()
            
            # Shared state for thread communication
            import threading
            training_done = threading.Event()
            log_queue = queue.Queue()
            result_queue = queue.Queue()
            
            # Custom Log Handler for the thread
            class ThreadLogHandler(logging.Handler):
                def emit(self, record):
                    msg = self.format(record)
                    log_queue.put(msg)
            
            # Setup logger to capture all framework output
            logger = logging.getLogger()
            for h in logger.handlers[:]:
                logger.removeHandler(h)
                
            handler = ThreadLogHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
            for lib in ['flaml', 'autogluon', 'mlflow', 'h2o', 'tpot']:
                l = logging.getLogger(lib)
                l.addHandler(handler)
                l.setLevel(logging.INFO)
            
            # Training function to run in thread
            def run_training():
                # Redirect stdout and stderr to capture everything
                import io
                from contextlib import redirect_stdout, redirect_stderr
                
                class LogIO(io.StringIO):
                    def write(self, s):
                        if s.strip():
                            log_queue.put(s.strip())
                        return super().write(s)

                with redirect_stdout(LogIO()), redirect_stderr(LogIO()):
                    try:
                        if framework == "AutoGluon":
                            res_predictor, res_run_id = train_autogluon(df, target, run_name, valid_df, test_df, time_limit, presets, seed, cv_folds)
                            result_queue.put({"predictor": res_predictor, "run_id": res_run_id, "type": "autogluon", "success": True})
                        elif framework == "FLAML":
                            res_automl, res_run_id = train_flaml_model(df, target, run_name, valid_df, test_df, time_budget, task, metric, estimator_list, seed, cv_folds)
                            result_queue.put({"predictor": res_automl, "run_id": res_run_id, "type": "flaml", "success": True})
                        elif framework == "H2O AutoML":
                            res_automl, res_run_id = train_h2o_model(
                                df, target, run_name, 
                                valid_data=valid_df, test_data=test_df,
                                max_runtime_secs=max_runtime_secs, 
                                max_models=max_models, 
                                nfolds=nfolds, 
                                balance_classes=balance_classes, 
                                seed=seed, 
                                sort_metric=sort_metric, 
                                exclude_algos=exclude_algos
                            )
                            result_queue.put({"predictor": res_automl, "run_id": res_run_id, "type": "h2o", "success": True})
                        elif framework == "TPOT":
                            res_tpot, res_pipeline, res_run_id, res_info = train_tpot_model(
                                df, target, run_name,
                                valid_data=valid_df, test_data=test_df,
                                generations=generations,
                                population_size=population_size,
                                cv=cv,
                                scoring=scoring,
                                max_time_mins=max_time_mins,
                                max_eval_time_mins=max_eval_time_mins,
                                random_state=seed,
                                verbosity=verbosity,
                                n_jobs=n_jobs,
                                config_dict=config_dict,
                                tfidf_max_features=tfidf_max_features,
                                tfidf_ngram_range=tfidf_ngram_range
                            )
                            result_queue.put({"predictor": res_tpot, "pipeline": res_pipeline, "run_id": res_run_id, "info": res_info, "type": "tpot", "success": True})
                    except Exception as e:
                        import traceback
                        error_msg = f"CRITICAL TRAINING ERROR: {str(e)}\n{traceback.format_exc()}"
                        log_queue.put(error_msg)
                        result_queue.put({"success": False, "error": str(e)})
                    finally:
                        training_done.set()
                        logger.removeHandler(handler)

            # Start training
            try:
                thread = threading.Thread(target=run_training)
                thread.start()
                
                # UI Update Loop
                all_logs = []
                performance_history = []
                last_pos = 0
                
                # Clear old flaml.log if exists
                if os.path.exists("flaml.log"):
                    try:
                        os.remove("flaml.log")
                    except:
                        pass

                while not training_done.is_set():
                    # 1. Capture logs from the queue (Standard logging/stdout)
                    new_logs = False
                    while not log_queue.empty():
                        log_line = log_queue.get()
                        # Filter out the annoying FLAML low-cost warning from UI logs
                        if "low-cost partial config" in log_line.lower():
                            continue
                            
                        if log_line not in all_logs:
                            all_logs.append(log_line)
                            new_logs = True
                    
                    # 2. Capture logs from flaml.log file (Iterative progress)
                    if os.path.exists("flaml.log"):
                        try:
                            with open("flaml.log", "r") as f:
                                f.seek(last_pos)
                                new_lines = f.readlines()
                                last_pos = f.tell()
                                if new_lines:
                                    for line in new_lines:
                                        clean_line = line.strip()
                                        # Filter warning from file logs too
                                        if "low-cost partial config" in clean_line.lower():
                                            continue
                                            
                                        if clean_line and clean_line not in all_logs:
                                            all_logs.append(clean_line)
                                            new_logs = True
                                            
                                            # 3. Parse performance metrics
                                            if any(kw in clean_line.lower() for kw in ["loss", "accuracy", "score"]):
                                                import re
                                                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", clean_line)
                                                if numbers:
                                                    try:
                                                        val = float(numbers[-1])
                                                        if 0 <= abs(val) <= 1000:
                                                            performance_history.append(val)
                                                            chart_placeholder.line_chart(performance_history)
                                                    except:
                                                        pass
                        except:
                            pass

                    if new_logs:
                        # Display only the most relevant recent logs
                        log_placeholder.code("\n".join(all_logs[-20:]))
                    
                    time.sleep(0.5)
                
                # Get final results from queue
                final_result = result_queue.get()
                
                if final_result["success"]:
                    st.session_state['predictor'] = final_result["predictor"]
                    st.session_state['run_id'] = final_result["run_id"]
                    st.session_state['model_type'] = final_result["type"]
                    st.success(f"Training completed successfully! Run ID: {final_result['run_id']}")
                    
                    # Log DVC hashes to MLflow run
                    if 'dvc_hashes' in st.session_state and st.session_state['dvc_hashes']:
                        try:
                            with mlflow.start_run(run_id=final_result["run_id"]):
                                mlflow.log_params(st.session_state['dvc_hashes'])
                            st.info("🧬 Data Lake (DVC) metadata successfully attached to Run!")
                        except Exception as e:
                            st.warning(f"Could not save DVC hashes to MLflow: {e}")
                            
                else:
                    st.error(f"Training failed: {final_result['error']}")

                # Show all logs at the end
                while not log_queue.empty():
                    all_logs.append(log_queue.get())
                
                if all_logs:
                    with st.expander("View Full Training Logs"):
                        st.code("\n".join(all_logs))
                
                # Post-training visualizations
                if final_result["success"]:
                    if final_result['type'] == "flaml":
                        predictor = final_result['predictor']
                        
                        st.subheader("🏆 Best Model (FLAML)")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Best Estimator", predictor.best_estimator)
                        col2.metric("Best Loss", f"{predictor.best_loss:.4f}")
                        col3.metric("Best Iteration", predictor.best_iteration)
                        
                        with st.expander("⚙️ Best Configuration (Hyperparameters)"):
                            st.json(predictor.best_config)
                            
                        if hasattr(predictor, 'best_config_per_estimator') and predictor.best_config_per_estimator:
                            with st.expander("📊 Best Configurations per Estimator"):
                                st.json(predictor.best_config_per_estimator)
                            
                        if hasattr(predictor, 'feature_importances_') and predictor.feature_importances_ is not None:
                            try:
                                fig, ax = plt.subplots()
                                importances = predictor.feature_importances_
                                if hasattr(predictor, 'feature_names_in_'):
                                    feature_names = predictor.feature_names_in_
                                else:
                                    feature_names = df.drop(columns=[target]).columns
                                
                                if len(importances) == len(feature_names):
                                    feat_importances = pd.Series(importances, index=feature_names)
                                    feat_importances.nlargest(10).plot(kind='barh', ax=ax)
                                    plt.title("Top 10 Feature Importances (FLAML)")
                                    st.pyplot(fig)
                                else:
                                    st.info(f"Feature importance available, but columns mismatch ({len(importances)} vs {len(feature_names)}).")
                            except Exception as feat_err:
                                st.warning(f"Error generating importance chart: {feat_err}")
                    elif final_result['type'] == "autogluon":
                        predictor = final_result['predictor']
                        st.subheader("🏆 AutoGluon Results")
                        
                        st.subheader("Final Leaderboard")
                        leaderboard = predictor.leaderboard(silent=True)
                        st.dataframe(leaderboard)
                        
                        best_model = leaderboard.iloc[0]['model'] if not leaderboard.empty else "Modelo principal"
                        st.success(f"Best model found: **{best_model}**")
                        
                        with st.expander("⚙️ Training Details (AutoGluon Info)"):
                            try:
                                info = predictor.info()
                                st.json(info)
                            except:
                                st.write("Detailed info not available for this model.")
                        
                        if st.checkbox("Generate Feature Importance (AutoGluon)"):
                            with st.spinner("Calculating importance (this may take a while)..."):
                                try:
                                    fi = predictor.feature_importance(df)
                                    st.dataframe(fi)
                                    fig, ax = plt.subplots()
                                    fi['importance'].nlargest(10).plot(kind='barh', ax=ax)
                                    plt.title("Feature Importance (AutoGluon)")
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error calculating importance: {e}")
                    
                    elif final_result['type'] == "h2o":
                        automl = final_result['predictor']
                        st.subheader("🏆 H2O AutoML Results")
                        
                        # Verify if H2O is still connected before accessing the model
                        try:
                            best_model = automl.leader
                            if best_model is not None:
                                st.success(f"Best model found: **{best_model.model_id}**")
                                
                                st.subheader("Final Leaderboard")
                                try:
                                    leaderboard = automl.leaderboard.as_data_frame()
                                    st.dataframe(leaderboard)
                                except Exception as e:
                                    st.warning(f"Could not display leaderboard: {e}")
                                    # Fallback to textual representation
                                    try:
                                        st.text(str(automl.leaderboard.head(10)))
                                    except:
                                        st.info("Leaderboard unavailable (H2O connection closed)")
                                
                                with st.expander("⚙️ Best Model Details (H2O)"):
                                    try:
                                        model_params = {
                                            "model_id": best_model.model_id,
                                            "algo": best_model.algo,
                                            "model_type": best_model._model_json["output"]["model_category"]
                                        }
                                        st.json(model_params)
                                    except Exception as e:
                                        st.warning(f"Could not retrieve model details: {e}")
                            else:
                                st.warning("⚠️ No models were trained during this execution.")
                                st.info("This might happen when:")
                                st.info("• The max runtime is severely constrained for the dataset size")
                                st.info("• The data format was rejected by the active algorithms")
                                st.info("• Bad algorithm exclusion constraints")
                                
                                # Try showing fallback info
                                try:
                                    st.subheader("📊 Training Information")
                                    st.info(f"• Type: H2O AutoML")
                                    st.info(f"• Run ID: {final_result['run_id']}")
                                    st.info(f"• Status: Completed, but without trained models")
                                    st.info(f"• Recommendation: Increase maximum runtime or decrease data constraints")
                                except:
                                    pass
                        except Exception as e:
                            st.error(f"⚠️ Could not access H2O model details: {e}")
                            st.info("This commonly happens when the H2O local cluster terminates after training. Check MLflow UI for saved metrics!")
                            
                            # Fallback training info
                            try:
                                st.info(f"📊 **Training Information:**")
                                st.info(f"• Type: H2O AutoML")
                                st.info(f"• Run ID: {final_result['run_id']}")
                                st.info(f"• Status: Completed successfully")
                                st.info(f"• Metrics properly recorded in MLflow")
                            except:
                                pass
                    
                    elif final_result['type'] == "tpot":
                        tpot = final_result['predictor']
                        pipeline = final_result['pipeline']
                        info = final_result['info']
                        
                        st.subheader("🧬 TPOT AutoML Results")
                        
                        # General information
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Problem Type", info['problem_type'].title())
                        col2.metric("Generations", info['generations'])
                        col3.metric("Population", info['population_size'])
                        col4.metric("Features", info['n_features'])
                        
                        # Metrics
                        if info['problem_type'] == 'classification':
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Accuracy", f"{info.get('accuracy', 0):.4f}")
                            col2.metric("F1 Macro", f"{info.get('f1_macro', 0):.4f}")
                            col3.metric("F1 Weighted", f"{info.get('f1_weighted', 0):.4f}")
                        else:
                            col1, col2, col3 = st.columns(3)
                            col1.metric("RMSE", f"{info.get('rmse', 0):.4f}")
                            col2.metric("R²", f"{info.get('r2', 0):.4f}")
                            col3.metric("MSE", f"{info.get('mse', 0):.4f}")
                        
                        # Optimized pipeline
                        with st.expander("🧬 Optimized Pipeline"):
                            st.code(str(tpot.fitted_pipeline_), language="python")
                        
                        # Detailed information
                        with st.expander("📊 Detailed Information"):
                            st.json(info)
                        
                        # Training time
                        st.info(f"⏱️ **Training Duration:** {info['training_duration']:.2f} seconds")
                        st.info(f"🎯 **Optimization Metric:** {info['scoring']}")

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"Error during training: {e}")
                with st.expander("View error details (Traceback)"):
                    st.code(error_details)
            finally:
                pass
    else:
        st.warning("Please upload or select Data Lake training sets first.")

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
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Loading error: {e}")

    if st.session_state['predictor'] is not None:
        predictor = st.session_state['predictor']
        m_type = st.session_state['model_type']
        
        st.info(f"Active model: {m_type}")
        
        predict_file = st.file_uploader("Upload prediction dataset", type=["csv", "xlsx", "xls"])
        
        if predict_file is not None:
            predict_df = load_data(predict_file)
            st.dataframe(predict_df.head())
            
            if st.button("Execute Prediction"):
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
