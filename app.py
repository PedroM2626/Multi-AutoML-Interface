import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import sys
import logging
import importlib
import queue


def _compat_fragment(*args, **kwargs):
    """Use st.fragment when available, otherwise behave as a no-op decorator."""
    fragment_fn = getattr(st, "fragment", None)
    if fragment_fn is not None:
        return fragment_fn(*args, **kwargs)

    def _decorator(func):
        return func

    # Support bare decorator usage: @_compat_fragment
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return _decorator

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
def cached_load_data(file_path_or_obj, no_header=False):
    from src.data_utils import load_data
    return load_data(file_path_or_obj, no_header=no_header)

@st.cache_data
def cached_get_data_summary(df):
    from src.data_utils import get_data_summary
    return get_data_summary(df)

@st.cache_data(ttl=60) # 1 Minute Cache for file list
def cached_get_data_lake_files():
    from src.data_utils import get_data_lake_files
    return get_data_lake_files()

# ── EDA stat helpers (cached per-DataFrame) ───────────────────────────────────
@st.cache_data(show_spinner=False)
def _compute_missing_stats(df: pd.DataFrame) -> pd.Series:
    return df.isnull().mean().sort_values(ascending=False) * 100

@st.cache_data(show_spinner=False)
def _compute_type_counts(df: pd.DataFrame) -> pd.Series:
    return df.dtypes.astype(str).value_counts()

@st.cache_data(show_spinner=False)
def _compute_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns.tolist(),
        "Type":   df.dtypes.astype(str).tolist(),
        "Missing": df.isnull().sum().tolist(),
        "Unique":  df.nunique().tolist(),
    })

@st.cache_data(show_spinner=False)
def _compute_overview_stats(df: pd.DataFrame):
    """Returns (missing_pct_mean, memory_mb) — avoids scanning DF on every rerun."""
    missing = df.isnull().mean().mean() * 100
    memory  = df.memory_usage(deep=True).sum() / 1024 ** 2
    return missing, memory

# ── Cached matplotlib figures ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _make_missing_fig(miss_series: pd.Series):
    import matplotlib.pyplot as _plt
    miss_df = miss_series[miss_series > 0]
    if len(miss_df) == 0:
        return None
    fig, ax = _plt.subplots(figsize=(9, max(2.5, len(miss_df) * 0.4)))
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#0d1117")
    ax.barh(miss_df.index.tolist(), miss_df.tolist(),
            color=["#f85149" if v > 30 else "#d29922" for v in miss_df.tolist()],
            edgecolor="#30363d")
    ax.set_xlabel("Missing %", color="#8b949e")
    ax.set_title("Missing Values per Column", color="#f0f6fc", fontsize=11)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    _plt.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def _make_type_pie(type_counts: pd.Series):
    import matplotlib.pyplot as _plt
    colors_t = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f85149"]
    fig, ax = _plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#161b22")
    _, _, autotexts = ax.pie(
        type_counts.values, labels=type_counts.index.tolist(),
        colors=colors_t[:len(type_counts)], autopct="%1.1f%%",
        textprops={"color": "#c9d1d9", "fontsize": 10}
    )
    for w in autotexts: w.set_color("#f0f6fc")
    ax.set_title("Column Data Types", color="#f0f6fc", fontsize=11)
    _plt.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def _make_dist_fig(col_data: pd.Series, col_name: str):
    import matplotlib.pyplot as _plt
    fig, ax = _plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#0d1117")
    ax.hist(col_data.dropna(), bins=40, color="#58a6ff", edgecolor="#30363d", linewidth=0.4, alpha=0.85)
    ax.set_title(f"Distribution: {col_name}", color="#f0f6fc", fontsize=11)
    ax.set_xlabel(col_name, color="#8b949e"); ax.set_ylabel("Count", color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    _plt.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def _make_metrics_bar(metrics_items: tuple):
    """metrics_items = tuple of (key, value) pairs — hashable for cache."""
    import matplotlib.pyplot as _plt
    import matplotlib.ticker as _mticker
    keys   = [k for k, _ in metrics_items]
    values = [v for _, v in metrics_items]
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in values]
    fig, ax = _plt.subplots(figsize=(9, max(2.5, len(keys) * 0.45)))
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#0d1117")
    ax.barh(keys, values, color=colors, edgecolor="#30363d", linewidth=0.5)
    ax.set_title("MLflow Metrics", color="#f0f6fc", fontsize=12, pad=12)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.xaxis.set_major_formatter(_mticker.FormatStrFormatter("%.4g"))
    _plt.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def _make_leaderboard_bar(labels: tuple, values: tuple, xlabel: str, title: str, color: str):
    """Generic horizontal bar chart for leaderboard tables."""
    import matplotlib.pyplot as _plt
    fig, ax = _plt.subplots(figsize=(9, max(2.5, len(labels) * 0.45)))
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#0d1117")
    ax.barh(list(labels), list(values), color=color, edgecolor="#30363d")
    ax.set_xlabel(xlabel, color="#8b949e")
    ax.set_title(title, color="#f0f6fc", fontsize=11)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    _plt.tight_layout()
    return fig

# ── MLflow data getters (top-level so @st.cache_data is effective) ────────────
@st.cache_data(ttl=30, show_spinner=False)
def _get_mlflow_run(run_id: str):
    return mlflow.get_run(run_id)

@st.cache_data(ttl=60, show_spinner=False)
def _get_mlflow_artifacts(run_id: str):
    return mlflow.MlflowClient().list_artifacts(run_id)

# ── Disk usage (cached 30 s to avoid high-frequency I/O in the 5 s fragment) ──
@st.cache_data(ttl=30, show_spinner=False)
def _get_disk_usage():
    import shutil
    return shutil.disk_usage(".")

# ── Log HTML builder (cached by content — avoids rebuilding every 5 s) ────────
@st.cache_data(show_spinner=False, max_entries=100)
def _build_log_html(log_tuple: tuple, max_lines: int = 80) -> str:
    keywords_error   = ["error", "exception", "traceback", "critical", "failed", "errno"]
    keywords_warning = ["warning", "warn", "deprecated", "no space", "could not"]
    keywords_success = ["success", "complete", "best model", "finished", "saved", "logged"]
    keywords_info    = ["info:", "[worker]", "starting", "initialized", "loading", "fitting"]
    keywords_metric  = ["accuracy", "f1", "score", "auc", "rmse", "mse", "r2", "loss"]
    lines_html = []
    for line in log_tuple[-max_lines:]:
        ll = line.lower()
        if any(k in ll for k in keywords_error):
            cls = "log-line-error"
        elif any(k in ll for k in keywords_warning):
            cls = "log-line-warning"
        elif any(k in ll for k in keywords_success):
            cls = "log-line-success"
        elif any(k in ll for k in keywords_metric):
            cls = "log-line-metric"
        elif any(k in ll for k in keywords_info):
            cls = "log-line-info"
        else:
            cls = "log-line-normal"
        safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines_html.append(f'<div class="{cls}">{safe_line}</div>')
    return '<div class="log-panel">' + "".join(lines_html) + '</div>'

# ── Pipeline steps (cached so log parsing doesn't run on every 5 s tick) ──────
@st.cache_data(show_spinner=False, max_entries=200)
def _get_pipeline_steps(framework_key: str, log_tuple: tuple, status: str):
    from src.pipeline_parser import infer_pipeline_steps
    return infer_pipeline_steps(framework_key, list(log_tuple), status)

@st.cache_data(show_spinner=False, max_entries=50)
def _get_column_nunique(df: pd.DataFrame, col: str) -> int:
    """Cached nunique for target column — avoids scanning the column on every rerun."""
    return int(df[col].nunique()) if col in df.columns else 2

from src.log_utils import setup_logging_to_queue, StdoutRedirector
from src.mlflow_utils import heal_mlruns
from src.mlflow_cache import mlflow_cache, get_cached_experiment_list
from src.experiment_manager import get_or_create_manager, ExperimentEntry
from src.training_worker import run_training_worker
import mlflow
import time
import threading

st.set_page_config(
    page_title="Multi-AutoML Interface",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Premium CSS Design System ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base & Reset ─────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #080c12 !important; color: #c9d1d9 !important; }

/* remove default streamlit header padding */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1400px; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050b19 0%, #091429 100%) !important;
    border-right: 1px solid #1f324e !important;
    min-width: 260px;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: #c9d1d9 !important; }

/* sidebar brand */
.sidebar-brand {
    background: linear-gradient(135deg, #071225 0%, #0d213f 100%);
    border-bottom: 1px solid #1f3e69;
    padding: 28px 20px 22px;
    margin: -16px -16px 20px;
    position: relative;
    overflow: hidden;
}
.sidebar-brand::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #2563eb, #38bdf8, #60a5fa, #2563eb);
    background-size: 300% 100%;
    animation: brand-shimmer 4s linear infinite;
}
@keyframes brand-shimmer { 0%{background-position:0% 0%} 100%{background-position:300% 0%} }
.sidebar-brand-logo { font-size: 32px; margin-bottom: 8px; }
.sidebar-brand-title {
    font-size: 18px; font-weight: 700;
    background: linear-gradient(90deg, #e2edff, #8bd1ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: 4px;
}
.sidebar-brand-sub { font-size: 11px; color: #8ca7c7; letter-spacing: 0.08em; text-transform: uppercase; }

.sidebar-nav-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #7f97b8;
    text-transform: uppercase;
    margin: 6px 0 8px;
}

/* Sidebar navigation list (styled radio) */
[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    margin: 0;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 14px;
    padding: 10px 12px;
    transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: #0c1a33;
    border-color: #1f3962;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
    margin: 0;
    color: #91add3;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.01em;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input[type="radio"]:checked) {
    background: linear-gradient(90deg, #122949 0%, #173258 100%);
    border-color: #3a8ed8;
    box-shadow: inset 0 0 0 1px #1f5ea0;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input[type="radio"]:checked) [data-testid="stMarkdownContainer"] p {
    color: #e5f0ff;
}
[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
    display: none;
}

/* sidebar separator */
.sidebar-sep {
    font-size: 10px; font-weight: 600; color: #374151;
    text-transform: uppercase; letter-spacing: 0.12em;
    padding: 12px 0 6px;
    border-top: 1px solid #1e2736;
    margin-top: 8px;
}

/* ── Page Title (replaces main-header) ───────────────────── */
.page-title {
    display: flex; align-items: center; gap: 14px;
    padding: 0 0 20px;
    border-bottom: 1px solid #1e2736;
    margin-bottom: 24px;
}
.page-title-icon {
    width: 48px; height: 48px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    background: linear-gradient(135deg, #1e1b4b, #1e3a5f);
    border: 1px solid #3730a3;
    flex-shrink: 0;
}
.page-title-text h1 {
    font-size: 22px; font-weight: 700; color: #f0f6fc; margin: 0 0 2px;
    line-height: 1.2;
}
.page-title-text p { font-size: 13px; color: #6b7280; margin: 0; }

/* ── Cards ───────────────────────────────────────────────── */
.stat-card {
    background: linear-gradient(135deg, #0f1729 0%, #111c30 100%);
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
}
.stat-card:hover {
    border-color: #4f46e5;
    box-shadow: 0 4px 24px #4f46e520;
    transform: translateY(-2px);
}
.stat-card .number { font-size: 38px; font-weight: 700; color: #f0f6fc; line-height: 1; }
.stat-card .label  { font-size: 11px; color: #6b7280; margin-top: 8px; text-transform: uppercase; letter-spacing: 0.1em; }

/* ── Status Badges ───────────────────────────────────────── */
.badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 11px; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.badge-running   { background: #052e16; color: #4ade80; border: 1px solid #166534; animation: pulse-green 2s ease-in-out infinite; }
.badge-completed { background: #0c1a3d; color: #60a5fa; border: 1px solid #1e40af; }
.badge-failed    { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; }
.badge-cancelled { background: #18181b; color: #71717a; border: 1px solid #27272a; }
.badge-queued    { background: #1c1007; color: #fbbf24; border: 1px solid #78350f; }
@keyframes pulse-green { 0%,100%{box-shadow:0 0 0 0 #4ade8040} 50%{box-shadow:0 0 0 5px #4ade8010} }

/* ── Framework Badges ────────────────────────────────────── */
.fw-badge { display:inline-block; padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700; }
.fw-autogluon { background: linear-gradient(135deg,#0c2340,#0f3460); color:#60a5fa; border:1px solid #1e40af; }
.fw-flaml     { background: linear-gradient(135deg,#0a1628,#0d2348); color:#7dd3fc; border:1px solid #1e4e8c; }
.fw-h2o       { background: linear-gradient(135deg,#052e16,#064e24); color:#4ade80; border:1px solid #166534; }
.fw-tpot      { background: linear-gradient(135deg,#2d0a4a,#3b0f63); color:#c084fc; border:1px solid #7e22ce; }
.fw-pycaret   { background: linear-gradient(135deg,#2d0a1b,#3c0e25); color:#fbcfe8; border:1px solid #be185d; }
.fw-lale      { background: linear-gradient(135deg,#0f1f2e,#1a3650); color:#bae6fd; border:1px solid #0284c7; }

/* ── Pipeline Visualizer ─────────────────────────────────── */
.pipeline-container {
    display: flex; align-items: center; gap: 0;
    padding: 20px 4px; overflow-x: auto;
    background: #0b1120; border-radius: 12px;
    border: 1px solid #1e2736;
    margin: 8px 0 16px;
}
.pipeline-step {
    display: flex; flex-direction: column; align-items: center;
    min-width: 110px; position: relative;
}
.pipeline-step-icon {
    width: 46px; height: 46px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; border: 2px solid #1e2736;
    background: #0b1120; z-index: 1; transition: all 0.3s;
}
.pipeline-step-icon.done     { background:#052e16; border-color:#166534; }
.pipeline-step-icon.active   { background:#0c1a3d; border-color:#3b82f6; box-shadow:0 0 18px #3b82f660; animation:glow-blue 2s ease-in-out infinite; }
.pipeline-step-icon.pending  { opacity:0.45; }
.pipeline-step-icon.failed   { background:#2d0a0a; border-color:#7f1d1d; }
.pipeline-step-icon.cancelled{ background:#18181b; border-color:#3f3f46; }
@keyframes glow-blue { 0%,100%{box-shadow:0 0 10px #3b82f650} 50%{box-shadow:0 0 26px #3b82f690} }

.pipeline-step-label { font-size:10px; text-align:center; margin-top:8px; color:#6b7280; max-width:90px; line-height:1.3; }
.pipeline-step-label.active { color:#60a5fa; font-weight:600; }
.pipeline-step-label.done   { color:#4ade80; }
.pipeline-step-label.failed { color:#f87171; }

.pipeline-connector { flex:1; height:2px; min-width:20px; max-width:44px; background:#1e2736; margin-top:-20px; }
.pipeline-connector.done   { background: linear-gradient(90deg,#166534,#4ade80); }
.pipeline-connector.active { background: linear-gradient(90deg,#166534,#3b82f6); }

/* ── Log Panel ───────────────────────────────────────────── */
.log-panel {
    background: #020408;
    border: 1px solid #1e2736;
    border-radius: 10px;
    padding: 16px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px; line-height: 1.65;
    max-height: 360px; overflow-y: auto;
}
.log-line-normal  { color: #64748b; }
.log-line-success { color: #4ade80; }
.log-line-warning { color: #fbbf24; }
.log-line-error   { color: #f87171; }
.log-line-info    { color: #60a5fa; }
.log-line-metric  { color: #c084fc; }
.log-panel::-webkit-scrollbar { width:5px; }
.log-panel::-webkit-scrollbar-track { background:#0b1120; }
.log-panel::-webkit-scrollbar-thumb { background:#1e2736; border-radius:3px; }
.log-panel::-webkit-scrollbar-thumb:hover { background:#3b82f6; }

/* ── Experiment Card ─────────────────────────────────────── */
.exp-timer { font-family:'JetBrains Mono',monospace; font-size:11px; color:#fbbf24; }

/* ── Metric Pills ────────────────────────────────────────── */
.metric-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: #0f1729; border: 1px solid #1e2d45;
    border-radius: 10px; padding: 12px 18px; margin: 4px;
}
.metric-pill .m-label { font-size:11px; color:#4b5563; text-transform:uppercase; letter-spacing:0.08em; }
.metric-pill .m-value { font-size:20px; font-weight:700; color:#e2e8f0; }

/* ── Preview Card ────────────────────────────────────────── */
.preview-card {
    background: #0f1729; border: 1px solid #1e2d45;
    border-radius: 12px; padding: 18px;
    margin-bottom: 10px; transition: border-color 0.2s;
}
.preview-card:hover { border-color: #4f46e5; }
.preview-card h4 { color:#e2e8f0; font-size:13px; font-weight:600; margin:0 0 6px; }
.preview-card p  { color:#6b7280; font-size:12px; margin:0; line-height:1.5; }

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important; font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important; transition: all 0.2s !important;
    font-size: 13px !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border: none !important; color: white !important;
    box-shadow: 0 4px 14px #4f46e540 !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px #4f46e570 !important;
    transform: translateY(-1px);
}
.stButton > button[kind="secondary"] {
    background: #0f1729 !important; border: 1px solid #1e2d45 !important;
    color: #94a3b8 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #4f46e5 !important; color: #a78bfa !important;
    background: #1e1b4b !important;
}

/* ── Expanders ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0f1729 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}
[data-testid="stExpander"]:hover {
    border-color: #4f46e5 !important;
}
[data-testid="stExpander"] details summary {
    font-weight: 500 !important; color: #94a3b8 !important;
    font-size: 14px !important; padding: 14px 16px !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-testid="stTab"] {
    background: transparent !important; color: #6b7280 !important;
    font-size: 13px; font-weight: 500;
    border-radius: 6px 6px 0 0; padding: 8px 16px;
    transition: color 0.15s;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #7c3aed !important;
}
[data-testid="stTabs"] { border-bottom: 1px solid #1e2736 !important; }

/* ── Inputs & Selects ────────────────────────────────────── */
.stTextInput input, .stSelectbox select, .stNumberInput input,
.stTextArea textarea {
    background: #0b1120 !important;
    border: 1px solid #1e2736 !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-size: 13px !important;
}
.stTextInput input:focus, .stSelectbox select:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px #6366f120 !important;
}
[data-testid="stSlider"] {
    padding: 0 4px;
}

/* ── Dataframes ──────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] [data-testid="data-grid-canvas"] {
    background: #0b1120 !important;
}

/* ── Alerts ──────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 13px !important;
}

/* ── Metrics ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0f1729; border: 1px solid #1e2d45;
    border-radius: 10px; padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; }

/* ── Section Headers ─────────────────────────────────────── */
.section-header {
    font-size: 15px; font-weight: 600;
    color: #94a3b8;
    padding: 4px 0 10px;
    border-bottom: 1px solid #1e2736;
    margin-bottom: 18px;
    display: flex; align-items: center; gap: 8px;
}

/* ── Info Cards ──────────────────────────────────────────── */
.info-card {
    background: #0c1628; border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 8px; padding: 14px 18px;
    font-size: 13px; color: #7dd3fc;
    margin: 8px 0;
}
.info-card strong { color: #93c5fd; }

/* ── Horizontal Rule ─────────────────────────────────────── */
hr { border: none; border-top: 1px solid #1e2736 !important; margin: 20px 0 !important; }

/* scrollbar global */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0b1120; }
::-webkit-scrollbar-thumb { background: #1e2736; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
</style>
""", unsafe_allow_html=True)

# ─── UI Helper Functions ────────────────────────────────────────────────────

def render_pipeline_visualization(framework_key: str, logs: list, status: str):
    """Render an interactive horizontal pipeline step visualization."""
    steps = _get_pipeline_steps(framework_key, tuple(logs), status)
    if not steps:
        return

    html_parts = ['<div class="pipeline-container">']
    for i, step in enumerate(steps):
        s = step["status"]  # done | active | pending | failed | cancelled
        icon_map = {"done": step["icon"], "active": step["icon"], "pending": step["icon"], "failed": "❌", "cancelled": "⛔"}
        icon = icon_map.get(s, step["icon"])

        if i > 0:
            connector_cls = "done" if steps[i-1]["status"] == "done" else ("active" if steps[i-1]["status"] == "active" else "")
            html_parts.append(f'<div class="pipeline-connector {connector_cls}"></div>')

        tooltip = step.get("description", "")
        html_parts.append(f'''
        <div class="pipeline-step" title="{tooltip}">
            <div class="pipeline-step-icon {s}">{icon}</div>
            <div class="pipeline-step-label {s}">{step["label"]}</div>
        </div>''')

    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_colored_logs(logs: list, max_lines: int = 80):
    """Render logs in a styled dark terminal panel with color-coded lines."""
    html = _build_log_html(tuple(logs), max_lines)
    st.markdown(html, unsafe_allow_html=True)


def render_stat_cards(running: int, completed: int, failed: int, cancelled: int):
    """Render animated status metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        (col1, running,   "🟢", "Running",   "#3fb950"),
        (col2, completed, "✅", "Completed", "#58a6ff"),
        (col3, failed,    "❌", "Failed",    "#f85149"),
        (col4, cancelled, "🚫", "Cancelled", "#d29922"),
    ]
    for col, val, icon, lbl, color in cards:
        with col:
            st.markdown(f'''
            <div class="stat-card">
                <div class="number" style="color:{color}">{val}</div>
                <div class="label">{icon} {lbl}</div>
            </div>''', unsafe_allow_html=True)


def fw_badge_html(framework: str) -> str:
    """Return colored framework badge HTML."""
    key = framework.lower().replace(" ", "").replace("automl", "")
    label_map = {
        "autogluon": ("AutoGluon", "fw-autogluon"),
        "flaml":     ("FLAML",     "fw-flaml"),
        "h2o":       ("H2O",       "fw-h2o"),
        "tpot":      ("TPOT",      "fw-tpot"),
        "pycaret":   ("PyCaret",   "fw-pycaret"),
        "lale":      ("Lale",      "fw-lale"),
    }
    label, cls = label_map.get(key, (framework, ""))
    return f'<span class="fw-badge {cls}">{label}</span>'


def status_badge_html(status: str) -> str:
    """Return colored status badge HTML."""
    labels = {
        "running":   "🟢 Running",
        "completed": "✅ Completed",
        "failed":    "❌ Failed",
        "cancelled": "🚫 Cancelled",
        "queued":    "⏳ Queued",
    }
    label = labels.get(status, status.capitalize())
    return f'<span class="badge badge-{status}">{label}</span>'


def render_metrics_pills(metrics: dict):
    """Render metric pills for key metrics."""
    if not metrics:
        return
    pill_html = '<div style="display:flex;flex-wrap:wrap;">'
    for k, v in metrics.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        pill_html += f'''
        <div class="metric-pill">
            <div><div class="m-label">{k}</div><div class="m-value">{val_str}</div></div>
        </div>'''
    pill_html += '</div>'
    st.markdown(pill_html, unsafe_allow_html=True)


# ─── End helpers ──────────────────────────────────────────────────────────────

# ── One-time startup: heal MLflow + set experiment (runs once per server session)
@st.cache_resource
def _startup_init():
    """Runs once when the server starts — keeps costly I/O out of the hot rerun path."""
    from src.mlflow_utils import heal_mlruns, safe_set_experiment
    try:
        heal_mlruns()
    except Exception:
        pass
    try:
        safe_set_experiment("Multi_AutoML_Project")
    except Exception:
        pass

_startup_init()

# ── Session state initialisation (single consolidated pass) ───────────────────
_SS_DEFAULTS: dict = {
    'df':           None,
    'predictor':    None,
    'model_type':   None,
    'valid_df':     None,
    'test_df':      None,
    'active_df':    None,
    'original_df':  None,
    'target':       None,
    'run_id':       None,
    'dvc_hashes':   {},
    'cv_folds':     0,
    'task_type':    'Classification',
    'framework':    'AutoGluon',
    'target_stats': {},
}
for _k, _v in _SS_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
if 'log_queue' not in st.session_state:
    st.session_state['log_queue'] = queue.Queue()

# Initialise the experiment manager singleton
exp_manager = get_or_create_manager(st.session_state)

# ── Sidebar brand ──────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-title">Multi-AutoML<br>Interface</div>
    <div class="sidebar-brand-sub">Intelligent MLOps Workspace</div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-nav-title">Navigation</div>', unsafe_allow_html=True)

_NAV_ITEMS = {
    "🏠  Overview": "Data Upload",
    "🗄️  Data": "Data Exploration",
    "⚙️  AutoML": "Training",
    "🧪  Experiments": "Experiments",
    "📦  Registry & Deploy": "Prediction",
    "📈  Monitoring": "History (MLflow)",
}

_default_nav_label = next((k for k, v in _NAV_ITEMS.items() if v == st.session_state.get('menu_page')), "🏠  Overview")
selected_nav_label = st.sidebar.radio(
    label="Main navigation",
    options=list(_NAV_ITEMS.keys()),
    index=list(_NAV_ITEMS.keys()).index(_default_nav_label),
    label_visibility="collapsed",
)

menu = _NAV_ITEMS[selected_nav_label]
st.session_state['menu_page'] = menu

st.sidebar.markdown('<div class="sidebar-sep">Integrations</div>', unsafe_allow_html=True)
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
    st.markdown("""
    <div class="page-title">
        <div class="page-title-icon">📂</div>
        <div class="page-title-text">
            <h1>Data Upload &amp; Lake</h1>
            <p>Upload datasets to the versioned Data Lake &mdash; available in Training and Prediction tabs.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    upload_tab, cv_upload_tab = st.tabs(["📄 Tabular Data (CSV/Excel)", "🖼️ Computer Vision Data (Images/ZIP)"])
    
    with upload_tab:
        upload_col, info_col = st.columns([2, 1])
        with upload_col:
            uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx", "xls"])
            no_header_upload = st.checkbox(
                "📋 This file has no header row (auto-generate col_0, col_1…)",
                value=False, key="upload_no_header",
                help="Check this if the first row of your file contains data, not column names."
            )
            filename_prefix = st.text_input("File prefix (name in Data Lake)", value="dataset", key="prefix_tab")
            upload_btn = st.button("💾 Process & Save Tabular Data", type="primary")

        with info_col:
            st.markdown("""
            <div class="preview-card">
                <h4>📖 About the Data Lake</h4>
                <p>Files are versioned using DVC and stored with a content hash. The same dataset at different times can be compared by hash. All frameworks read from this shared storage.</p>
            </div>""", unsafe_allow_html=True)

        if upload_btn and uploaded_file is not None:
            try:
                with st.spinner("Processing and versioning tabular data…"):
                    from src.data_utils import init_dvc, save_to_data_lake
                    init_dvc()
                    df = cached_load_data(uploaded_file, no_header=no_header_upload)
                    t_path, t_tag, t_hash = save_to_data_lake(df, filename_prefix)
                    st.cache_data.clear()

                st.success(f"✅ Saved to Data Lake! Hash: `{t_hash}`")
                st.session_state['_just_uploaded'] = df
            except Exception as e:
                st.error(f"Error processing tabular data: {e}")

    with cv_upload_tab:
        cv_col, cv_info_col = st.columns([2, 1])
        with cv_col:
            st.info("Upload multiple images (PNG/JPG) or a single ZIP archive containing your images.")
            uploaded_images = st.file_uploader("Upload Images or ZIP", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True)
            dataset_name = st.text_input("Computer Vision Dataset Name", value="image_dataset")
            cv_upload_btn = st.button("📸 Extract & Save Image Dataset", type="primary")
            
        with cv_info_col:
            st.markdown("""
            <div class="preview-card">
                <h4>🖼️ CV Datasets</h4>
                <p>Images are stored in a dedicated <code>data_lake/images/</code> structured directory. Frameworks like AutoGluon and AutoKeras will automatically traverse these dirs for training.</p>
            </div>""", unsafe_allow_html=True)

        if cv_upload_btn and uploaded_images:
            try:
                with st.spinner("Processing and transferring images to Data Lake…"):
                    from src.data_utils import process_image_upload
                    is_zip = len(uploaded_images) == 1 and uploaded_images[0].name.endswith('.zip')
                    cv_dir, full_hash, short_hash = process_image_upload(uploaded_images, dataset_name, is_zip)
                    st.cache_data.clear()
                st.success(f"✅ Image Dataset ready in Data Lake! Hash: `{short_hash}`")
            except Exception as e:
                st.error(f"Error processing images: {e}")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("2. Preview & Profiling")
    
    available_files = cached_get_data_lake_files()
    if not available_files and st.session_state.get('_just_uploaded') is None:
        st.info("Upload a file above to see its preview and profiling.")
    else:
        df = None
        if st.session_state.get('_just_uploaded') is not None:
            df = st.session_state['_just_uploaded']
            st.info("Previewing most recently uploaded dataset. Select another file from the dropdown to dismiss this.")
            prev_file = st.selectbox("Select file to preview", available_files, index=0 if available_files else None)
            if prev_file:
                try:
                    st.session_state.pop('_just_uploaded', None)
                    df = cached_load_data(prev_file)
                except Exception:
                    pass
        else:
            prev_file = st.selectbox("Select file to preview", available_files)
            if prev_file:
                try:
                    df = cached_load_data(prev_file)
                except Exception as e:
                    st.error(f"Error loading preview file: {e}")
                    
        if df is not None:
            try:
                # ── Quick EDA panels ─────────────────────────────────────
                st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
                summary = cached_get_data_summary(df)
                _missing_pct_mean, _memory_mb = _compute_overview_stats(df)

                ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)
                for col, label, val, color in [
                    (ov_col1, "Rows",     summary['rows'],    "#58a6ff"),
                    (ov_col2, "Columns",  summary['columns'], "#3fb950"),
                    (ov_col3, "Missing %", f"{_missing_pct_mean:.1f}%", "#d29922"),
                    (ov_col4, "Memory",   f"{_memory_mb:.1f} MB", "#bc8cff"),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="number" style="color:{color}">{val}</div>
                            <div class="label">{label}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                tab_preview, tab_missing, tab_types, tab_dist = st.tabs([
                    "🔍 Preview", "❓ Missing Values", "📐 Data Types", "📊 Distribution"
                ])

                with tab_preview:
                    st.dataframe(df.head(10), use_container_width=True)

                with tab_missing:
                    miss_series = _compute_missing_stats(df)
                    miss_df = miss_series[miss_series > 0]
                    if len(miss_df) == 0:
                        st.success("✅ No missing values found!")
                    else:
                        _fig_m = _make_missing_fig(miss_series)
                        if _fig_m is not None:
                            st.pyplot(_fig_m, use_container_width=True)
                        st.dataframe(pd.DataFrame({"Column": miss_df.index, "Missing %": miss_df.values.round(2)}), use_container_width=True)

                with tab_types:
                    type_counts = _compute_type_counts(df)
                    _fig_t = _make_type_pie(type_counts)
                    st.pyplot(_fig_t, use_container_width=True)
                    summary_df = _compute_column_summary(df)
                    st.dataframe(summary_df, use_container_width=True)

                with tab_dist:
                    num_cols_list = df.select_dtypes(include="number").columns.tolist()
                    if num_cols_list:
                        dist_col = st.selectbox("Select column for distribution", num_cols_list, key="dist_col_sel")
                        _fig_d = _make_dist_fig(df[dist_col], dist_col)
                        st.pyplot(_fig_d, use_container_width=True)
                        st.dataframe(df[[dist_col]].describe().T, use_container_width=True)
                    else:
                        st.info("No numeric columns found for distribution plot.")
            except Exception as e:
                st.error(f"Error loading UI previews: {e}")


elif menu == "Data Exploration":
    st.markdown("""
    <div class="page-title">
        <div class="page-title-icon">🔍</div>
        <div class="page-title-text">
            <h2>Data Exploration & Auto-EDA</h2>
            <p>Automatically profile your datasets to find correlations, missing values, and imbalances before training.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Select a dataset from the Data Lake to generate a comprehensive Exploratory Data Analysis (EDA) report.")
    
    available_files = cached_get_data_lake_files()
    if not available_files:
        st.warning("No files in Data Lake. Please upload data first in the 'Data Upload' tab.")
    else:
        file_options = [os.path.basename(f) for f in available_files]
        file_paths_map = {os.path.basename(f): f for f in available_files}
        
        selected_file = st.selectbox("Select Dataset to Profile", file_options)
        
        if st.button("Generate Auto-EDA Report", type="primary"):
            try:
                import ydata_profiling
                from streamlit_pandas_profiling import st_profile_report
                
                with st.spinner("Generating EDA Report... This may take a moment for large datasets."):
                    file_path = file_paths_map[selected_file]
                    df_eda = cached_load_data(file_path)
                    
                    # Basic Health Checks built-in warnings
                    n_rows = len(df_eda)
                    missing_cells = df_eda.isnull().sum().sum()
                    missing_pct = (missing_cells / (df_eda.shape[0] * df_eda.shape[1])) * 100
                    
                    if missing_pct > 5:
                        st.warning(f"⚠️ Health Alert: Your dataset has {missing_pct:.1f}% missing values overall. Consider imputation before training.")
                    
                    # Generate and display report
                    pr = ydata_profiling.ProfileReport(df_eda, explorative=True, minimal=n_rows > 10000)
                    st_profile_report(pr)
                    
            except ImportError:
                st.error("Missing dependency. Please ensure `ydata-profiling` and `streamlit-pandas-profiling` are installed in your environment.")
            except Exception as e:
                st.error(f"Error generating report: {e}")

elif menu == "Training":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Model Training</h1>
        <p>Configure and launch an AutoML experiment. Training runs in the background — you can start multiple at once.</p>
    </div>""", unsafe_allow_html=True)
    
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

    with st.expander("🔧 Dataset Loading Options (no header row)", expanded=False):
        nh_col1, nh_col2, nh_col3 = st.columns(3)
        with nh_col1:
            train_no_header = st.checkbox("Train CSV has no header", value=False, key="train_no_header",
                help="Auto-generate col_0, col_1… if the training file has no column names.")
        with nh_col2:
            valid_no_header = st.checkbox("Validation CSV has no header", value=False, key="valid_no_header",
                help="Auto-generate col_0, col_1… if the validation file has no column names.")
        with nh_col3:
            test_no_header = st.checkbox("Test CSV has no header", value=False, key="test_no_header",
                help="Auto-generate col_0, col_1… if the test file has no column names.")

    if train_file_selection:
        try:
            from src.data_utils import get_dvc_hash
            # Load Train
            train_path = file_paths_map[train_file_selection]
            df = cached_load_data(train_path, no_header=train_no_header)
            
            # Fetch Hash
            t_hash_full, t_hash_short = get_dvc_hash(train_path)
            dvc_hashes = {"dvc_train_hash": t_hash_short}
            
            # Load Valid
            valid_df = None
            if valid_file_selection != "None":
                valid_path = file_paths_map[valid_file_selection]
                valid_df = cached_load_data(valid_path, no_header=valid_no_header)
                v_hash_full, v_hash_short = get_dvc_hash(valid_path)
                dvc_hashes["dvc_valid_hash"] = v_hash_short
                
            # Load Test
            test_df = None
            if test_file_selection != "None":
                test_path = file_paths_map[test_file_selection]
                test_df = cached_load_data(test_path, no_header=test_no_header)
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
    if st.session_state.get('df') is not None:
        df = st.session_state.get('df')
        valid_df_session = st.session_state.get('valid_df', None)
        test_df_session = st.session_state.get('test_df', None)
        
        split_strategy = st.radio(
            "Data Split Strategy", 
            ["Random", "Manual", "Chronological"], 
            horizontal=True, 
            help="Choose how the data will be separated for model evaluation."
        )

        val_size_pct = 0
        test_size_pct = 0
        cv_folds = st.session_state.get('cv_folds', 5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Test Set (Final Evaluation)**")
            if test_df_session is not None:
                st.success("Test-set provided through a dedicated Data Lake file.")
            else:
                if split_strategy == "Random":
                    test_size_pct = st.slider("Test Percentage (%)", 0, 50, 10, 5)
                elif split_strategy == "Chronological":
                    test_size_pct = st.slider("Latest data for Test (%)", 0, 50, 10, 5)

        with col2:
            st.markdown("**Validation Strategy**")
            if valid_df_session is not None:
                st.success("Validation-set provided via file in Data Lake.")
            else:
                if split_strategy == "Random":
                    val_method = st.radio("Method", ["Simple Holdout", "Cross-Validation"], horizontal=True)
                    if val_method == "Simple Holdout":
                        val_size_pct = st.slider("Validation Percentage (%)", 0, 50, 10, 5)
                    else:
                        cv_folds = st.slider("Number of Folds (K)", 2, 10, 5)
                elif split_strategy == "Chronological":
                    val_size_pct = st.slider("Preceding data for Validation (%)", 0, 50, 10, 5)
        
        manual_split_col = None
        chrono_col = None
        if split_strategy == "Manual":
            manual_split_col = st.selectbox("Select Split Column (must contain 'train', 'val', 'test')", df.columns)
        elif split_strategy == "Chronological":
            chrono_col = st.selectbox("Select Time/Date Column to sort by", df.columns)
                
        # Apply splits safely on a pristine base DataFrame.
        original_df = st.session_state.get('original_df')
        needs_reset_original = (
            original_df is None
            or (('has_split' not in st.session_state) and len(original_df) != len(df))
        )
        if needs_reset_original:
            original_df = df.copy()
            st.session_state['original_df'] = original_df

        base_df = original_df.copy()
        
        if split_strategy == "Manual" and manual_split_col:
            val_mask = base_df[manual_split_col].astype(str).str.lower().str.contains("val|valid")
            test_mask = base_df[manual_split_col].astype(str).str.lower().str.contains("test")
            train_mask = ~(val_mask | test_mask)
            
            valid_df_session = base_df[val_mask].copy() if val_mask.sum() > 0 else None
            test_df_session = base_df[test_mask].copy() if test_mask.sum() > 0 else None
            base_df = base_df[train_mask].copy()
            st.session_state['valid_df'] = valid_df_session
            st.session_state['test_df'] = test_df_session
            
        elif split_strategy == "Chronological" and chrono_col:
            base_df = base_df.sort_values(by=chrono_col).reset_index(drop=True)
            total_len = len(base_df)
            test_idx = int(total_len * (1 - test_size_pct/100.0))
            val_idx = int(total_len * (1 - (test_size_pct + val_size_pct)/100.0))
            
            if test_size_pct > 0:
                test_df_session = base_df.iloc[test_idx:].copy()
                st.session_state['test_df'] = test_df_session
            if val_size_pct > 0:
                valid_df_session = base_df.iloc[val_idx:test_idx].copy()
                st.session_state['valid_df'] = valid_df_session
            base_df = base_df.iloc[:val_idx].copy()
            
        elif split_strategy == "Random":
            from sklearn.model_selection import train_test_split
            if test_size_pct > 0:
                base_df, fresh_test_df = train_test_split(base_df, test_size=(test_size_pct/100.0), random_state=42)
                test_df_session = fresh_test_df
                st.session_state['test_df'] = test_df_session
                
            if val_size_pct > 0:
                if len(base_df) > 100:
                    # Adjust proportion relative to remaining data
                    adj_val_pct = val_size_pct / (100 - test_size_pct)
                    base_df, fresh_val_df = train_test_split(base_df, test_size=adj_val_pct, random_state=42)
                    valid_df_session = fresh_val_df
                    st.session_state['valid_df'] = valid_df_session
                
        # Update current working df
        df = base_df
        st.session_state['active_df'] = df
        st.session_state['cv_folds'] = cv_folds

    st.markdown("---")
    st.subheader("3. AutoML Configuration")
    
    if st.session_state.get('df') is not None:
        df = st.session_state.get('df')
        valid_df = st.session_state.get('valid_df', None)
        test_df = st.session_state.get('test_df', None)
        
        columns = df.columns.tolist()
        
        # Task Type Filtering
        task_type = st.selectbox("Task Type", [
            "Classification", "Regression", "Computer Vision - Multi-Label Classification", "Time Series Forecasting", "Ranking",
            "Computer Vision - Image Classification", "Computer Vision - Object Detection", "Computer Vision - Image Segmentation"
        ])
        st.session_state['task_type'] = task_type
        
        task_fw_map = {
            "Classification": ["AutoGluon", "FLAML", "H2O AutoML", "TPOT", "PyCaret", "Lale"],
            "Regression": ["AutoGluon", "FLAML", "H2O AutoML", "TPOT", "PyCaret", "Lale"],
            "Computer Vision - Multi-Label Classification": ["AutoGluon", "AutoKeras"],
            "Time Series Forecasting": ["AutoGluon", "FLAML", "PyCaret"],
            "Ranking": ["FLAML"],
            "Computer Vision - Image Classification": ["AutoGluon", "AutoKeras"],
            "Computer Vision - Object Detection": ["AutoGluon"],
            "Computer Vision - Image Segmentation": ["AutoGluon"]
        }
        available_frameworks = task_fw_map.get(task_type, ["FLAML"])
        framework = st.selectbox("Select AutoML Framework", available_frameworks)
        st.session_state['framework'] = framework
        
        if task_type.startswith("Computer Vision"):
            target = "label"
            st.info("Target column is automatically set to 'label' for Image tasks (inferred from directory structure).")
        else:
            target = st.selectbox("Select Target Column", columns, index=columns.index(st.session_state.get('target', columns[0])) if st.session_state.get('target') in columns else 0)
        st.session_state['target'] = target
        run_name = st.text_input("Run Name", value=f"{framework.lower()}_run_{int(time.time())}")

        # Datasets info card
        r_cnt = len(df)
        v_cnt = len(valid_df) if valid_df is not None else 0
        t_cnt = len(test_df) if test_df is not None else 0
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;display:flex;gap:24px;margin:8px 0;">
            <span><span style="color:#8b949e;font-size:11px">TRAIN</span><br><span style="color:#58a6ff;font-weight:700">{r_cnt:,} rows</span></span>
            <span><span style="color:#8b949e;font-size:11px">VALID</span><br><span style="color:#3fb950;font-weight:700">{'None' if v_cnt==0 else f'{v_cnt:,} rows'}</span></span>
            <span><span style="color:#8b949e;font-size:11px">TEST</span><br><span style="color:#d29922;font-weight:700">{'None' if t_cnt==0 else f'{t_cnt:,} rows'}</span></span>
            <span><span style="color:#8b949e;font-size:11px">TARGET</span><br><span style="color:#bc8cff;font-weight:700">{target}</span></span>
        </div>""", unsafe_allow_html=True)

        # ── Framework "What happens" preview ─────────────────────────────
        fw_previews = {
            "AutoGluon": {
                "color": "#58a6ff", "icon": "🤖",
                "steps": [
                    ("📊 Data Prep", "Validates columns, encodes categoricals, handles nulls."),
                    ("🏋️ Model Fit", "Trains LightGBM, XGBoost, CatBoost, RF, KNN in parallel."),
                    ("🏗️ Ensembling", "Stacks top models with weighted ensembling."),
                    ("📏 Evaluation", "Scores all models on validation set — builds leaderboard."),
                    ("📝 MLflow Log", "Saves model, metrics, params, and artifacts to MLflow."),
                ]
            },
            "FLAML": {
                "color": "#79c0ff", "icon": "🔍",
                "steps": [
                    ("📊 Data Prep", "Feature-type inference, optional label encoding."),
                    ("🔍 HP Search", "Cost-effective Bayesian search over estimators & hyperparams."),
                    ("🏆 Selection", "Picks best estimator + configuration from search results."),
                    ("💾 Saving", "Serializes model to disk using pickle."),
                    ("📝 MLflow Log", "Saves model, metrics, params, and artifacts to MLflow."),
                ]
            },
            "H2O AutoML": {
                "color": "#3fb950", "icon": "🌊",
                "steps": [
                    ("🌊 Cluster Init", "Starts local H2O Java cluster with allocated memory."),
                    ("📊 Data Prep", "Converts DataFrames to H2OFrames, applies type casting."),
                    ("🤖 AutoML Fit", "Trains GBM, XGBoost, DRF, GLM, DeepLearning variants."),
                    ("📏 Leaderboard", "Ranks all models; evaluates leader on validation set."),
                    ("📝 MLflow Log", "Saves model, metrics, params, and artifacts to MLflow."),
                ]
            },
            "TPOT": {
                "color": "#bc8cff", "icon": "🧬",
                "steps": [
                    ("📊 Data Prep", "TF-IDF for text, ordinal encoding, standard scaling."),
                    ("🧬 GA Search", "Genetic Algorithm evolves scikit-learn pipeline configs."),
                    ("🏆 Selection", "Selects highest-scoring pipeline from all generations."),
                    ("📤 Export", "Exports best pipeline as .py file with classification report."),
                    ("📝 MLflow Log", "Saves model, metrics, params, and artifacts to MLflow."),
                ]
            },
            "PyCaret": {
                "color": "#fbcfe8", "icon": "⚙️",
                "steps": [
                    ("⚙️ Setup", "Data normalization, splits, implicit encoding."),
                    ("⚖️ Compare", "Trains multiple baseline models to find the top candidates."),
                    ("🔧 Tuning", "Optimizes hyperparameters of the best model."),
                    ("🌪️ Blending", "Creates an ensemble of the best found models."),
                    ("📝 MLflow Log", "Saves model, metrics, params, and artifacts to MLflow."),
                ]
            },
            "Lale": {
                "color": "#bae6fd", "icon": "🌳",
                "steps": [
                    ("⚙️ Planned Pipe", "Maps PCA/Scaler to SKLearn classifiers."),
                    ("🔧 Hyperopt", "Executes intelligent bayesian HP tuning with Hyperopt."),
                    ("🕒 Fit Opt.", "Trains models matching config iteratively."),
                    ("🏆 Extract Model", "Selects best tuned scikit-learn pipeline compatible object."),
                    ("📝 MLflow Log", "Saves model, metrics, params, and artifacts to MLflow."),
                ]
            },
        }
        if framework in fw_previews:
            prev = fw_previews[framework]
            with st.expander(f"🗺️ What happens during {framework} training?", expanded=False):
                cols_prev = st.columns(len(prev["steps"]))
                for i, (step_name, step_desc) in enumerate(prev["steps"]):
                    with cols_prev[i]:
                        st.markdown(f"""
                        <div class="preview-card" style="border-color:{prev['color']}30;min-height:120px;">
                            <h4 style="color:{prev['color']}">{step_name}</h4>
                            <p>{step_desc}</p>
                        </div>""", unsafe_allow_html=True)

        # Framework specific options
        st.markdown('<div class="section-header">⚙️ Framework Configuration</div>', unsafe_allow_html=True)
        
        # Common framework options
        seed = st.number_input("Seed (reproducibility)", value=42, min_value=0, max_value=9999)

        # ── Global Parallelism ─────────────────────────────────────────────────
        import os as _os
        _cpu_count = _os.cpu_count() or 4
        with st.expander("⚡ Parallelism (n_jobs)", expanded=False):
            parallelism_mode = st.radio(
                "Parallelism Mode",
                ["Auto (all cores)", "Manual"],
                horizontal=True,
                help="Controls how many CPU cores are used by supported frameworks (FLAML, TPOT, PyCaret).",
                key="parallelism_mode"
            )
            if parallelism_mode == "Manual":
                global_n_jobs = st.slider(
                    "Number of parallel jobs",
                    min_value=1, max_value=_cpu_count,
                    value=min(2, _cpu_count),
                    help=f"Your machine has {_cpu_count} logical cores. Higher values speed up training but use more memory.",
                    key="global_n_jobs"
                )
            else:
                global_n_jobs = -1
                st.info(f"Auto mode: all {_cpu_count} logical cores will be used (n_jobs = -1).")
        # ──────────────────────────────────────────────────────────────────────

        # Init vars
        time_limit = time_budget = max_runtime_secs = 60
        presets = task = metric = estimator_list = None
        nfolds = balance_classes = sort_metric = exclude_algos = None
        
        if framework == "AutoGluon":
            use_time_limit = st.checkbox("Enable Time Limit", value=True, help="If disabled, AutoGluon will train until all models are fully evaluated without time restrictions.")
            if use_time_limit:
                time_limit = st.slider("Time limit (seconds)", 30, 3600, 60)
            else:
                time_limit = None
            presets = st.selectbox("Presets", ['medium_quality', 'best_quality', 'high_quality', 'good_quality', 'optimize_for_deployment'])
        elif framework == "FLAML":
            use_time_limit = st.checkbox("Enable Time Limit", value=True, help="If disabled, FLAML will train until convergence or all configurations are exhausted.")
            if use_time_limit:
                time_budget = st.slider("Time budget (seconds)", 30, 3600, 60)
            else:
                time_budget = None
                
            # Map global task_type to FLAML task
            if task_type == 'Classification':
                task = 'classification'
            elif task_type == 'Regression':
                task = 'regression'
            elif task_type == 'Time Series Forecasting':
                task = 'ts_forecast'
            elif task_type == 'Ranking':
                task = 'rank'
            else:
                task = 'classification'
            
            st.info(f"FLAML internal task synced to: **{task}**")
            
            # Smart metric selection for FLAML
            target_stats = st.session_state.get('target_stats', {})
            if target_stats.get('name') != target:
                num_classes = _get_column_nunique(df, target)
                target_stats = {'name': target, 'nunique': num_classes}
                st.session_state['target_stats'] = target_stats
            else:
                num_classes = target_stats['nunique']

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
            
            use_time_limit = st.checkbox("Enable Time Limit", value=True, help="If disabled, H2O will train until the max number of models is reached.")
            if use_time_limit:
                max_runtime_secs = st.slider("Max runtime (seconds)", 60, 3600, 300)
            else:
                max_runtime_secs = 0
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
                
            use_time_limit = st.checkbox("Enable Time Limit", value=True, help="If disabled, TPOT will run for the exact number of generations requested.")
            if use_time_limit:
                max_time_mins = st.slider("Max time (minutes)", 5, 120, 30, help="Maximum training time in minutes")
            else:
                max_time_mins = None
            max_eval_time_mins = st.slider("Max time per evaluation (minutes)", 1, 20, 5, help="Maximum time per pipeline evaluation")
            verbosity = st.slider("Log verbosity level", 0, 3, 2, help="TPOT feedback verbosity")
            
            # Advanced TPOT Options
            with st.expander("⚙️ Advanced TPOT Options"):
                config_dict = st.selectbox("TPOT Configuration", [
                    'TPOT light', 'TPOT MDR', 'TPOT sparse', 'TPOT NN'
                ], help="Predefined TPOT configuration for different types of problems")
                
                tfidf_max_features = st.number_input("Text features max dimensions (TF-IDF)", min_value=100, max_value=10000, value=500, step=100)
                ngram_max = st.slider("Max text N-Gram size", 1, 3, 2, help="If 2, evaluates unigrams and bigrams. If 3, unigrams, bigrams, and trigrams.")
                tfidf_ngram_range = (1, ngram_max)
                
                # Auto problem detection
                target_stats = st.session_state.get('target_stats', {})
                if target_stats.get('name') != target:
                    num_classes = _get_column_nunique(df, target)
                    st.session_state['target_stats'] = {'name': target, 'nunique': num_classes}
                else:
                    num_classes = target_stats['nunique']
                
                problem_type = 'classification' if num_classes <= 20 or df[target].dtype == 'object' else 'regression'
                st.info(f"🎯 Problem type detected: **{problem_type}**")
                
                # Metrics based on problem type
                if problem_type == 'classification':
                    scoring_options = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'precision_macro', 'recall_macro']
                else:
                    scoring_options = ['neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2', 'explained_variance']
                
                scoring = st.selectbox("Optimization Metric", scoring_options, help="Metric used to optimize the pipelines")
        elif framework == "PyCaret":
            st.info("⚙️ PyCaret automates complex end-to-end pipelines.")
            use_time_limit = st.checkbox("Enable Tuning Iterator Limit", value=True, help="Limits tuning iterations based on a pseudo-time limiter.")
            if use_time_limit:
                 time_limit = st.slider("Time limit equivalent (seconds) - impacts n_iter", 60, 1200, 300)
            else:
                 time_limit = None
                 
            fh = 1
            seasonal_period = 1
            if task_type == "Time Series Forecasting":
                 st.markdown("#### 📈 Time Series Configuration")
                 fh = st.number_input("Forecasting Horizon (fh)", min_value=1, value=12, help="Number of steps into the future to predict")
                 seasonal_period = st.number_input("Seasonal Period", min_value=1, value=12, help="Seasonal frequency (e.g., 12 for monthly data, 7 for daily)")
                 st.session_state['pycaret_fh'] = fh
                 st.session_state['pycaret_sp'] = seasonal_period
        elif framework == "Lale":
            st.info("🌳 Lale extends scikit-learn with Hyperopt topology optimizations.")
            use_time_limit = st.checkbox("Enable Tune Limit", value=True, help="Max evals limitation during optimization")
            if use_time_limit:
                 time_limit = st.slider("Max internal search equivalent (seconds)", 60, 600, 120)
            else:
                 time_limit = None

        st.markdown("---")
        st.subheader("4. Launch Experiment")

        if st.button("🚀 Start Training", type="primary"):
            import time as _t
            _key = f"{framework.lower()}_{int(_t.time())}"

            # Build kwargs dict for the trainer
            if framework == "AutoGluon":
                from src.autogluon_utils import train_model as train_autogluon
                _fn = train_autogluon
                _kwargs = dict(train_data=df, target=target, run_name=run_name,
                               valid_data=valid_df, test_data=test_df,
                               time_limit=time_limit, presets=presets, seed=seed, cv_folds=cv_folds, task_type=task_type)
                _fw_key = "autogluon"
            elif framework == "AutoKeras":
                from src.autokeras_utils import run_autokeras_experiment
                _fn = run_autokeras_experiment
                _kwargs = dict(train_data=df, target=target, run_name=run_name,
                               valid_data=valid_df, task_type=task_type, time_limit=time_limit)
                _fw_key = "autokeras"
            elif framework == "FLAML":
                from src.flaml_utils import train_flaml_model
                _fn = train_flaml_model
                _kwargs = dict(train_data=df, target=target, run_name=run_name,
                               valid_data=valid_df, test_data=test_df,
                               time_budget=time_budget, task=task, metric=metric,
                               estimator_list=estimator_list, seed=seed, cv_folds=cv_folds,
                               n_jobs=global_n_jobs)
                _fw_key = "flaml"
            elif framework == "H2O AutoML":
                from src.h2o_utils import train_h2o_model
                _fn = train_h2o_model
                _kwargs = dict(train_data=df, target=target, run_name=run_name,
                               valid_data=valid_df, test_data=test_df,
                               max_runtime_secs=max_runtime_secs, max_models=max_models,
                               nfolds=nfolds, balance_classes=balance_classes,
                               seed=seed, sort_metric=sort_metric, exclude_algos=exclude_algos)
                _fw_key = "h2o"
            elif framework == "PyCaret":
                from src.pycaret_utils import run_pycaret_experiment
                _fn = run_pycaret_experiment
                
                # Fetch TS params if applicable
                _fh = st.session_state.get('pycaret_fh', 12) if task_type == 'Time Series Forecasting' else None
                _sp = st.session_state.get('pycaret_sp', 12) if task_type == 'Time Series Forecasting' else None
                
                _kwargs = dict(train_df=df, target_col=target, run_name=run_name,
                               val_df=valid_df, time_limit=time_limit,
                               task_type=task_type, fh=_fh, seasonal_period=_sp,
                               n_jobs=global_n_jobs,
                               log_queue=None)  # patched below after _entry creation
                _fw_key = "pycaret"
            elif framework == "Lale":
                from src.lale_utils import run_lale_experiment
                _fn = run_lale_experiment
                _kwargs = dict(train_df=df, target_col=target, run_name=run_name,
                               val_df=valid_df, time_limit=time_limit, task_type=task_type,
                               log_queue=None)  # patched below after _entry creation
                _fw_key = "lale"
            else:  # TPOT
                from src.tpot_utils import train_tpot_model
                _fn = train_tpot_model
                _kwargs = dict(df=df, target_column=target, run_name=run_name,
                               valid_data=valid_df, test_data=test_df,
                               generations=generations, population_size=population_size,
                               cv=cv, scoring=scoring, max_time_mins=max_time_mins,
                               max_eval_time_mins=max_eval_time_mins,
                               random_state=seed, verbosity=verbosity, n_jobs=global_n_jobs,
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
            # Patch log_queue for frameworks that need it (assigned after _entry is created)
            if "log_queue" in _kwargs and _kwargs["log_queue"] is None:
                _kwargs["log_queue"] = _entry.log_queue
            exp_manager.add(_entry)
            _t_obj.start()

            st.success(f"🚀 Experiment **{run_name}** queued! Navigate to the **Experiments** tab to monitor progress.")
            st.info("You can start another training right away or switch tabs — training runs in the background.")
    else:
        st.warning("Please upload or select Data Lake training sets first.")

elif menu == "Experiments":
    st.markdown("""
    <div class="main-header">
        <h1>🧪 Experiments Dashboard</h1>
        <p>Monitor and manage your concurrent AutoML training runs in real time.</p>
    </div>""", unsafe_allow_html=True)

    @_compat_fragment(run_every="5s")
    def render_experiment_dashboard():
        # Only refresh if there are active runs to save resources
        is_active = exp_manager.has_running()
        if is_active:
            exp_manager.refresh_all()
        
        all_exps = exp_manager.get_all()

        if not all_exps:
            st.markdown("""
            <div style="text-align:center;padding:60px;color:#8b949e;">
                <div style="font-size:48px">🚀</div>
                <div style="font-size:18px;font-weight:600;color:#f0f6fc;margin:12px 0;">No experiments yet</div>
                <div>Go to the <strong>Training</strong> tab to launch your first AutoML run.</div>
            </div>""", unsafe_allow_html=True)
            return

        # Summary stat cards
        n_running   = sum(1 for e in all_exps if e.status == "running")
        n_completed = sum(1 for e in all_exps if e.status == "completed")
        n_failed    = sum(1 for e in all_exps if e.status == "failed")
        n_cancelled = sum(1 for e in all_exps if e.status == "cancelled")
        render_stat_cards(n_running, n_completed, n_failed, n_cancelled)

        # Maintenance Section (collapsed by default)
        with st.expander("🛠️ Maintenance & Storage Management", expanded=False):
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                if st.button("🧹 Clear Local Models", use_container_width=True,
                             help="Deletes all folders inside /models. Safe if runs were synced to MLflow."):
                    try:
                        import shutil
                        if os.path.exists("models"):
                            shutil.rmtree("models"); os.makedirs("models")
                            st.success("Local models cleared!")
                        else:
                            st.info("Models folder is already empty.")
                    except Exception as me:
                        st.error(f"Cleanup error: {me}")
            with m_col2:
                if st.button("🔥 Reset MLflow (mlruns)", use_container_width=True,
                             help="DANGER: Deletes the local mlruns folder. All local experiment history will be lost."):
                    try:
                        import shutil
                        if os.path.exists("mlruns"):
                            shutil.rmtree("mlruns"); st.success("Local MLflow history reset!")
                        else:
                            st.info("mlruns folder not found.")
                    except Exception as reset_err:
                        st.error(f"Reset error: {reset_err}")
            with m_col3:
                try:
                    total, used, free = _get_disk_usage()
                    free_gb = free // (2**30)
                    used_gb = used // (2**30)
                    pct = int((used / total) * 100)
                    color = "#f85149" if free_gb < 2 else ("#d29922" if free_gb < 10 else "#3fb950")
                    st.markdown(f"""
                    <div style="padding:8px 0;">
                        <div style="font-size:12px;color:#8b949e;margin-bottom:4px;">DISK SPACE</div>
                        <div style="font-size:20px;font-weight:700;color:{color};">{free_gb} GB free</div>
                        <div style="background:#30363d;border-radius:4px;height:6px;margin-top:4px;">
                            <div style="background:{color};width:{pct}%;height:6px;border-radius:4px;"></div>
                        </div>
                        <div style="font-size:11px;color:#8b949e;margin-top:2px;">{used_gb} GB used of {total//(2**30)} GB</div>
                    </div>""", unsafe_allow_html=True)
                except Exception:
                    pass

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        for entry in all_exps:
            fw      = entry.metadata.get("framework", "Unknown")
            fw_key  = entry.metadata.get("framework_key", "unknown")
            rname   = entry.metadata.get("run_name", entry.key)
            elapsed = entry.elapsed_str()
            run_id  = entry.result.get("run_id") if entry.result else None
            is_active = entry.status == "running" and time.time() - getattr(entry, "last_update", 0) < 5

            expander_label = f"{entry.status_icon()} {rname}  ·  {fw}  ·  ⏱ {elapsed}"
            if is_active:
                expander_label += "  ·  💓"

            with st.expander(expander_label, expanded=(entry.status == "running")):

                # ── Card header row ────────────────────────────────────────
                h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns([3, 1, 1, 1, 1])
                with h_col1:
                    st.markdown(
                        fw_badge_html(fw) + " &nbsp; " + status_badge_html(entry.status) +
                        (f' &nbsp; <span class="exp-timer">⏱ {elapsed}</span>' if entry.status == "running" else ""),
                        unsafe_allow_html=True)
                    if run_id:
                        st.caption(f"Run ID: {run_id}")
                    else:
                        st.caption(f"Key: {entry.key}")
                with h_col2:
                    if entry.status == "running":
                        if st.button("⛔ Cancel", key=f"cancel_{entry.key}", use_container_width=True):
                            exp_manager.cancel(entry.key); st.rerun()
                with h_col3:
                    if entry.status in ("completed", "cancelled", "failed"):
                        if st.button("🗑️ Delete", key=f"delete_{entry.key}", use_container_width=True):
                            exp_manager.delete(entry.key); st.rerun()
                with h_col4:
                    if entry.status == "completed" and entry.result and entry.result.get("predictor"):
                        if st.button("🔮 Predict", key=f"load_{entry.key}", use_container_width=True):
                            st.session_state["predictor"]  = entry.result["predictor"]
                            st.session_state["model_type"] = entry.result.get("type", "unknown")
                            st.session_state["run_id"]     = entry.result.get("run_id")
                            st.success("Model loaded! Switch to the Prediction tab.")
                with h_col5:
                    if entry.status == "completed" and run_id:
                        try:
                            if st.button("📋 Register", key=f"reg_{entry.key}", use_container_width=True):
                                mlflow.register_model(f"runs:/{run_id}/model", rname)
                                st.success("Model registered!")
                        except Exception:
                            pass

                # ── Additional Actions: ONNX & Hugging Face ─────────────
                act_col1, act_col2, act_col3 = st.columns([1, 1, 1])
                with act_col1:
                    if entry.status == "completed" and run_id:
                        if st.button("📦 Export to ONNX", key=f"onnx_exp_{entry.key}", use_container_width=True):
                            try:
                                from src.onnx_utils import export_to_onnx
                                from src.data_utils import load_data
                                predictor = entry.result.get("predictor")
                                fw_key = entry.metadata.get("framework_key", "unknown")
                                target = entry.metadata.get("target", "target")
                                
                                # We need a sample of the data to infer shapes
                                available_files = cached_get_data_lake_files()
                                if available_files:
                                    sample_df = load_data(available_files[0])
                                    onnx_path = os.path.join("models", f"{rname}.onnx")
                                    export_to_onnx(predictor, fw_key, target, onnx_path, input_sample=sample_df[:1])
                                    mlflow.log_artifact(onnx_path, artifact_path="model")
                                    st.success(f"Model exported to ONNX and logged to MLflow!")
                                else:
                                    st.error("Need a dataset in Data Lake for shape inference.")
                            except Exception as oe:
                                st.error(f"ONNX Export error: {oe}")

                with act_col2:
                    if entry.status == "completed" and run_id:
                        if st.button("🚀 Push to Hugging Face", key=f"hf_push_{entry.key}", use_container_width=True):
                            st.session_state[f"show_hf_push_{entry.key}"] = True
                
                if st.session_state.get(f"show_hf_push_{entry.key}"):
                    with st.container():
                        st.markdown("##### 🚀 Push to Hugging Face Hub")
                        hf_repo = st.text_input("Repository ID (e.g., username/model-name)", key=f"hf_repo_{entry.key}")
                        hf_token = st.text_input("HF Access Token", type="password", key=f"hf_token_{entry.key}")
                        if st.button("Confirm Push", key=f"hf_confirm_{entry.key}"):
                            try:
                                from src.huggingface_utils import HuggingFaceService
                                hf = HuggingFaceService(token=hf_token)
                                # For simplicity, push the native model .pkl
                                model_path = os.path.join("models", f"{fw_key}_{rname}.pkl")
                                if not os.path.exists(model_path):
                                    # Try to download from mlflow if local not found
                                    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
                                    model_path = local_path
                                hf.upload_model(model_path, hf_repo)
                                st.success(f"Model successfully pushed to https://huggingface.co/{hf_repo}")
                                st.session_state[f"show_hf_push_{entry.key}"] = False
                            except Exception as hfe:
                                st.error(f"HF Upload error: {hfe}")

                # ── Pipeline visualization ────────────────────────────────
                st.markdown('<div class="section-header">🔄 Training Pipeline</div>', unsafe_allow_html=True)
                render_pipeline_visualization(fw_key, entry.all_logs, entry.status)

                # ── Tabs ──────────────────────────────────────────────────
                tab_logs, tab_metrics, tab_inspector, tab_mlflow, tab_code = st.tabs([
                    "📋 Logs", "📈 Metrics", "🔬 Pipeline Inspector", "🔍 MLflow", "💻 Code & Deploy"
                ])

                with tab_logs:
                    if entry.all_logs:
                        render_colored_logs(entry.all_logs, max_lines=100)
                    else:
                        st.markdown('<div class="log-panel"><span class="log-line-normal">(Waiting for logs…)</span></div>', unsafe_allow_html=True)

                with tab_metrics:
                    if entry.status == "completed" and run_id:
                        try:
                            run_data = _get_mlflow_run(run_id)
                            if run_data and run_data.data.metrics:
                                metrics = run_data.data.metrics
                                render_metrics_pills(metrics)
                                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                                _fig_metrics = _make_metrics_bar(tuple(metrics.items()))
                                st.pyplot(_fig_metrics, use_container_width=True)
                            else:
                                st.info("No metrics logged to MLflow yet.")
                        except Exception as me:
                            st.warning(f"Could not load metrics: {me}")
                    elif entry.status == "running":
                        st.info("⏳ Training in progress — metrics will appear here after completion.")
                    else:
                        st.info("No metrics available.")

                with tab_inspector:
                    st.markdown('<div class="section-header">🔬 Best Pipeline Inspector</div>', unsafe_allow_html=True)
                    fw_type   = entry.result.get("type", "") if entry.result else ""
                    predictor = entry.result.get("predictor") if entry.result else None

                    if fw_type == "autogluon" and predictor:
                        try:
                            lb = predictor.leaderboard(silent=True)
                            st.markdown("**🏆 Model Leaderboard**")
                            st.dataframe(lb, use_container_width=True)
                            # Bar chart of top models
                            import matplotlib.pyplot as _plt2
                            top = lb.head(min(10, len(lb)))
                            val_col = "score_val" if "score_val" in top.columns else top.select_dtypes("number").columns[0]
                            _fig_lb = _make_leaderboard_bar(
                                tuple(top["model"].tolist()),
                                tuple(top[val_col].tolist()),
                                val_col, "Top Models by Score", "#58a6ff"
                            )
                            st.pyplot(_fig_lb, use_container_width=True)
                            best_model = lb.iloc[0]["model"] if "model" in lb.columns else "N/A"
                            st.success(f"✅ Best model: **{best_model}**")
                        except Exception as lb_err:
                            st.warning(f"Could not render leaderboard: {lb_err}")

                    elif fw_type == "flaml" and predictor:
                        try:
                            st.markdown(f"""
                            <div class="metric-pill" style="display:inline-flex;margin-bottom:16px;">
                                <div><div class="m-label">Best Estimator</div>
                                <div class="m-value" style="color:#3fb950">{predictor.best_estimator}</div></div>
                            </div>""", unsafe_allow_html=True)
                            st.markdown("**⚙️ Best Configuration**")
                            st.json(predictor.best_config if hasattr(predictor, "best_config") else {})
                        except Exception as fe:
                            st.warning(f"Could not read FLAML results: {fe}")

                    elif fw_type == "h2o" and predictor:
                        if predictor.leader:
                            st.success(f"✅ Best model: **{predictor.leader.model_id}**")
                            lb_key = f"lb_df_{entry.key}"
                            if lb_key not in st.session_state or st.button("🔄 Refresh", key=f"h2o_ref_{entry.key}"):
                                try:
                                    st.session_state[lb_key] = predictor.leaderboard.as_data_frame()
                                except Exception as h2o_lb_err:
                                    st.warning(f"Leaderboard: {h2o_lb_err}")
                                    st.session_state[lb_key] = None
                            lb_df = st.session_state.get(lb_key)
                            if lb_df is not None:
                                st.dataframe(lb_df, use_container_width=True)
                                id_col  = lb_df.columns[0]
                                num_cols = lb_df.select_dtypes("number").columns.tolist()
                                if num_cols:
                                    metric_col = num_cols[0]
                                    top_h2o = lb_df.head(10)
                                    _fig_h2o = _make_leaderboard_bar(
                                        tuple(top_h2o[id_col].tolist()),
                                        tuple(top_h2o[metric_col].tolist()),
                                        metric_col, "H2O Model Leaderboard", "#3fb950"
                                    )
                                    st.pyplot(_fig_h2o, use_container_width=True)

                    elif fw_type == "tpot" and predictor:
                        from src.pipeline_parser import extract_best_tpot_pipeline
                        best_pipe = extract_best_tpot_pipeline(entry.all_logs)
                        if best_pipe:
                            st.markdown("**🧬 Best Pipeline (from logs)**")
                            st.code(best_pipe, language="python")
                            pipe_bytes = best_pipe.encode()
                            st.download_button("📥 Download Pipeline", pipe_bytes, "best_pipeline.py", "text/plain", key=f"dl_{entry.key}")
                        elif hasattr(predictor, "fitted_pipeline_"):
                            try:
                                import sklearn
                                pipe_str = str(predictor.fitted_pipeline_)
                                st.code(pipe_str, language="python")
                                st.download_button("📥 Download Pipeline", pipe_str.encode(), "best_pipeline.py", key=f"dl2_{entry.key}")
                            except Exception:
                                pass
                        else:
                            st.info("Best pipeline will appear here after training completes.")
                    elif entry.status == "running":
                        tel = entry.latest_telemetry
                        if tel:
                            st.subheader("📡 Live Analytics")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Models Trained", tel.get("models_trained", tel.get("iterations", 0)))
                            
                            best_val = tel.get("best_value", tel.get("best_loss", 0.0))
                            metric_name = tel.get("best_metric", "Score")
                            col2.metric(f"Current {metric_name}", f"{best_val:.4f}")
                            
                            elapsed = entry.elapsed_str()
                            col3.metric("Running For", elapsed)
                            
                            if "leaderboard_preview" in tel:
                                st.markdown("**Top Models (Partial)**")
                                st.dataframe(pd.DataFrame(tel["leaderboard_preview"]), use_container_width=True)
                            elif "best_estimator" in tel:
                                st.success(f"Best so far: **{tel['best_estimator']}**")
                                with st.expander("Best Config Preview"):
                                    st.code(tel.get("best_config_preview", "N/A"))
                        else:
                            st.info("🔄 Training in progress... analytics will appear shortly.")
                    else:
                        st.info("No result available for inspection.")

                with tab_mlflow:
                    if run_id:
                        try:
                            run_data = _get_mlflow_run(run_id)
                            if run_data:
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("**⚙️ Parameters**")
                                    if run_data.data.params:
                                        st.dataframe(
                                            pd.DataFrame([{"Parameter": k, "Value": v} for k, v in run_data.data.params.items()]),
                                            use_container_width=True
                                        )
                                    else:
                                        st.caption("No parameters logged.")
                                with c2:
                                    st.markdown("**📊 Metrics**")
                                    if run_data.data.metrics:
                                        st.dataframe(
                                            pd.DataFrame([{"Metric": k, "Value": round(v, 6)} for k, v in run_data.data.metrics.items()]),
                                            use_container_width=True
                                        )
                                    else:
                                        st.caption("No metrics logged.")
                                st.markdown("**📦 Artifacts**")
                                arts = _get_mlflow_artifacts(run_id)
                                if arts:
                                    for art in arts:
                                        size_str = f"{art.file_size:,} bytes" if art.file_size else "dir"
                                        st.markdown(f'<span style="color:#79c0ff">📄 `{art.path}`</span> <span style="color:#8b949e;font-size:11px">({size_str})</span>', unsafe_allow_html=True)
                                else:
                                    st.caption("No artifacts logged yet.")
                                # Run metadata
                                with st.expander("📋 Run Metadata"):
                                    meta = {
                                        "Experiment ID": run_data.info.experiment_id,
                                        "Run ID": run_data.info.run_id,
                                        "Status": run_data.info.status,
                                        "Start Time": pd.to_datetime(run_data.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S") if run_data.info.start_time else "N/A",
                                    }
                                    for k, v in meta.items():
                                        st.markdown(f"**{k}:** `{v}`")
                            else:
                                st.info("MLflow data is being fetched…")
                        except Exception as mfe:
                            st.warning(f"Could not load MLflow details: {mfe}")
                    else:
                        st.info("MLflow Run ID not available yet — training may still be initializing.")

                with tab_code:
                    if run_id:
                        try:
                            from src.code_gen_utils import generate_consumption_code, generate_api_deployment
                            fw_key_code = entry.metadata.get("framework_key", "unknown")
                            target_col  = entry.metadata.get("target", "target")
                            code_snippet = generate_consumption_code(fw_key_code, run_id, target_col)
                            st.markdown("**💻 Model Consumption Code**")
                            st.code(code_snippet, language="python")
                            st.download_button("📥 Download Script", code_snippet.encode(), "consume_model.py", "text/plain", key=f"dlcode_{entry.key}")
                            st.markdown("---")
                            st.markdown("**🚀 One-Click API Deployment**")
                            deploy_dir = f"deploy_{entry.key}"
                            if st.button("🐳 Generate FastAPI + Docker Package", key=f"deploy_{entry.key}", type="primary"):
                                generate_api_deployment(fw_key_code, run_id, target_col, output_dir=deploy_dir)
                                st.success(f"✅ Ready at `{deploy_dir}/` — includes `main.py`, `Dockerfile`, and `requirements.txt`.")
                        except Exception as ce:
                            st.warning(f"Could not generate code: {ce}")
                    else:
                        st.info("Consumption code will appear here after training completes.")

                if entry.status == "failed":
                    err = entry.result.get("error", "Unknown") if entry.result else "Unknown"
                    tb  = entry.result.get("traceback", "") if entry.result else ""
                    st.markdown(f"""
                    <div style="background:#2a0a0a;border:1px solid #f85149;border-radius:8px;padding:16px;margin-top:8px;">
                        <div style="color:#f85149;font-weight:600;margin-bottom:8px;">❌ Training Failed</div>
                        <div style="color:#ff7b72;font-family:'JetBrains Mono',monospace;font-size:12px;white-space:pre-wrap;">{err}</div>
                    </div>""", unsafe_allow_html=True)
                    if tb:
                        with st.expander("🔍 Traceback"):
                            st.code(tb, language="python")

    render_experiment_dashboard()


    st.header("🔮 Prediction")
    
    load_option = st.radio("Choose the model source", ["Current session model", "Load from MLflow runs", "Load from ONNX / Hugging Face"])
    
    if load_option == "Load from MLflow runs":
        col1, col2 = st.columns(2)
        m_type = col1.selectbox("Model Framework", ["AutoGluon", "FLAML", "H2O AutoML", "TPOT", "PyCaret", "Lale"])
        run_id_input = col2.text_input("Run ID")
        
        if st.button("Load Model"):
            try:
                if m_type == "AutoGluon":
                    from src.autogluon_utils import load_model_from_mlflow
                    st.session_state['predictor'] = load_model_from_mlflow(run_id_input)
                    st.session_state['model_type'] = "autogluon"
                elif m_type == "FLAML":
                    from src.flaml_utils import load_flaml_model
                    st.session_state['predictor'] = load_flaml_model(run_id_input)
                    st.session_state['model_type'] = "flaml"
                elif m_type == "H2O AutoML":
                    from src.h2o_utils import load_h2o_model
                    st.session_state['predictor'] = load_h2o_model(run_id_input)
                    st.session_state['model_type'] = "h2o"
                elif m_type == "TPOT":
                    from src.tpot_utils import load_tpot_model
                    st.session_state['predictor'] = load_tpot_model(run_id_input)
                    st.session_state['model_type'] = "tpot"
                elif m_type == "PyCaret":
                    import mlflow, joblib, os
                    from pycaret.classification import load_model as _pc_load
                    local_path = mlflow.artifacts.download_artifacts(run_id=run_id_input, artifact_path="model")
                    mpath = None
                    for root, _, files in os.walk(local_path):
                        for f in files:
                            if f.endswith(".pkl"):
                                mpath = os.path.join(root, f).replace(".pkl", "")
                                break
                    if mpath is None:
                        raise FileNotFoundError("PyCaret .pkl not found.")
                    st.session_state['predictor'] = _pc_load(mpath)
                    st.session_state['model_type'] = "pycaret"
                elif m_type == "Lale":
                    import mlflow, joblib, os
                    local_path = mlflow.artifacts.download_artifacts(run_id=run_id_input, artifact_path="model")
                    bundle = None
                    for root, _, files in os.walk(local_path):
                        for f in files:
                            if f.endswith(".pkl"):
                                bundle = joblib.load(os.path.join(root, f))
                                break
                    if bundle is None:
                        raise FileNotFoundError("Lale .pkl not found.")
                    st.session_state['predictor'] = bundle
                    st.session_state['model_type'] = "lale"
                
                st.session_state['run_id'] = run_id_input
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Loading error: {e}")

    elif load_option == "Load from ONNX / Hugging Face":
        st.markdown("##### 📦 Load from External Sources")
        src_mode = st.radio("Source", ["Local ONNX File", "Hugging Face Hub"], horizontal=True)
        if src_mode == "Local ONNX File":
            onnx_file = st.file_uploader("Upload .onnx file", type=["onnx"])
            if onnx_file:
                try:
                    from src.onnx_utils import load_onnx_session
                    # Save temp file
                    os.makedirs("temp", exist_ok=True)
                    temp_path = os.path.join("temp", onnx_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(onnx_file.getbuffer())
                    st.session_state['predictor'] = load_onnx_session(temp_path)
                    st.session_state['model_type'] = "onnx"
                    st.success("ONNX session loaded!")
                except Exception as oe:
                    st.error(f"ONNX error: {oe}")
        else:
            hf_repo_search = st.text_input("HF Repo ID")
            hf_file_search = st.text_input("Filename in Repo (e.g. model.onnx or model.pkl)", value="model.onnx")
            if st.button("Download & Load"):
                try:
                    from src.huggingface_utils import HuggingFaceService
                    hf = HuggingFaceService()
                    local_path = hf.download_model(hf_repo_search, hf_file_search)
                    if hf_file_search.endswith(".onnx"):
                        from src.onnx_utils import load_onnx_session
                        st.session_state['predictor'] = load_onnx_session(local_path)
                        st.session_state['model_type'] = "onnx"
                    else:
                        st.info("File downloaded. Generic loading for non-ONNX files from Hub is not yet unified. Please load via MLflow if registered.")
                    st.success(f"Loaded {hf_file_search} from Hub!")
                except Exception as he:
                    st.error(f"Hub error: {he}")

    if st.session_state.get('predictor') is not None:
        predictor = st.session_state.get('predictor')
        m_type = st.session_state.get('model_type')
        if not m_type:
            st.error("Loaded model is missing model_type in session. Reload the model before predicting.")
            st.stop()
        run_id = st.session_state.get('run_id', 'N/A')
        
        st.info(f"Active model: {m_type}")
        
        with st.expander("💻 View Model Consumption Code"):
            try:
                from src.code_gen_utils import generate_consumption_code
                code_sample = generate_consumption_code(m_type, run_id, "target")
                st.code(code_sample, language="python")
            except Exception as e:
                st.warning(f"Could not generate code sample: {e}")
        
        input_mode = st.radio("Input Mode", ["Batch Prediction (CSV/Excel)", "Real-time Prediction (Manual Entry)"], horizontal=True)
        
        # execute_pred and predict_df must always be defined to avoid NameError
        execute_pred = False
        predict_df = None

        if input_mode == "Batch Prediction (CSV/Excel)":
            predict_file = st.file_uploader("Upload prediction dataset", type=["csv", "xlsx", "xls"])
            if predict_file is not None:
                from src.data_utils import load_data
                predict_df = load_data(predict_file)
                st.dataframe(predict_df.head())
                execute_pred = st.button("Execute Prediction")
        else:
            st.subheader("📝 Manual Entry")
            # Try to get features from session state DF first
            features = []
            df_session = st.session_state.get('df')
            if df_session is not None:
                # Assuming all columns except target are features
                target_col = st.session_state.get('target', None)
                features = [c for c in df_session.columns if c != target_col]
            else:
                st.warning("Feature list unknown (Training data not in session). Please upload a file once to identify features, or use File Upload.")
                features = []
            
            if features:
                manual_data = {}
                cols = st.columns(min(len(features), 3))
                for i, feat in enumerate(features):
                    with cols[i % 3]:
                        # Basic guess of type based on training data
                        dtype = df_session[feat].dtype
                        if pd.api.types.is_numeric_dtype(dtype):
                            manual_data[feat] = st.number_input(feat, value=float(df_session[feat].median()))
                        else:
                            options = df_session[feat].unique().tolist()
                            manual_data[feat] = st.selectbox(feat, options)
                
                predict_df = pd.DataFrame([manual_data])
                execute_pred = st.button("Confirm Manual Prediction")
            # (else: no features — execute_pred stays False)

        if execute_pred and predict_df is not None:
                try:
                    if predictor is None:
                        st.error("No model is loaded. Please train or load a model first.")
                        st.stop()

                    # Always drop the target column if the user uploaded it
                    target_col = st.session_state.get('target', None)
                    if target_col and target_col in predict_df.columns:
                        pred_input_df = predict_df.drop(columns=[target_col])
                    else:
                        pred_input_df = predict_df.copy()

                    if m_type == "autogluon":
                        predictions = predictor.predict(pred_input_df)
                    elif m_type == "onnx":
                        from src.onnx_utils import predict_onnx
                        predictions = predict_onnx(predictor, pred_input_df)
                    elif m_type == "h2o":
                        from src.h2o_utils import predict_with_h2o
                        predictions = predict_with_h2o(predictor, pred_input_df)
                    elif m_type == "pycaret":
                        from pycaret.classification import predict_model as _pc_pred
                        preds_df = _pc_pred(predictor, data=pred_input_df)
                        label_col = "prediction_label" if "prediction_label" in preds_df.columns else preds_df.columns[-1]
                        predictions = preds_df[label_col]
                    elif m_type == "lale":
                        import joblib, numpy as np
                        # predictor is a bundle dict: {model, col_encoders, y_encoder}
                        if isinstance(predictor, dict):
                            _model = predictor["model"]
                            _col_enc = predictor.get("col_encoders", {})
                            _y_enc   = predictor.get("y_encoder", None)
                        else:
                            _model, _col_enc, _y_enc = predictor, {}, None
                        _df = pred_input_df.copy()
                        # Ensure only features that were present during training are used
                        # and apply encoders
                        for col, enc in _col_enc.items():
                            if col in _df.columns:
                                _df[col] = enc.transform(_df[[col]].astype(str)).ravel()
                        
                        # Convert to numeric to find any missed strings
                        for col in _df.columns:
                            if _df[col].dtype == object:
                                try:
                                    _df[col] = pd.to_numeric(_df[col])
                                except:
                                    # Fallback: if it's still string, it's a new feature not in col_encoders
                                    # or it's a feature we didn't encode. Let's try to fill with -1 or 0
                                    _df[col] = 0.0 # or drop it
                        
                        raw = _model.predict(_df.values)
                        predictions = _y_enc.inverse_transform(raw) if _y_enc else raw
                    else:  # flaml / tpot
                        predictions = predictor.predict(pred_input_df)
                    
                    # --- POST-PROCESSING: Decode numeric IDs to class names ---
                    try:
                        target_session = st.session_state.get('target', None)
                        train_df_ref = st.session_state.get('df')
                        if target_session and train_df_ref is not None:
                            if target_session in train_df_ref.columns:
                                trg_series = train_df_ref[target_session]
                                if trg_series.dtype == object or str(trg_series.dtype) == 'category':
                                    pred_s = pd.Series(predictions)
                                    # If the model output numeric IDs but target was string:
                                    if pd.api.types.is_numeric_dtype(pred_s):
                                        from sklearn.preprocessing import LabelEncoder
                                        le = LabelEncoder()
                                        le.fit(trg_series.astype(str))
                                        
                                        decoded = []
                                        for p in pred_s:
                                            try:
                                                idx = int(p)
                                                if 0 <= idx < len(le.classes_):
                                                    decoded.append(le.inverse_transform([idx])[0])
                                                else:
                                                    decoded.append(p)
                                            except:
                                                decoded.append(p)
                                        predictions = decoded
                    except Exception as dec_err:
                        # Non-fatal decoding error
                        import logging
                        logging.warning(f"Could not decode class names: {dec_err}")
                    # ----------------------------------------------------------

                    result_df = pred_input_df.copy()
                    result_df['Predictions'] = predictions

                    st.success("Predictions concluded!")
                    st.dataframe(result_df)

                    # ─── XAI (Explainable AI) for Tabular Predictions ───
                    # Only show XAI for single manual entries to avoid lag on large file uploads
                    if input_mode == "Real-time Prediction (Manual Entry)":
                        st.markdown("---")
                        if st.button("🧠 Explain Prediction (SHAP)"):
                            with st.spinner("Generating Explanations..."):
                                from src.xai_utils import generate_shap_explanation
                                train_data_ref = st.session_state.get('df')
                                target_ref = st.session_state.get('target', "target")
                                
                                if train_data_ref is not None:
                                    bg_data = train_data_ref.drop(columns=[target_ref], errors='ignore')
                                    # For local explanation, evaluate on the single entry
                                    fig = generate_shap_explanation(
                                        model=predictor, 
                                        X_train=bg_data, 
                                        X_valid=pred_input_df, # single row
                                        task_type=st.session_state.get('task_type', "Classification")
                                    )
                                    if fig is not None:
                                        st.pyplot(fig)
                                        st.info("The Waterfall/Summary plot shows how each feature pushed the model output from the base value to the final prediction.")
                                    else:
                                        st.error("SHAP explanation not supported for this model architecture.")
                                else:
                                    st.warning("Training data not available in session to generate background SHAP values.")

                    # ─── XAI (Explainable AI) for Computer Vision Predictions ───
                    if input_mode == "Real-time Prediction (Manual Entry)" and "Computer Vision" in st.session_state.get('task_type', ""):
                        st.markdown("---")
                        if st.button("👁️ Explain AI Decision (Saliency Map)"):
                            with st.spinner("Generating Saliency Map... (This might take a minute depending on image size)"):
                                from src.xai_utils import generate_cv_saliency_map
                                
                                # Find the image path in the predictions dataframe
                                img_path = None
                                if 'image' in pred_input_df.columns:
                                    img_path = pred_input_df['image'].iloc[0]
                                elif 'image_path' in pred_input_df.columns:
                                    img_path = pred_input_df['image_path'].iloc[0]
                                else:
                                    # Try to guess which column might be an image path
                                    for col in pred_input_df.columns:
                                        val = str(pred_input_df[col].iloc[0])
                                        if val.lower().endswith(('.png', '.jpg', '.jpeg')):
                                            img_path = val
                                            break
                                            
                                if img_path and os.path.exists(img_path):
                                    fig = generate_cv_saliency_map(
                                        model=predictor, 
                                        image_path=img_path
                                    )
                                    if fig is not None:
                                        st.pyplot(fig)
                                        st.info("The Occlusion Saliency Map highlights regions of the image that the model found most informative for its prediction (warmer colors = more important).")
                                    else:
                                        st.error("Failed to generate Saliency Map for this model.")
                                else:
                                    st.warning("Could not identify a valid image path in the input data to generate an explanation.")

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
