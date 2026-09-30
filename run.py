"""
run.py - Local launcher: starts the Streamlit app on the current interpreter, bound to loopback.

Usage:
    python run.py
    python run.py --server.address 0.0.0.0
"""
import sys
import os

MIN_MINOR = 11


def _missing_engine_report():
    """Which catalog engines this interpreter cannot import, and what would add them.

    The app itself only needs the core stack, so a missing engine is a capability note and not a
    reason to refuse to start - the Training page hides those rows for the same reason. Only
    engines the catalog offers are named: AutoKeras has no release that runs, and promising it
    would be a false fix.
    """
    try:
        from src.task_catalog import TASK_FRAMEWORK_MAP, framework_available, install_hint
    except Exception as error:  # the app's own dependencies are missing
        return (
            "The app's own dependencies are missing, so it will not start: "
            f"{error}. Install them with `pip install -r requirements.txt`."
        )

    offered = {name for names in TASK_FRAMEWORK_MAP.values() for name in names}
    missing = sorted(name for name in offered if not framework_available(name))
    if not missing:
        return None
    return (
        f"engines this interpreter cannot import: {', '.join(missing)} - "
        f"`pip install {install_hint(missing)}`. PyCaret, Lale and TPOT need an older "
        "numpy/pandas/scikit-learn and Python 3.11 (PyCaret 3.3.2 refuses to import on 3.12); "
        "`pip install -r requirements-all.txt` installs every engine at once."
    )


def _streamlit_args():
    """
    Local launcher default: bind loopback. Streamlit listens on every interface when
    server.address is unset, which on a shared network would hand the app - and the
    machine's Python processes - to anyone on that network. Deployments that mean to be
    reachable set --server.address or STREAMLIT_SERVER_ADDRESS explicitly.
    """
    args = sys.argv[1:]
    configured = any(arg.startswith("--server.address") for arg in args)
    if not configured and not os.environ.get("STREAMLIT_SERVER_ADDRESS"):
        return ["--server.address", "127.0.0.1"] + args
    return args


def _start_streamlit():
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", "app.py"] + _streamlit_args()
    sys.exit(stcli.main())


def main():
    major, minor = sys.version_info.major, sys.version_info.minor
    if major != 3 or minor < MIN_MINOR:
        print(
            f"ERROR: this project pins libraries that need Python 3.{MIN_MINOR} or newer.\n"
            f"Currently running: Python {major}.{minor}\n"
            "Launch with a newer interpreter, e.g. `py -3.12 run.py`."
        )
        sys.exit(1)

    report = _missing_engine_report()
    if report:
        print(f"NOTE: {report}")

    _start_streamlit()


if __name__ == "__main__":
    main()
