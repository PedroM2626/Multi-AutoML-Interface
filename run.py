"""
run.py - Entry point that ensures the app is launched with the correct Python (3.11).

Usage:
    python run.py
    py -3.11 run.py
"""
import sys
import os
import shutil
import subprocess

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 11


def _is_target_python(version_output: str) -> bool:
    return f"Python {REQUIRED_MAJOR}.{REQUIRED_MINOR}" in version_output


def _find_python_311_cmd():
    """Return a command prefix that launches Python 3.11, or None if unavailable."""
    candidates = [
        ["py", f"-{REQUIRED_MAJOR}.{REQUIRED_MINOR}"],
        ["python3.11"],
        ["python"],
    ]

    for cmd_prefix in candidates:
        exe = shutil.which(cmd_prefix[0])
        if not exe:
            continue
        try:
            result = subprocess.run(
                cmd_prefix + ["--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            version_text = (result.stdout or "") + (result.stderr or "")
            if _is_target_python(version_text):
                return cmd_prefix
        except Exception:
            continue
    return None

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
    major = sys.version_info.major
    minor = sys.version_info.minor

    if major == REQUIRED_MAJOR and minor == REQUIRED_MINOR:
        _start_streamlit()

    py311 = _find_python_311_cmd()
    if py311 is not None:
        # Try to re-launch using a discovered Python 3.11 interpreter
        print(f"Re-launching with Python {REQUIRED_MAJOR}.{REQUIRED_MINOR}...")
        cmd = py311 + ["-m", "streamlit", "run", "app.py"] + _streamlit_args()
        raise SystemExit(subprocess.call(cmd))

    if major != REQUIRED_MAJOR or minor < REQUIRED_MINOR:
        print(
            f"ERROR: Python {REQUIRED_MAJOR}.{REQUIRED_MINOR} not found.\n"
            f"Currently running: Python {major}.{minor}\n"
            f"Frameworks need scikit-learn built for this project's target interpreter.\n"
            f"Please run:\n"
            f"  py -3.11 -m streamlit run app.py"
        )
        sys.exit(1)

    # Newer interpreter and no 3.11 around: keep going, but say what is degraded.
    print(
        f"WARNING: running on Python {major}.{minor}; Python {REQUIRED_MAJOR}.{REQUIRED_MINOR} "
        "was not found. The app starts, but PyCaret and Lale may fail to import because they "
        "need scikit-learn packages built for "
        f"{REQUIRED_MAJOR}.{REQUIRED_MINOR}."
    )
    _start_streamlit()


if __name__ == "__main__":
    main()
