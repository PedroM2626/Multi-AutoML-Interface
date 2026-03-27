"""
run.py - Entry point that ensures the app is launched with the correct Python (3.11).

Usage:
    python run.py
    py -3.11 run.py
"""
import sys
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
            )
            version_text = (result.stdout or "") + (result.stderr or "")
            if _is_target_python(version_text):
                return cmd_prefix
        except Exception:
            continue
    return None

def main():
    major = sys.version_info.major
    minor = sys.version_info.minor

    if major != REQUIRED_MAJOR or minor < REQUIRED_MINOR:
        # Try to re-launch using a discovered Python 3.11 interpreter
        py311 = _find_python_311_cmd()

        if py311 is None:
            print(
                f"ERROR: Python {REQUIRED_MAJOR}.{REQUIRED_MINOR} not found.\n"
                f"Currently running: Python {major}.{minor}\n"
                f"PyCaret and Lale require Python 3.11 with the correct scikit-learn version.\n"
                f"Please run:\n"
                f"  py -3.11 -m streamlit run app.py"
            )
            sys.exit(1)

        print(f"Re-launching with Python {REQUIRED_MAJOR}.{REQUIRED_MINOR}...")
        cmd = py311 + ["-m", "streamlit", "run", "app.py"] + sys.argv[1:]
        raise SystemExit(subprocess.call(cmd))
    else:
        # We are already in the correct interpreter, just start streamlit
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "app.py"] + sys.argv[1:]
        sys.exit(stcli.main())


if __name__ == "__main__":
    main()
