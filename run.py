"""
run.py - Entry point that ensures the app is launched with the correct Python (3.11).

Usage:
    python run.py
    py -3.11 run.py
"""
import sys
import os
import subprocess

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 11

def main():
    major = sys.version_info.major
    minor = sys.version_info.minor

    if major != REQUIRED_MAJOR or minor < REQUIRED_MINOR:
        # Try to re-launch with py -3.11
        py311 = None
        for candidate in ["py", "python3.11", "python3"]:
            try:
                result = subprocess.run(
                    [candidate, f"-{REQUIRED_MAJOR}.{REQUIRED_MINOR}", "--version"],
                    capture_output=True, text=True
                )
                if f"{REQUIRED_MAJOR}.{REQUIRED_MINOR}" in result.stdout:
                    py311 = [candidate, f"-{REQUIRED_MAJOR}.{REQUIRED_MINOR}"]
                    break
            except FileNotFoundError:
                continue

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
        os.execv(subprocess.check_output(["where", py311[0]]).decode().splitlines()[0], cmd)
    else:
        # We are already in the correct interpreter, just start streamlit
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "app.py"] + sys.argv[1:]
        sys.exit(stcli.main())


if __name__ == "__main__":
    main()
