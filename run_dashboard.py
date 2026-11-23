#!/usr/bin/env python3
"""Launcher script for the INRS Bayes Adaptive Filters Experiments Dashboard."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP_PATH = REPO_ROOT / "dashboard" / "app.py"


def main() -> None:
    # Prefer .venv Python if available
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    venv_streamlit = REPO_ROOT / ".venv" / "bin" / "streamlit"

    if venv_streamlit.is_file():
        cmd = [str(venv_streamlit), "run", str(APP_PATH)]
    elif venv_python.is_file():
        cmd = [str(venv_python), "-m", "streamlit", "run", str(APP_PATH)]
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]

    # Forward any extra CLI arguments
    cmd.extend(sys.argv[1:])

    print("=" * 70)
    print("Starting INRS Bayes Adaptive Filters Experiments Dashboard")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        sys.stderr.write(f"Error launching dashboard: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
