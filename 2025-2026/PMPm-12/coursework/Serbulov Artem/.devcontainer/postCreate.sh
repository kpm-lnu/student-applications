#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/course_work_2026

# Prefer Python 3.12+ if available, fall back to default python.
PYTHON_BIN="python"
if command -v python3.12 >/dev/null 2>&1; then
    if python3.12 -c "import sys; assert sys.version_info[:2] >= (3, 12)" >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
    fi
fi

echo "Using interpreter: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V

# Ensure pip is present/updated, then install the project in editable mode.
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -e .

# Sanity check
"${PYTHON_BIN}" -c "import sys; print('executable:', sys.executable)"

# Install pre-commit hooks
pre-commit install
