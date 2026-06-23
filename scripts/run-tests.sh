#!/bin/bash
set -euo pipefail

if [ -x .venv/bin/python ]; then
    PYTHON_BIN=.venv/bin/python
else
    PYTHON_BIN=python3
fi

PYTHONPATH=. "$PYTHON_BIN" -m pytest tests/ "$@"
