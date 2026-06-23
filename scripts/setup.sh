#!/bin/bash
set -euo pipefail

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --find-links=https://dd-trace-py-builds.s3.amazonaws.com/96035140/index.html -r requirements.txt

echo "Setup complete."
echo "Run tests with: bash scripts/run-tests.sh"
