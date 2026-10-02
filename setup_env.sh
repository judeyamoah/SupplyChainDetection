#!/usr/bin/env bash
# Creates a project-local virtual environment in ./.venv and installs everything the pipeline needs.
# Run from the project root (the folder that contains scripts/, data/, results/):   bash setup_env.sh
# Optional: choose the interpreter with   PYTHON=python3.12 bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"' \
  || { echo "Install a newer Python first, e.g.: brew install python@3.12"; exit 1; }

# XGBoost on macOS may need the OpenMP runtime (libomp). Try Homebrew, but do not stop if Homebrew is broken:
# check_env.py below tells you whether xgboost actually loads, and prints the fix if it does not.
if [[ "$(uname)" == "Darwin" ]]; then
  if command -v brew >/dev/null 2>&1; then
    brew list libomp >/dev/null 2>&1 || brew install libomp \
      || echo "WARNING: could not install libomp with Homebrew. Continuing; check_env.py will show whether xgboost still works."
  else
    echo "Homebrew not found; continuing. If xgboost fails to import, install libomp (see check_env.py output)."
  fi
fi

[[ -d .venv ]] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ -f requirements.lock.txt ]]; then
  echo "Found requirements.lock.txt: installing the exact recorded versions."
  pip install -r requirements.lock.txt
else
  pip install -r requirements.txt
fi
python check_env.py
[[ -f requirements.lock.txt ]] || pip freeze > requirements.lock.txt
echo
echo "Done. Activate the environment in every new terminal with:  source .venv/bin/activate"