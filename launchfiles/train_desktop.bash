#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SO101_VENV_DIR="$ROOT/libraries/venv-desktop"
exec "$SO101_VENV_DIR/bin/python" "$ROOT/scripts/training/run_training.py" \
  "$@" --profile training_desktop
