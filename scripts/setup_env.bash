#!/usr/bin/env bash
# Source this file after cloning/bootstrap. No user-home path is hard-coded.

_so101_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_so101_find_dir() {
  local explicit="$1"
  shift
  if [[ -n "$explicit" && -d "$explicit" ]]; then
    cd "$explicit" && pwd
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -d "$candidate" ]]; then
      cd "$candidate" && pwd
      return 0
    fi
  done
  return 1
}

SO101_VENV_DIR="$(_so101_find_dir "${SO101_VENV_DIR:-}" \
  "$_so101_root/libraries/venv")" || {
  echo "SO-101 error: libraries/venv was not found. Run ./scripts/bootstrap_environment.bash" >&2
  return 1 2>/dev/null || exit 1
}

SO101_LEROBOT_DIR="$(_so101_find_dir "${SO101_LEROBOT_DIR:-}" \
  "$_so101_root/libraries/lerobot")" || {
  echo "SO-101 error: libraries/lerobot was not found. Run ./scripts/bootstrap_environment.bash" >&2
  return 1 2>/dev/null || exit 1
}

export SO101_PROJECT_ROOT="$_so101_root"
export SO101_VENV_DIR
export SO101_LEROBOT_DIR
export VIRTUAL_ENV="$SO101_VENV_DIR"
export PATH="$SO101_VENV_DIR/bin:$PATH"
export PYTHONPATH="$SO101_LEROBOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$_so101_root/data/models/huggingface"
export HF_LEROBOT_HOME="$_so101_root/data/downloaded_datasets/lerobot"
export HF_LEROBOT_CALIBRATION="$_so101_root/data/calibration"

# A moved venv has stale absolute paths in activate and console-script shebangs.
# Calling its python symlink directly and adding LeRobot/src avoids those paths.
hash -r 2>/dev/null || true

echo "SO101_PROJECT_ROOT=$SO101_PROJECT_ROOT"
echo "SO101_VENV_DIR=$SO101_VENV_DIR"
echo "SO101_LEROBOT_DIR=$SO101_LEROBOT_DIR"
echo "HF_HOME=$HF_HOME"
