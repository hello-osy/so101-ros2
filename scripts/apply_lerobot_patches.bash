#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_DIR="${SO101_LEROBOT_DIR:-$ROOT/libraries/lerobot}"

for patch in "$ROOT"/libraries/patches/*.patch; do
  if git -C "$LEROBOT_DIR" apply --check "$patch" 2>/dev/null; then
    git -C "$LEROBOT_DIR" apply "$patch"
    echo "Applied $(basename "$patch")"
  elif git -C "$LEROBOT_DIR" apply --reverse --check "$patch" 2>/dev/null; then
    echo "Already applied $(basename "$patch")"
  else
    echo "Patch does not match the pinned LeRobot revision: $patch" >&2
    exit 1
  fi
done
