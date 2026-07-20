#!/usr/bin/env bash
set -euo pipefail

BRIDGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$BRIDGE_ROOT/../.." && pwd)"
source "$PROJECT_ROOT/scripts/setup_env.bash"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ROS 2 is not sourced. Install/source ROS 2 before building." >&2
  exit 1
fi
if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon is not installed or not on PATH." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
colcon build --base-paths "$BRIDGE_ROOT" --symlink-install --packages-select so101_ros2
echo "Build complete. Run: source $PROJECT_ROOT/install/setup.bash"
