#!/usr/bin/env bash
# Usage: play_bag.bash BAG_DIRECTORY [additional ros2 bag play options]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$PROJECT_ROOT/scripts/setup_env.bash"
if [[ -f "$PROJECT_ROOT/install/setup.bash" ]]; then
  source "$PROJECT_ROOT/install/setup.bash"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BAG_DIRECTORY [ros2 bag play options]" >&2
  exit 2
fi
if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 is unavailable; source the ROS 2 installation first." >&2
  exit 1
fi

bag="$1"
shift
ros2 bag play "$bag" "$@"
