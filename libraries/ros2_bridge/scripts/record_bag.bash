#!/usr/bin/env bash
# Usage: record_bag.bash NAME [CAMERA_TOPIC ...]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$PROJECT_ROOT/scripts/setup_env.bash"
if [[ -f "$PROJECT_ROOT/install/setup.bash" ]]; then
  source "$PROJECT_ROOT/install/setup.bash"
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 is unavailable; source the ROS 2 installation first." >&2
  exit 1
fi

name="${1:-recording_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi
output="$PROJECT_ROOT/data/rosbags/$name"
topics=("/so101/record/joint_states")
if [[ $# -gt 0 ]]; then
  topics+=("$@")
fi

mkdir -p "$PROJECT_ROOT/data/rosbags"
echo "Recording to $output"
echo "Topics: ${topics[*]}"
echo "Stop with Ctrl+C so rosbag can write its metadata."
ros2 bag record --output "$output" "${topics[@]}"
