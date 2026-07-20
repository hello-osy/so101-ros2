#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/libraries/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

cd "$ROOT"
bash -n launchfiles/*.bash scripts/*.bash libraries/ros2_bridge/scripts/*.bash
"$PYTHON" -m compileall -q scripts libraries/ros2_bridge/so101_ros2 libraries/ros2_bridge/launch
env -u DISPLAY PYTHONPATH="$ROOT/libraries/ros2_bridge${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m unittest discover --start-directory libraries/ros2_bridge/test --verbose
env -u DISPLAY "$PYTHON" -m unittest discover --start-directory scripts/tests_for_codex --verbose

for launcher in calibrate collect_data train inference benchmark_inference profile_nsys profile_ncu; do
  "./launchfiles/$launcher.bash" --check >/dev/null
done

if [[ "${BUILD_ROS2:-0}" == "1" ]]; then
  ./libraries/ros2_bridge/scripts/build.bash
fi

echo "Static checks and hardware-free tests passed."
