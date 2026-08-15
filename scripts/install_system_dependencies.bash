#!/usr/bin/env bash
# Install the Ubuntu packages needed by LeRobot, cameras, video decoding, and profiling.
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "warning: this project targets Jetson aarch64; detected $(uname -m)." >&2
fi

sudo apt-get update
sudo apt-get install -y \
  git git-lfs ffmpeg cmake build-essential python3-dev python3-venv python3-yaml \
  pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev v4l-utils

git lfs install
echo "System dependencies installed. Install NVIDIA Jetson PyTorch, then run ./scripts/bootstrap_environment.bash config/system.yaml."
