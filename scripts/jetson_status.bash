#!/usr/bin/env bash
# Read-only Jetson/thermal/power-mode diagnostics.
set -euo pipefail

echo "=== Platform ==="
tr -d '\0' </proc/device-tree/model 2>/dev/null || uname -a
echo

echo "=== JetPack / L4T ==="
if [[ -r /etc/nv_tegra_release ]]; then
  sed -n '1p' /etc/nv_tegra_release
else
  dpkg-query --showformat='${Version}\n' --show nvidia-l4t-core 2>/dev/null || echo "nvidia-l4t-core not found"
fi

echo "=== Power mode ==="
if command -v nvpmodel >/dev/null 2>&1; then
  nvpmodel -q
else
  echo "nvpmodel not found"
fi

echo "=== CUDA tools ==="
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -n 1 || echo "nvcc not found"
command -v nsys >/dev/null 2>&1 && nsys --version || echo "nsys not found"
command -v ncu >/dev/null 2>&1 && ncu --version || echo "ncu not found"

echo "=== Python / PyTorch ==="
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/libraries/venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
"$PYTHON" - <<'PY'
import platform
try:
    import torch
except ImportError:
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    print("torch not installed")
else:
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    print(f"Torch: {torch.__version__}; CUDA build: {torch.version.cuda}; available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo "=== Memory ==="
free -h

echo "=== One tegrastats sample ==="
if command -v tegrastats >/dev/null 2>&1; then
  timeout 2 tegrastats --interval 1000 || true
else
  echo "tegrastats not found"
fi
