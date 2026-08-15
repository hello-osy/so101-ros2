#!/usr/bin/env bash
# Reproduce the local LeRobot + Jetson virtual environment from a clean repo clone.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "사용법: ./scripts/bootstrap_environment.bash config/system.yaml" >&2
  exit 2
fi
CONFIG_PATH="$1"
mapfile -t SYSTEM_VERSIONS < <(python3 - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
import yaml

system = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))["system"]
print(system["lerobot_revision"])
print(system["torch"])
print(system["torchvision"])
PY
)
mkdir -p "$PROJECT_ROOT/libraries"
VENV_DIR="$PROJECT_ROOT/libraries/venv"
LEROBOT_DIR="$PROJECT_ROOT/libraries/lerobot"
LEROBOT_REV="${SYSTEM_VERSIONS[0]}"
EXPECTED_TORCH="${SYSTEM_VERSIONS[1]}"
EXPECTED_TORCHVISION="${SYSTEM_VERSIONS[2]}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating $VENV_DIR with access to Jetson system packages..."
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"

"$PYTHON" - "$EXPECTED_TORCH" "$EXPECTED_TORCHVISION" <<'PY'
import sys

try:
    import torch
    import torchvision
except ImportError as exc:
    raise SystemExit(
        "Install the NVIDIA Jetson PyTorch and torchvision wheels before running this script."
    ) from exc

expected_torch, expected_torchvision = sys.argv[1:]
if torch.__version__ != expected_torch or torchvision.__version__ != expected_torchvision:
    raise SystemExit(
        "Unexpected Jetson packages: "
        f"torch={torch.__version__}, torchvision={torchvision.__version__}; "
        f"expected {expected_torch}, {expected_torchvision}. "
        "Override EXPECTED_TORCH/EXPECTED_TORCHVISION only if this is intentional."
    )
print(f"Jetson Torch OK: {torch.__version__} / {torchvision.__version__}")
PY

if [[ ! -d "$LEROBOT_DIR/.git" ]]; then
  echo "Cloning LeRobot into $LEROBOT_DIR..."
  git clone --filter=blob:none https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
  git -C "$LEROBOT_DIR" checkout --detach "$LEROBOT_REV"
else
  current_rev="$(git -C "$LEROBOT_DIR" rev-parse HEAD)"
  if [[ "$current_rev" != "$LEROBOT_REV" ]]; then
    echo "Existing LeRobot checkout is $current_rev; reproducible revision is $LEROBOT_REV." >&2
    echo "Move/remove libraries/lerobot, or intentionally set LEROBOT_REV=$current_rev." >&2
    exit 1
  fi
fi

"$PYTHON" - "$LEROBOT_DIR/pyproject.toml" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
replacements = {
    '"torch>=2.7,<2.12.0"': '"torch>=2.7,<2.14.0"',
    '"torchvision>=0.22.0,<0.27.0"': '"torchvision>=0.22.0,<0.29.0"',
}
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise SystemExit(f"LeRobot dependency line was not recognized: {old}")
path.write_text(text)
print("LeRobot dependency ranges patched for Jetson CUDA 13.2.")
PY

printf 'torch==%s\ntorchvision==%s\n' \
  "$EXPECTED_TORCH" "$EXPECTED_TORCHVISION" \
  >"$LEROBOT_DIR/jetson-constraints.txt"

SO101_LEROBOT_DIR="$LEROBOT_DIR" "$PROJECT_ROOT/scripts/apply_lerobot_patches.bash"

"$PYTHON" -m pip install --upgrade pip "setuptools<81" wheel
"$PYTHON" -m pip install \
  --constraint "$LEROBOT_DIR/jetson-constraints.txt" \
  --editable "$LEROBOT_DIR[core_scripts,feetech,training,smolvla,peft]"
"$PYTHON" -m pip install \
  "numpy==2.2.6" \
  "pandas==2.2.3" \
  "scipy==1.15.3" \
  "cffi>=1.17"

source "$PROJECT_ROOT/scripts/setup_env.bash"
python - <<'PY'
import lerobot
import scservo_sdk
import torch
import torchvision

print("LeRobot:", lerobot.__file__)
print("Feetech SDK:", scservo_sdk.__file__)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
elif __import__("os").environ.get("REQUIRE_CUDA", "1") == "1":
    raise SystemExit(
        "CUDA is not available to PyTorch. Check the JetPack/PyTorch wheel pairing. "
        "Set REQUIRE_CUDA=0 only for a non-Jetson static-check environment."
    )
PY

echo "LeRobot environment bootstrap complete. Next: ./launchfiles/download_models.bash $CONFIG_PATH"
