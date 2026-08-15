#!/usr/bin/env bash
# Ubuntu 24.04 / Python 3.12 / NVIDIA Blackwell desktop training environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "사용법: ./scripts/bootstrap_desktop.bash config/system.yaml" >&2
  exit 2
fi
CONFIG="$1"

mapfile -t VALUES < <(python3 - "$CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
desktop = cfg["system"]["desktop"]
print(cfg["system"]["lerobot_revision"])
print(desktop["python"])
print(str(desktop["torch"]).split("+")[0])
print(str(desktop["torchvision"]).split("+")[0])
print(str(desktop["cuda"]).replace(".", ""))
PY
)
LEROBOT_REV="${VALUES[0]}"
PYTHON_VERSION="${VALUES[1]}"
TORCH_VERSION="${VALUES[2]}"
TORCHVISION_VERSION="${VALUES[3]}"
CUDA_TAG="cu${VALUES[4]}"
VENV="$ROOT/libraries/venv-desktop"
LEROBOT="$ROOT/libraries/lerobot"

command -v "python${PYTHON_VERSION}" >/dev/null || {
  echo "python${PYTHON_VERSION}이 없습니다. README의 데스크탑 apt 명령을 먼저 실행하세요." >&2
  exit 1
}
mkdir -p "$ROOT/libraries"
if [[ ! -x "$VENV/bin/python" ]]; then
  "python${PYTHON_VERSION}" -m venv "$VENV"
fi
PYTHON="$VENV/bin/python"

if [[ ! -d "$LEROBOT/.git" ]]; then
  git clone --filter=blob:none https://github.com/huggingface/lerobot.git "$LEROBOT"
  git -C "$LEROBOT" checkout --detach "$LEROBOT_REV"
elif [[ "$(git -C "$LEROBOT" rev-parse HEAD)" != "$LEROBOT_REV" ]]; then
  echo "libraries/lerobot revision이 system.yaml과 다릅니다. 별도 clone에서 다시 실행하세요." >&2
  exit 1
fi

"$PYTHON" - "$LEROBOT/pyproject.toml" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
for old, new in {
    '"torch>=2.7,<2.12.0"': '"torch>=2.7,<2.13.0"',
    '"torchvision>=0.22.0,<0.27.0"': '"torchvision>=0.22.0,<0.28.0"',
}.items():
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text and '"torch>=2.7,<2.14.0"' not in text and '"torchvision>=0.22.0,<0.29.0"' not in text:
        raise SystemExit(f"인식할 수 없는 LeRobot dependency: {old}")
path.write_text(text, encoding="utf-8")
PY

SO101_LEROBOT_DIR="$LEROBOT" "$ROOT/scripts/apply_lerobot_patches.bash"
"$PYTHON" -m pip install --upgrade pip "setuptools<81" wheel
"$PYTHON" -m pip install \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" \
  >"$LEROBOT/desktop-constraints.txt"
"$PYTHON" -m pip install \
  --constraint "$LEROBOT/desktop-constraints.txt" \
  --editable "$LEROBOT[core_scripts,training,smolvla,peft]"
"$PYTHON" -m pip install ninja

SO101_VENV_DIR="$VENV" SO101_LEROBOT_DIR="$LEROBOT" "$PYTHON" - <<'PY'
import torch
import torchvision

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA를 사용할 수 없습니다. NVIDIA driver와 nvidia-smi를 확인하세요.")
print("GPU:", torch.cuda.get_device_name(0))
major, _ = torch.cuda.get_device_capability(0)
if major < 12:
    print("주의: RTX 5070 Ti(Blackwell)가 아닌 GPU가 감지되었습니다.")
PY

echo "완료: ./launchfiles/download_models_desktop.bash $CONFIG"
