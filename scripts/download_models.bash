#!/usr/bin/env bash
# Pre-fetch and verify SmolVLA assets in the project-local Hugging Face cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "사용법: ./launchfiles/download_models.bash config/system.yaml" >&2
  exit 2
fi
CONFIG_PATH="$1"
source "$ROOT/scripts/setup_env.bash"

HF_BIN="$SO101_VENV_DIR/bin/hf"
if [[ ! -x "$HF_BIN" ]]; then
  echo "hf CLI가 없습니다. 먼저 ./scripts/bootstrap_environment.bash config/system.yaml을 실행하세요." >&2
  exit 1
fi

mapfile -t MODEL_SETTINGS < <("$SO101_VENV_DIR/bin/python" - "$CONFIG_PATH" "$ROOT" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
for name in ("base", "vlm"):
    item = config["model"][name]
    path = Path(item["path"])
    if not path.is_absolute():
        path = root / path
    print(item["repo_id"])
    print(item["revision"])
    print(path.resolve())
PY
)
MODEL_ID="${MODEL_SETTINGS[0]}"
MODEL_REVISION="${MODEL_SETTINGS[1]}"
MODEL_LOCAL_DIR="${MODEL_SETTINGS[2]}"
VLM_ID="${MODEL_SETTINGS[3]}"
VLM_REVISION="${MODEL_SETTINGS[4]}"
VLM_LOCAL_DIR="${MODEL_SETTINGS[5]}"

mkdir -p "$ROOT/data/models"
echo "Downloading $MODEL_ID@$MODEL_REVISION ..."
MODEL_SNAPSHOT="$($HF_BIN download "$MODEL_ID" --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_LOCAL_DIR" --quiet)"

# SmolVLA constructs the VLM architecture and processor from this repository, but the
# fine-tuned SmolVLA checkpoint already contains the model parameters. Avoid downloading
# a second copy of the backbone weights on an 8 GB Jetson.
echo "Downloading $VLM_ID@$VLM_REVISION metadata/tokenizer (weights excluded) ..."
VLM_SNAPSHOT="$($HF_BIN download "$VLM_ID" --revision "$VLM_REVISION" \
  --local-dir "$VLM_LOCAL_DIR" \
  --exclude '*.safetensors' --exclude '*.bin' --exclude '*.gguf' \
  --exclude '*.onnx' --exclude '*.h5' --exclude '*.msgpack' --quiet)"

"$HF_BIN" cache verify "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_LOCAL_DIR"
"$HF_BIN" cache verify "$VLM_ID" --revision "$VLM_REVISION" --local-dir "$VLM_LOCAL_DIR"

printf 'SmolVLA snapshot: %s\nVLM metadata snapshot: %s\n' "$MODEL_SNAPSHOT" "$VLM_SNAPSHOT"
echo "Offline use: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
