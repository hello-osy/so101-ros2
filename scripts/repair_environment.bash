#!/usr/bin/env bash
# Recreate editable metadata and console scripts after moving the venv/repository.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.bash"

SO101_LEROBOT_DIR="$SO101_LEROBOT_DIR" "$SCRIPT_DIR/apply_lerobot_patches.bash"

"$SO101_VENV_DIR/bin/python" - "$SO101_VENV_DIR" <<'PY'
from pathlib import Path
import re
import sys

venv = Path(sys.argv[1]).resolve()
targets = [
    path
    for path in (venv / "bin").iterdir()
    if path.is_file() and not path.is_symlink()
]
targets.append(venv / "pyvenv.cfg")
old_venv_pattern = re.compile(
    rb"/[A-Za-z0-9._/-]*/(?:libraries/venv|venv-lerobot-jp72|venvs/lerobot-jp72)"
)
changed = 0

for path in targets:
    try:
        original = path.read_bytes()
    except OSError:
        continue
    if b"\0" in original[:4096]:
        continue
    updated = old_venv_pattern.sub(str(venv).encode(), original)
    if updated != original:
        path.write_bytes(updated)
        changed += 1

print(f"Repaired absolute venv paths in {changed} file(s).")
PY

"$SO101_VENV_DIR/bin/python" -m pip install \
  --no-deps \
  --no-build-isolation \
  --editable "$SO101_LEROBOT_DIR[core_scripts,feetech,training,smolvla,peft]"

"$SO101_VENV_DIR/bin/python" -c \
  'import lerobot; print(f"LeRobot import OK: {lerobot.__file__}")'

echo "Environment paths repaired. Re-run this script whenever either directory moves."
