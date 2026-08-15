#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/libraries/venv/bin/python" "$ROOT/scripts/sync_artifacts.py" pull-model "$@"
