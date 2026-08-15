#!/usr/bin/env bash
# Explicitly select an nvpmodel mode and lock clocks for reproducible benchmarks.
set -euo pipefail

if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "usage: $0 NVP_MODEL_ID" >&2
  echo "Run 'sudo nvpmodel -q --verbose' first and choose the Super/MAXN mode supported by this Jetson." >&2
  exit 2
fi
if [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "NVP_MODEL_ID must be a non-negative integer." >&2
  exit 2
fi

sudo nvpmodel -m "$1"
sudo jetson_clocks
nvpmodel -q
echo "Clocks locked. Reboot or run 'sudo jetson_clocks --restore' (when supported) to restore defaults."
