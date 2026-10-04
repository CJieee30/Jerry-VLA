#!/usr/bin/env bash
# Build suite×skill frame index, then dump 32×8 calib for seeds 42 and 43.
# CPU / video decode; does not occupy a GPU.
# Usage:
#   bash run_dump_calib_suite_skill.sh
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FN_ROOT="$(cd "$ROOT/../.." && pwd)"
PY="${PYTHON:-/home/guorj/.miniforge3/envs/lerobot/bin/python}"
LOG="$FN_ROOT/results/logs/dump_calib_suite_skill.log"
mkdir -p "$FN_ROOT/results/logs" "$FN_ROOT/results/calib"

{
  echo "==== dump suite×skill calib $(date -Iseconds) ===="
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export PYTHONUNBUFFERED=1
  cd "$ROOT"
  "$PY" atomic_skill_index.py
  "$PY" dump_calib_suite_skill.py --seed 42
  "$PY" dump_calib_suite_skill.py --seed 43
  echo "==== done $(date -Iseconds) EXIT=$? ===="
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
