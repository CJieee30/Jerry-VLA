#!/usr/bin/env bash
# Dump 32×8 held-out suite×skill calib for seeds 44 and 45.
# Excludes seed42/43 sampled indices. CPU / video decode; does not occupy a GPU.
# Usage:
#   bash run_dump_calib_suite_skill_heldout.sh
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FN_ROOT="$(cd "$ROOT/../.." && pwd)"
PY="${PYTHON:-/home/guorj/.miniforge3/envs/lerobot/bin/python}"
LOG="$FN_ROOT/results/logs/dump_calib_suite_skill_heldout.log"
mkdir -p "$FN_ROOT/results/logs" "$FN_ROOT/results/calib"

{
  echo "==== dump suite×skill held-out calib $(date -Iseconds) ===="
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export PYTHONUNBUFFERED=1
  cd "$ROOT"
  "$PY" dump_calib_suite_skill_heldout.py --seed 44
  "$PY" dump_calib_suite_skill_heldout.py --seed 45
  echo "==== done $(date -Iseconds) EXIT=$? ===="
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
