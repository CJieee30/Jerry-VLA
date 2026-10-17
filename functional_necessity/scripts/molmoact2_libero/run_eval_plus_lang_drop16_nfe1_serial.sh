#!/usr/bin/env bash
# Drop-16 materialized Language NFE=1 — four suites serial, shared flock (one instance cluster-wide).
# Usage: bash run_eval_plus_lang_drop16_nfe1_serial.sh [GPU]
set -u
set -o pipefail

GPU="${1:-2}"
NFE=1
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PREFIX="molmoact2_drop16_materialized_nfe${NFE}_lang"
MASTER_LOG="$ROOT/results/logs/eval_molmoact2_drop16_nfe1_lang_serial_gpu${GPU}.log"
CLASS=/home/guorj/LIBERO-plus/libero/libero/benchmark/task_classification.json
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
CHECK="$ROOT/scripts/molmoact2_libero/check_plus_eval_integrity.py"
FLOCK="$ROOT/results/logs/molmo_drop16_nfe1_lang.flock"

exec 9>"$FLOCK"
if ! flock -n 9; then
  echo "==== drop16 nfe1 lang already running (flock) $(date -Iseconds) ====" | tee -a "$MASTER_LOG"
  exit 0
fi

echo "==== DROP16 NFE1 LANG GPU${GPU} START $(date -Iseconds) ====" | tee -a "$MASTER_LOG"

_run_suite() {
  local suffix=$1 suite=$2
  local TAG="${PREFIX}_${suffix}"
  local OUT="$ROOT/results/eval_libero_plus/$TAG"
  if "$PY" "$CHECK" --output-dir "$OUT" --suite "$suite" --category "Language Instructions" \
    --classification-json "$CLASS" >/tmp/plus_check.json 2>/dev/null; then
    echo "[skip complete] $TAG" | tee -a "$MASTER_LOG"
    return 0
  fi
  if pgrep -f "eval_plus_molmo.py.*${TAG}" >/dev/null 2>&1; then
    echo "[skip running] $TAG" | tee -a "$MASTER_LOG"
    return 0
  fi
  bash "$DIR/run_eval_plus_lang.sh" "$TAG" "$suite" "Language Instructions" "$NFE" "$GPU" drop16_materialized \
    || { echo "==== FAIL $TAG ====" | tee -a "$MASTER_LOG"; return 1; }
}

_run_suite spatial libero_spatial || exit 1
_run_suite object libero_object || exit 1
_run_suite goal libero_goal || exit 1
_run_suite 10 libero_10 || exit 1

echo "==== DROP16 NFE1 LANG GPU${GPU} DONE $(date -Iseconds) ====" | tee -a "$MASTER_LOG"
