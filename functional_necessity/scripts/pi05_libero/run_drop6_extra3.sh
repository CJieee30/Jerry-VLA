#!/usr/bin/env bash
# Drop-6 + extra-3 scoring on v044.
# Usage:
#   bash run_drop6_extra3.sh dump
#   bash run_drop6_extra3.sh fisher 0
#   bash run_drop6_extra3.sh greedy_ppl 3
set -u
set -o pipefail

MODE="${1:?dump|fisher|greedy_ppl}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FN_ROOT="$(cd "$ROOT/../.." && pwd)"
PY="${PYTHON:-/home/guorj/.miniforge3/envs/lerobot/bin/python}"
CALIB_DIR="$FN_ROOT/results/calib/per_task_16x8_seed42"
mkdir -p "$FN_ROOT/results/logs" "$CALIB_DIR" "$FN_ROOT/results/drop6_extra3"

gpu_idle() {
  local gpu="$1" q mem apps
  q=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  mem=$(awk '{print int($1+0)}' <<<"$q")
  [[ "$q" =~ [0-9] && "$mem" =~ ^[0-9]+$ ]] || return 1
  apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | awk 'BEGIN{c=0} $1 ~ /^[0-9]+$/ {c++} END{print c}')
  [[ "$apps" =~ ^[0-9]+$ ]] || return 1
  [[ "$mem" -lt 1500 && "$apps" -eq 0 ]]
}

wait_gpu() {
  local gpu="$1" streak=0
  echo "waiting for GPU $gpu idle (mem<1500MiB, no compute apps, 2x10s)..."
  while true; do
    if gpu_idle "$gpu"; then
      streak=$((streak + 1))
      echo "[idle $streak/2] GPU $gpu free at $(date -Iseconds)"
      if [[ "$streak" -ge 2 ]]; then
        break
      fi
    else
      if [[ "$streak" -ne 0 ]]; then
        echo "[busy] GPU $gpu not stably free, reset idle streak"
      fi
      streak=0
      echo "[busy $(date -Iseconds)] GPU $gpu: $(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
    fi
    sleep 10
  done
}

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHDYNAMO_DISABLE=1
cd "$ROOT"

if [[ "$MODE" == "dump" ]]; then
  LOG="$FN_ROOT/results/logs/dump_calib_per_task.log"
  {
    echo "==== dump per-task calib $(date -Iseconds) ===="
    export CUDA_VISIBLE_DEVICES=""
    "$PY" dump_calib_per_task.py --out-dir "$CALIB_DIR"
    echo "==== dump done $(date -Iseconds) EXIT=$? ===="
  } 2>&1 | tee -a "$LOG"
  exit "${PIPESTATUS[0]}"
fi

GPU="${2:?gpu index}"
if [[ "$MODE" != "fisher" && "$MODE" != "greedy_ppl" ]]; then
  echo "unknown mode: $MODE" >&2
  exit 2
fi
LOG="$FN_ROOT/results/logs/drop6_extra3_${MODE}_gpu${GPU}.log"
{
  echo "==== drop6 extra-3 method=$MODE gpu=$GPU $(date -Iseconds) ===="
  wait_gpu "$GPU"
  export CUDA_VISIBLE_DEVICES="$GPU"
  "$PY" score_drop6_extra3.py --method "$MODE" --device cuda --calib-dir "$CALIB_DIR"
  echo "==== done $(date -Iseconds) EXIT=$? ===="
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
