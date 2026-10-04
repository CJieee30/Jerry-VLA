#!/usr/bin/env bash
# Held-out matched/swap Δloss for pre-registered suite×skill extra-2 cores.
# Usage:
#   bash run_score_suite_skill_core2_heldout.sh <gpu> [calib_seed ...]
# Default calib seeds: 44 45. Pass a single seed to split across GPUs.
set -u
set -o pipefail

GPU="${1:?gpu index}"
shift
if [[ $# -eq 0 ]]; then
  SEEDS=(44 45)
else
  SEEDS=("$@")
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FN_ROOT="$(cd "$ROOT/../.." && pwd)"
PY="${PYTHON:-/home/guorj/.miniforge3/envs/lerobot/bin/python}"
SEED_TAG=$(IFS=-; echo "${SEEDS[*]}")
LOG="$FN_ROOT/results/logs/suite_skill_core2_heldout_s${SEED_TAG}_gpu${GPU}.log"
mkdir -p "$FN_ROOT/results/logs" "$FN_ROOT/results/drop6_extra3"

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

{
  echo "==== suite×skill core-2 held-out seeds=${SEEDS[*]} gpu=$GPU $(date -Iseconds) ===="
  wait_gpu "$GPU"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCHDYNAMO_DISABLE=1
  cd "$ROOT"
  "$PY" score_suite_skill_core2_heldout.py --device cuda --calib-seeds "${SEEDS[@]}"
  echo "==== done $(date -Iseconds) EXIT=$? ===="
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
