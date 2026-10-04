#!/usr/bin/env bash
# Skill-pooled independent Drop-9 ranks (full model, no skip(U)).
# Usage:
#   bash run_skill_drop9.sh <method> <gpu>
# method: fisher | gateprobe | igia
set -u
set -o pipefail

METHOD="${1:?fisher|gateprobe|igia}"
GPU="${2:?gpu index}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FN_ROOT="$(cd "$ROOT/../.." && pwd)"
PY="${PYTHON:-/home/guorj/.miniforge3/envs/lerobot/bin/python}"
LOG="$FN_ROOT/results/logs/skill_drop9_${METHOD}_gpu${GPU}.log"
mkdir -p "$FN_ROOT/results/logs" "$FN_ROOT/results/drop9_skill" "$FN_ROOT/results/scores/skill_drop9"

if [[ "$METHOD" != "fisher" && "$METHOD" != "gateprobe" && "$METHOD" != "igia" ]]; then
  echo "unknown method: $METHOD" >&2
  exit 2
fi

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
  echo "==== skill drop9 method=$METHOD gpu=$GPU $(date -Iseconds) ===="
  wait_gpu "$GPU"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCHDYNAMO_DISABLE=1
  cd "$ROOT"
  "$PY" score_skill_drop9.py --method "$METHOD" --device cuda --calib-seed 42 --microbatch-size 2
  echo "==== done $(date -Iseconds) EXIT=$? ===="
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
