#!/usr/bin/env bash
# Wait until GPU $1 is idle, then run v044 greedy PPL Drop-9 search.
# Usage: bash run_v044_greedy_ppl_k9.sh <gpu>
set -u
set -o pipefail

GPU="${1:?gpu index}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FN_ROOT="$(cd "$ROOT/../.." && pwd)"
LOG="$FN_ROOT/results/logs/v044_greedy_ppl_k9_gpu${GPU}.log"
PY="${PYTHON:-/home/guorj/.miniforge3/envs/lerobot/bin/python}"

mkdir -p "$(dirname "$LOG")" "$FN_ROOT/results/ppl_subset_search"

gpu_idle() {
  local q mem apps
  q=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  mem=$(awk '{print int($1+0)}' <<<"$q")
  [[ "$q" =~ [0-9] && "$mem" =~ ^[0-9]+$ ]] || return 1
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | awk 'BEGIN{c=0} $1 ~ /^[0-9]+$/ {c++} END{print c}')
  [[ "$apps" =~ ^[0-9]+$ ]] || return 1
  [[ "$mem" -lt 1500 && "$apps" -eq 0 ]]
}

{
  echo "==== v044 greedy PPL Drop-9 gpu=$GPU $(date -Iseconds) ===="
  echo "waiting for GPU $GPU idle (mem<1500MiB, no compute apps, 4x30s)..."
  streak=0
  while true; do
    if gpu_idle; then
      streak=$((streak + 1))
      echo "[idle $streak/4] GPU $GPU free at $(date -Iseconds)"
      if [[ "$streak" -ge 4 ]]; then
        break
      fi
    else
      if [[ "$streak" -ne 0 ]]; then
        echo "[busy] GPU $GPU not stably free, reset idle streak"
      fi
      streak=0
      echo "[busy $(date -Iseconds)] GPU $GPU: $(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
    fi
    sleep 30
  done
  echo "GPU $GPU idle. starting search $(date -Iseconds)"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCHDYNAMO_DISABLE=1
  cd "$ROOT"
  "$PY" search_ppl_drop9.py --device cuda
  echo "==== done $(date -Iseconds) EXIT=$? ===="
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
