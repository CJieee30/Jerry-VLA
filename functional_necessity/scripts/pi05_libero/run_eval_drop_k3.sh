#!/usr/bin/env bash
# Unique Drop-3 closed-loop eval on one GPU (v044 / seed=1000 / compile=false).
# Usage: bash run_eval_drop_k3.sh <gpu> <method:3> [method:3 ...]
# Waits until that GPU is idle (mem<1500MiB, no compute apps) so we do not steal.
set -u
set -o pipefail

GPU="${1:?gpu index}"
shift
if [ "$#" -eq 0 ]; then
  echo "usage: $0 <gpu> method:3 [method:3 ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${PYTHON:-/home/guorj/.miniforge3/envs/lerobot/bin/python}"
LOG="$FN_ROOT/results/logs/eval_drop_k3_gpu${GPU}.log"
mkdir -p "$FN_ROOT/results/logs"

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
  echo "==== eval_drop k3 gpu=$GPU jobs=$* $(date -Iseconds) ===="
  wait_gpu "$GPU"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export EVAL_DROP_LOG="/dev/null"
  export PYTHON="$PY"
  bash "$SCRIPT_DIR/run_eval_drop.sh" queue "$@"
  rc=$?
  "$PY" "$SCRIPT_DIR/replicate_eval_drop_aliases.py" || true
  echo "==== done $(date -Iseconds) EXIT=$rc ===="
  exit "$rc"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
