#!/usr/bin/env bash
# Wait until this GPU is idle, then eval FN Drop-3 methods sequentially.
# Usage: bash run_eval_queue.sh <gpu> <method> [method ...]
# Example: bash run_eval_queue.sh 0 fisher_ppl gateprobe
set -u
set -o pipefail

GPU="${1:?gpu index 0-3}"
shift
if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <gpu> <method> [method ...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ "${1:-}" =~ ^l[0-9]+$ ]]; then
  LOG="$ROOT/results/logs/eval_drop1_gpu${GPU}_queue.log"
else
  LOG="$ROOT/results/logs/eval_drop3_gpu${GPU}_queue.log"
fi
PORT=$((18100 + GPU))
mkdir -p "$(dirname "$LOG")"

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
  echo "==== queue gpu=$GPU port=$PORT methods=$* $(date -Iseconds) ===="
  echo "waiting for GPU $GPU to be idle (mem<1500MiB and no compute apps, 4x30s)..."
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
      echo "[busy $(date -Iseconds)] GPU $GPU mem/util: $(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
    fi
    sleep 30
  done
  echo "GPU $GPU idle. starting eval $(date -Iseconds)"

  FAILED=0
  for tag in "$@"; do
    if bash "$ROOT/run_eval_one.sh" "$GPU" "$PORT" "$tag"; then
      echo "queue: $tag ok"
    else
      echo "queue: $tag FAILED (continue)"
      FAILED=1
    fi
  done
  echo "==== queue gpu=$GPU done $(date -Iseconds) failed=$FAILED ===="
  exit "$FAILED"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
