#!/usr/bin/env bash
# Wait until a GPU is not owned by an FN Drop-3 queue and is idle, then eval tags.
# Usage: bash run_eval_next_free_gpu.sh <method> [method ...]
set -u
set -o pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <method> [method ...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG="$ROOT/results/logs/eval_drop3_next_free_queue.log"
mkdir -p "$(dirname "$LOG")"

queue_owns_gpu() {
  local gpu="$1"
  pgrep -f "run_eval_queue.sh ${gpu} " >/dev/null && return 0
  pgrep -f "run_eval_one.sh ${gpu} " >/dev/null && return 0
  return 1
}

gpu_idle() {
  local gpu="$1" q mem apps
  q=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  mem=$(awk '{print int($1+0)}' <<<"$q")
  [[ "$q" =~ [0-9] && "$mem" =~ ^[0-9]+$ ]] || return 1
  apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | awk 'BEGIN{c=0} $1 ~ /^[0-9]+$/ {c++} END{print c}')
  [[ "$apps" =~ ^[0-9]+$ ]] || return 1
  [[ "$mem" -lt 1500 && "$apps" -eq 0 ]]
}

{
  echo "==== next-free methods=$* $(date -Iseconds) ===="
  echo "waiting for a GPU not owned by fn-dtr-drop3 queues..."
  declare -A streak
  streak[0]=0; streak[1]=0; streak[2]=0; streak[3]=0
  GPU=""
  while true; do
    for g in 0 1 2 3; do
      if queue_owns_gpu "$g"; then
        streak[$g]=0
        continue
      fi
      if gpu_idle "$g"; then
        streak[$g]=$((streak[$g] + 1))
        echo "[idle ${streak[$g]}/4] GPU $g free of queues at $(date -Iseconds)"
        if [[ "${streak[$g]}" -ge 4 ]]; then
          GPU="$g"
          break
        fi
      else
        if [[ "${streak[$g]}" -ne 0 ]]; then
          echo "[busy] GPU $g not stably free, reset idle streak"
        fi
        streak[$g]=0
      fi
    done
    if [[ -n "$GPU" ]]; then
      break
    fi
    sleep 30
  done
  PORT=$((18100 + GPU))
  echo "using GPU $GPU port=$PORT $(date -Iseconds)"
  FAILED=0
  for tag in "$@"; do
    if bash "$ROOT/run_eval_one.sh" "$GPU" "$PORT" "$tag"; then
      echo "queue: $tag ok"
    else
      echo "queue: $tag FAILED (continue)"
      FAILED=1
    fi
  done
  echo "==== next-free done $(date -Iseconds) failed=$FAILED ===="
  exit "$FAILED"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
