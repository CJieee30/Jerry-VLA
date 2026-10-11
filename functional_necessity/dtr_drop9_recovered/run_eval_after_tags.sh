#!/usr/bin/env bash
# Wait until two Drop-1 tags on this GPU are fully written and the current
# queue has exited, then evaluate more tags on the same GPU.
# Usage: bash run_eval_after_tags.sh <gpu> <wait_tag1> <wait_tag2> <next_tag>
set -u
set -o pipefail

GPU="${1:?gpu}"
WAIT1="${2:?wait tag 1}"
WAIT2="${3:?wait tag 2}"
NEXT="${4:?next tag}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG="$ROOT/results/logs/eval_drop1_gpu${GPU}_after.log"
mkdir -p "$(dirname "$LOG")"

tag_complete() {
  local tag="$1" info
  info="$ROOT/results/eval_drop1/$tag/eval_info.json"
  [[ -f "$info" ]] || return 1
  "$ROOT/.venv/bin/python" - "$info" <<'PY'
import json, sys
from pathlib import Path
ov = json.loads(Path(sys.argv[1]).read_text()).get("overall") or {}
sys.exit(0 if ov.get("n_suites") == 4 and ov.get("pc_success_mean") is not None else 1)
PY
}

queue_still_running() {
  pgrep -f "run_eval_queue.sh ${GPU} ${WAIT1} " >/dev/null && return 0
  pgrep -f "run_eval_one.sh ${GPU} " >/dev/null && return 0
  return 1
}

{
  echo "==== after-tags gpu=$GPU wait=$WAIT1,$WAIT2 then=$NEXT $(date -Iseconds) ===="
  while true; do
    if tag_complete "$WAIT1" && tag_complete "$WAIT2" && ! queue_still_running; then
      echo "prereqs done at $(date -Iseconds); starting $NEXT"
      break
    fi
    echo "[wait $(date -Iseconds)] $WAIT1=$(tag_complete "$WAIT1" && echo done || echo pending) $WAIT2=$(tag_complete "$WAIT2" && echo done || echo pending) queue=$(queue_still_running && echo running || echo idle)"
    sleep 30
  done
} 2>&1 | tee -a "$LOG"

exec bash "$ROOT/run_eval_queue.sh" "$GPU" "$NEXT"
