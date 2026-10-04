#!/usr/bin/env bash
# Score assigned methods on one GPU, then run this GPU's Drop-8 suites.
# Usage: bash run_score.sh <gpu> <method> [method ...]
set -u
set -o pipefail

GPU="${1:?gpu index}"
shift
if [ "$#" -eq 0 ]; then
  echo "usage: $0 <gpu> magnitude|taylor|... " >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=/home/guorj/.miniforge3/envs/openvla-oft/bin/python
DATA="$FN_ROOT/datasets/modified_libero_rlds"
RESULT="$FN_ROOT/results/openvla_oft_libero/drop8"
SCORE_DIR="$RESULT/scores"
mkdir -p "$RESULT/calib" "$SCORE_DIR" "$RESULT/logs"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OPENVLA_SAVE_ROLLOUT_VIDEO=0
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/home/guorj/openvla-oft/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="/home/guorj/openvla-oft/libero_config"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$SCRIPT_DIR"

gpu_idle() {
  local gpu="$1" q mem
  q=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  mem=$(awk '{print int($1+0)}' <<<"$q")
  [[ "$q" =~ [0-9] && "$mem" =~ ^[0-9]+$ ]] || return 1
  # Share the card while the other job stays under 9GB (~15GB left).
  [[ "$mem" -lt 9000 ]]
}

wait_gpu() {
  local gpu="$1" streak=0
  echo "waiting for GPU $gpu to have room (mem<9000MiB, 2x10s; sharing is ok)..."
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

dump_calib() {
  local rc
  exec 9>"$RESULT/calib/dump.lock"
  flock 9
  while true; do
    "$PY" dump_calib.py
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      break
    fi
    if [[ "$rc" -eq 75 ]]; then
      echo "RLDS incomplete, retry in 60s $(date -Iseconds)"
      sleep 60
      continue
    fi
    echo "dump_calib failed exit=$rc"
    exit "$rc"
  done
  flock -u 9
}

smoke() {
  local method="$1"
  if [[ -f "$SCORE_DIR/${method}.pt" ]]; then
    echo "skip smoke $method (score exists)"
    return 0
  fi
  echo "==== smoke $method $(date -Iseconds) ===="
  "$PY" score.py --method "$method" --limit 1 || exit $?
}

echo "==== oft drop8 gpu=$GPU methods=$* $(date -Iseconds) ===="
dump_calib
wait_gpu "$GPU"

if [[ "$GPU" == "2" ]]; then
  smoke cossim
fi
if [[ "$GPU" == "3" ]]; then
  smoke gateprobe
  smoke fisher
fi

for method in "$@"; do
  echo "==== score $method $(date -Iseconds) gpu=$GPU ===="
  "$PY" score.py --method "$method" || exit $?
done

echo "==== join eval gpu=$GPU $(date -Iseconds) ===="
wait_gpu "$GPU"
"$PY" join_eval.py "$GPU"
echo "==== done gpu=$GPU $(date -Iseconds) ===="
