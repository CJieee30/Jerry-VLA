#!/usr/bin/env bash
# Task-aware extra Drop-3 closed-loop on v044 (U ∪ extra3(task)).
# Usage:
#   bash run_eval_task_extra.sh <gpu> <map.json> <tag>
# Example:
#   bash run_eval_task_extra.sh 0 ../results/drop6_extra3/maps/map_fisher_extra3.json fisher_extra3
set -u
set -o pipefail

GPU="${1:?gpu index}"
MAP="${2:?task extra map json}"
TAG="${3:?output tag}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FN_ROOT="$(cd "$ROOT/../.." && pwd)"
MAP="$(readlink -f "$MAP")"
OUT="$FN_ROOT/results/eval_task_extra/$TAG"
LOG="$FN_ROOT/results/logs/eval_task_extra_${TAG}_gpu${GPU}.log"
PY="${PYTHON:-/home/guorj/.miniforge3/envs/lerobot/bin/python}"

mkdir -p "$OUT" "$(dirname "$LOG")"

gpu_idle() {
  local q mem apps
  q=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  mem=$(awk '{print int($1+0)}' <<<"$q")
  [[ "$q" =~ [0-9] && "$mem" =~ ^[0-9]+$ ]] || return 1
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | awk 'BEGIN{c=0} $1 ~ /^[0-9]+$/ {c++} END{print c}')
  [[ "$apps" =~ ^[0-9]+$ ]] || return 1
  [[ "$mem" -lt 1500 && "$apps" -eq 0 ]]
}

wait_gpu() {
  local streak=0
  echo "waiting for GPU $GPU idle (mem<1500MiB, no compute apps, 2x10s)..."
  while true; do
    if gpu_idle; then
      streak=$((streak + 1))
      echo "[idle $streak/2] GPU $GPU free at $(date -Iseconds)"
      if [[ "$streak" -ge 2 ]]; then
        break
      fi
    else
      if [[ "$streak" -ne 0 ]]; then
        echo "[busy] GPU $GPU not stably free, reset idle streak"
      fi
      streak=0
      echo "[busy $(date -Iseconds)] GPU $GPU: $(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
    fi
    sleep 10
  done
}

{
  echo "==== eval_task_extra tag=$TAG gpu=$GPU map=$MAP $(date -Iseconds) ===="
  if [ -f "$OUT/eval_info.json" ]; then
    echo "[skip] $OUT/eval_info.json exists"
    exit 0
  fi
  wait_gpu
  export CUDA_VISIBLE_DEVICES="$GPU"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export PYTHONUNBUFFERED=1
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCHDYNAMO_DISABLE=1
  cd "$ROOT"
  "$PY" eval_drop.py \
    --task-extra-map "$MAP" \
    --method "$TAG" \
    --k 9 \
    --output_dir "$OUT" \
    --job_name "$TAG" \
    --policy.path=lerobot/pi05_libero_finetuned_v044 \
    --policy.n_action_steps=10 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=false \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --env.control_mode=relative \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --seed=1000
  echo "==== done $(date -Iseconds) EXIT=$? ===="
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
