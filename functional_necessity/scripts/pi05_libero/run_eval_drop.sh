#!/usr/bin/env bash
# Closed-loop LIBERO eval with runtime DTR-B skip (no recovery).
# Usage:
#   bash run_eval_drop.sh smoke
#   bash run_eval_drop.sh baseline
#   bash run_eval_drop.sh gateprobe 6
#   bash run_eval_drop.sh k6
#   bash run_eval_drop.sh queue baseline gateprobe:6 taylor:6
#
# queue: run several jobs on this GPU, one after another. A failed job does not
# skip the rest. Re-running skips any tag that already has eval_info.json.
# Job tokens are "baseline" or "method:k" (colon, so "gateprobe 6" is not split).
set -u
set -o pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_DIR="$ROOT/results/eval_drop"
RANKS="$ROOT/results/ranks/ranks.json"
LOG="${EVAL_DROP_LOG:-$ROOT/results/logs/eval_drop.log}"
mkdir -p "$EVAL_DIR" "$(dirname "$LOG")"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x /home/guorj/.miniforge3/envs/lerobot/bin/python ]; then
    PYTHON=/home/guorj/.miniforge3/envs/lerobot/bin/python
  else
    PYTHON=python
  fi
fi

PROTO=(
  --policy.path=lerobot/pi05_libero_finetuned_v044
  --policy.n_action_steps=10
  --policy.compile_model=false
  --policy.gradient_checkpointing=false
  --env.type=libero
  --env.task=libero_spatial,libero_object,libero_goal,libero_10
  --env.control_mode=relative
  --env.max_parallel_tasks=1
  --eval.batch_size=1
  --eval.n_episodes=10
  --seed=1000
)

K6_METHODS=(gateprobe taylor fisher igia hessian magnitude cossim cka ppl)
K3_METHODS=(gateprobe taylor fisher igia hessian magnitude cossim cka ppl)
FAILED=0

run_one() {
  local method="$1"
  local k="$2"
  local tag
  if [ "$method" = "baseline" ]; then
    tag="baseline"
  else
    tag="k${k}_${method}"
  fi
  local out="$EVAL_DIR/$tag"
  if [ -f "$out/eval_info.json" ]; then
    echo "[skip] $tag ($out/eval_info.json exists)"
    return 0
  fi
  echo "==== START $tag $(date -Iseconds) gpu=$CUDA_VISIBLE_DEVICES ===="
  local extra=(--output_dir "$out" --job_name "$tag" --from-ranks "$RANKS")
  if [ "$method" = "baseline" ]; then
    extra+=(--method baseline)
  else
    extra+=(--method "$method" --k "$k")
  fi
  if "$PYTHON" eval_drop.py "${extra[@]}" "${PROTO[@]}"; then
    echo "==== OK $tag $(date -Iseconds) ===="
    return 0
  fi
  local rc=$?
  echo "==== FAIL $tag exit=$rc $(date -Iseconds) ===="
  FAILED=1
  return 0
}

run_token() {
  local token="$1"
  if [ "$token" = "baseline" ]; then
    run_one baseline 0
    return
  fi
  if [[ "$token" == *:* ]]; then
    run_one "${token%%:*}" "${token##*:}"
    return
  fi
  echo "bad queue token: $token (use baseline or method:k, e.g. gateprobe:6)"
  FAILED=1
}

{
  echo "==== eval_drop $(date -Iseconds) gpu=$CUDA_VISIBLE_DEVICES log=$LOG ===="
  target="${1:-}"
  if [ -z "$target" ]; then
    echo "usage: $0 smoke|baseline|k3|k6|queue| <method> <k>"
    exit 2
  fi
  if [ "$target" = "smoke" ]; then
    "$PYTHON" eval_drop.py --smoke
  elif [ "$target" = "baseline" ]; then
    run_one baseline 0
  elif [ "$target" = "k3" ]; then
    for m in "${K3_METHODS[@]}"; do
      run_one "$m" 3
    done
  elif [ "$target" = "k6" ]; then
    for m in "${K6_METHODS[@]}"; do
      run_one "$m" 6
    done
  elif [ "$target" = "queue" ]; then
    shift
    if [ "$#" -eq 0 ]; then
      echo "queue needs jobs, e.g. $0 queue baseline gateprobe:6 taylor:6"
      exit 2
    fi
    for token in "$@"; do
      run_token "$token"
    done
  else
    if [ -z "${2:-}" ]; then
      echo "method jobs need k, e.g. $0 gateprobe 6"
      exit 2
    fi
    run_one "$target" "$2"
  fi
  echo "==== done $(date -Iseconds) failed=$FAILED ===="
  exit "$FAILED"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
