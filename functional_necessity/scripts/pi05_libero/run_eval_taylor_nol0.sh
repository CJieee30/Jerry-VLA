#!/usr/bin/env bash
# π0.5 v044 Taylor Drop-k with layer 0 excluded from the redundancy order.
# Usage: bash run_eval_taylor_nol0.sh GPU K DROP_CSV
set -u
set -o pipefail
GPU="${1:?gpu}"
K="${2:?k}"
DROP="${3:?drop csv}"
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
TAG="k${K}_taylor_nol0"
OUT="$ROOT/results/eval_drop/$TAG"
LOG="$ROOT/results/logs/eval_${TAG}.log"
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
mkdir -p "$(dirname "$LOG")"
if [ -f "$OUT/eval_info.json" ]; then
  echo "[skip] $TAG exists" | tee -a "$LOG"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/guorj/lerobot/src
cd "$ROOT/scripts/pi05_libero"
echo "==== START $TAG $(date -Iseconds) gpu=$GPU drop=$DROP ====" >>"$LOG"
"$PY" eval_drop.py \
  --method taylor_nol0 --k "$K" --drop-layers "$DROP" \
  --output_dir "$OUT" --job_name "$TAG" \
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
  --seed=1000 >>"$LOG" 2>&1
RC=$?
echo "==== EXIT $TAG exit=$RC $(date -Iseconds) ====" >>"$LOG"
exit "$RC"
