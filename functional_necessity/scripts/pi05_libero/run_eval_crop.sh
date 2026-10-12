#!/usr/bin/env bash
# CROP-VLA 400-ep LIBERO eval on top of Fisher Drop-6 keep-KV + NFE=1.
# Does not rerun the protocol-matched baseline at
# results/stage_nfe/pi05_k6_fisher_nfe1/. Refuses to start unless the GPU is idle.
#
# Usage:
#   bash run_eval_crop.sh            # pick the first idle GPU
#   bash run_eval_crop.sh 2          # require GPU 2 idle
set -u
set -o pipefail

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
SCRIPT_DIR="$ROOT/scripts/pi05_libero"
OUT="$ROOT/results/stage_nfe/crop_vla/eval_crop"
LOG="$ROOT/results/logs/eval_crop_vla.log"
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
DROP="1,13,14,15,16,17"
GPU_REQ="${1:-}"

mkdir -p "$(dirname "$LOG")" "$OUT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH=/home/guorj/lerobot/src

if [ -f "$OUT/eval_info.json" ]; then
  echo "[skip] CROP eval already exists: $OUT/eval_info.json" | tee -a "$LOG"
  exit 0
fi

cd "$SCRIPT_DIR"
if ! IDLE="$("$PY" -c "
from crop_vla import first_exclusive_gpu, gpu_is_exclusive
import sys
want = sys.argv[1]
if want == '':
    gpu, note = first_exclusive_gpu()
    print(note, file=sys.stderr)
    if gpu is None:
        raise SystemExit(2)
    print(gpu)
else:
    gpu = int(want)
    ok, why = gpu_is_exclusive(gpu)
    print(why, file=sys.stderr)
    if not ok:
        raise SystemExit(2)
    print(gpu)
" "$GPU_REQ")"; then
  echo "==== WAIT GPU not idle $(date -Iseconds) ====" | tee -a "$LOG"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$IDLE"
echo "==== OFFLINE CHECK crop_vla gpu=$IDLE $(date -Iseconds) ====" | tee -a "$LOG"
if ! "$PY" check_crop_vla.py --gpu "$IDLE" --n-frames 4 2>&1 | tee -a "$LOG"; then
  echo "==== FAIL offline check; skip 400-ep eval $(date -Iseconds) ====" | tee -a "$LOG"
  exit 1
fi

echo "==== START crop_vla gpu=$IDLE $(date -Iseconds) ====" | tee -a "$LOG"
"$PY" eval_drop.py \
  --method fisher --k 6 --drop-layers "$DROP" --crop \
  --output_dir "$OUT" --job_name crop_vla \
  --policy.path=lerobot/pi05_libero_finetuned_v044 \
  --policy.n_action_steps=10 \
  --policy.num_inference_steps=1 \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --seed=1000 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "==== EXIT crop_vla exit=$RC $(date -Iseconds) ====" | tee -a "$LOG"
exit "$RC"
