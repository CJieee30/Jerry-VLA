#!/usr/bin/env bash
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
LOG="$FN/results/smolvla_libero/baseline.log"
DONE="$FN/results/smolvla_libero/baseline_eval/eval_info.json"
SCRIPT="$FN/scripts/smolvla_libero/run_baseline.py"

mkdir -p "$(dirname "$LOG")"
if [ -f "$DONE" ]; then
  echo "ALREADY_DONE smolvla libero baseline $(date -Iseconds)" | tee -a "$LOG"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=device
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/home/guorj/lerobot/src:${PYTHONPATH:-}
echo "launching SmolVLA LIBERO baseline on GPU${CUDA_VISIBLE_DEVICES} $(date -Iseconds)" | tee -a "$LOG"
cd /home/guorj/lerobot
"$PY" "$SCRIPT" >>"$LOG" 2>&1
echo "EXIT:$? $(date -Iseconds)" >>"$LOG"
