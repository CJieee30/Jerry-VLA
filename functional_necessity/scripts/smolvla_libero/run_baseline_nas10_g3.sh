#!/usr/bin/env bash
# Co-locate n_action_steps=10 on GPU3 with the nas=1 baseline (~2 GiB each).
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
LOG="$FN/results/smolvla_libero/baseline_nas10.log"
DONE="$FN/results/smolvla_libero/baseline_eval_nas10/eval_info.json"
SCRIPT="$FN/scripts/smolvla_libero/run_baseline_nas10.py"

mkdir -p "$(dirname "$LOG")"
if [ -f "$DONE" ]; then
  echo "ALREADY_DONE smolvla libero baseline nas10 $(date -Iseconds)" | tee -a "$LOG"
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
echo "launching SmolVLA LIBERO baseline nas10 colocated on GPU${CUDA_VISIBLE_DEVICES} $(date -Iseconds)" | tee -a "$LOG"
cd /home/guorj/lerobot
"$PY" "$SCRIPT" >>"$LOG" 2>&1
echo "EXIT:$? $(date -Iseconds)" >>"$LOG"
