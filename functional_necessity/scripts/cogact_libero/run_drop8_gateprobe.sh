#!/usr/bin/env bash
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
LOG="$FN/results/cogact_libero/drop8_gateprobe.log"
DONE="$FN/results/cogact_libero/drop8_gateprobe/eval_info.json"
SCRIPT="$FN/scripts/cogact_libero/eval_drop8_gateprobe.py"

mkdir -p "$(dirname "$LOG")"
if [ -f "$DONE" ]; then
  echo "ALREADY_DONE cogact gateprobe drop-8 $(date -Iseconds)" | tee -a "$LOG"
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
export PYTHONPATH=/home/guorj/dexbotic:${PYTHONPATH:-}
echo "launching DB-CogACT GateProbe Drop-8 on GPU${CUDA_VISIBLE_DEVICES} $(date -Iseconds)" | tee -a "$LOG"
cd /home/guorj/dexbotic
"$PY" "$SCRIPT" >>"$LOG" 2>&1
echo "EXIT:$? $(date -Iseconds)" >>"$LOG"
