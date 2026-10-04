#!/usr/bin/env bash
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
LOG="$FN/results/molmoact2_libero/drop8_magnitude.log"
WAIT="$FN/scripts/wait_gpu_idle.sh"
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
SCRIPT="$FN/scripts/molmoact2_libero/run_drop8_magnitude.py"

mkdir -p "$(dirname "$LOG")"
chmod +x "$WAIT"
bash "$WAIT" 2 800 30
if [ -f "$FN/results/molmoact2_libero/drop8_magnitude/eval_info.json" ] \
   || [ -f "$FN/results/molmoact2_libero/drop8_magnitude/results.json" ]; then
  echo "ALREADY_DONE magnitude drop-8 $(date -Iseconds)" | tee -a "$LOG"
  exit 0
fi
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/home/guorj/lerobot/src
echo "launching MolmoAct2 Magnitude Drop-8 on GPU2 $(date -Iseconds)" | tee -a "$LOG"
cd /home/guorj/lerobot
"$PY" "$SCRIPT" >>"$LOG" 2>&1
echo "EXIT:$? $(date -Iseconds)" >>"$LOG"
