#!/usr/bin/env bash
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
GPU="${1:?gpu index}"
TAG="${2:?job tag}"
SCRIPT="$FN/scripts/molmoact2_libero/${3:?python script}"
LOG="$FN/results/molmoact2_libero/${TAG}.log"
DONE="$FN/results/molmoact2_libero/${TAG}/eval_info.json"

mkdir -p "$(dirname "$LOG")"
if [ -f "$DONE" ]; then
  echo "ALREADY_DONE $TAG $(date -Iseconds)" | tee -a "$LOG"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/home/guorj/lerobot/src
echo "launching MolmoAct2 $TAG on GPU${GPU} $(date -Iseconds)" | tee -a "$LOG"
cd /home/guorj/lerobot
"$PY" "$SCRIPT" >>"$LOG" 2>&1
echo "EXIT:$? $(date -Iseconds)" >>"$LOG"
