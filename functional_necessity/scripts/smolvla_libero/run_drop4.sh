#!/usr/bin/env bash
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
GPU="${1:?gpu index}"
METHOD="${2:?method}"
K="${3:-4}"
TAG="drop${K}_${METHOD}"
LOG="$FN/results/smolvla_libero/${TAG}.log"
DONE="$FN/results/smolvla_libero/${TAG}/eval_info.json"
SCRIPT="$FN/scripts/smolvla_libero/run_drop4.py"

mkdir -p "$(dirname "$LOG")"
if [ -f "$DONE" ]; then
  echo "ALREADY_DONE $TAG $(date -Iseconds)" | tee -a "$LOG"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export SMOLVLA_DROP_METHOD="$METHOD"
export SMOLVLA_DROP_K="$K"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=device
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/home/guorj/lerobot/src
echo "launching SmolVLA $TAG on GPU${GPU} $(date -Iseconds)" | tee -a "$LOG"
cd /home/guorj/lerobot
set +e
"$PY" "$SCRIPT" >>"$LOG" 2>&1
RC=$?
echo "EXIT:$RC $(date -Iseconds)" >>"$LOG"
exit "$RC"
