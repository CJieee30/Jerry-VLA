#!/usr/bin/env bash
# Usage: bash run_drop_named.sh GPU TAG K METHOD DROP_CSV
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
GPU="${1:?gpu index}"
TAG="${2:?job tag}"
K="${3:?k}"
METHOD="${4:?method}"
DROP="${5:?drop csv}"
LOG="$FN/results/molmoact2_libero/${TAG}.log"
DONE="$FN/results/molmoact2_libero/${TAG}/eval_info.json"

mkdir -p "$(dirname "$LOG")"
if [ -f "$DONE" ]; then
  echo "ALREADY_DONE $TAG $(date -Iseconds)" | tee -a "$LOG"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export MOLMO_DROP_TAG="$TAG"
export MOLMO_DROP_K="$K"
export MOLMO_DROP_METHOD="$METHOD"
export MOLMO_DROP="$DROP"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/home/guorj/lerobot/src
echo "launching MolmoAct2 $TAG on GPU${GPU} drop=$DROP $(date -Iseconds)" | tee -a "$LOG"
cd /home/guorj/lerobot
set +e
"$PY" "$FN/scripts/molmoact2_libero/run_drop_named.py" >>"$LOG" 2>&1
RC=$?
echo "EXIT:$RC $(date -Iseconds)" >>"$LOG"
exit "$RC"
