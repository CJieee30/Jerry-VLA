#!/usr/bin/env bash
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
GPU="${1:-3}"
LOG="$FN/results/smolvla_libero/score_g${GPU}.log"

mkdir -p "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/home/guorj/lerobot/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "launching SmolVLA 9-method score on GPU${GPU} $(date -Iseconds)" | tee -a "$LOG"
cd "$FN/scripts/smolvla_libero"
"$PY" score.py >>"$LOG" 2>&1
echo "EXIT:$? $(date -Iseconds)" >>"$LOG"
