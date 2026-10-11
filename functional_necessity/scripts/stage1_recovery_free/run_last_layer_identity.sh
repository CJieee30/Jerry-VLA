#!/usr/bin/env bash
# Open-loop last-VLM-layer keep-KV identity. Not a LIBERO eval.
# Usage: bash run_last_layer_identity.sh GPU
set -u
set -o pipefail
GPU="${1:?gpu}"
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
LOG="$ROOT/results/logs/last_layer_identity.log"
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
mkdir -p "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/guorj/lerobot/src
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
cd "$ROOT"
echo "==== START last_layer_identity $(date -Iseconds) gpu=$GPU ====" | tee -a "$LOG"
"$PY" scripts/stage1_recovery_free/probe_last_layer_identity.py >>"$LOG" 2>&1
RC=$?
echo "==== EXIT last_layer_identity exit=$RC $(date -Iseconds) ====" | tee -a "$LOG"
exit "$RC"
