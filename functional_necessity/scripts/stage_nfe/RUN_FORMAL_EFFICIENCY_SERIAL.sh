#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
LOG="$ROOT/results/stage_nfe/logs_formal_efficiency"
OUT="$ROOT/results/stage_nfe"
SCRIPT="$ROOT/scripts/stage_nfe/measure_formal_efficiency.py"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LOG"

echo "==== pi05 GPU0 $(date -Iseconds) ====" | tee -a "$LOG/serial.log"
export CUDA_VISIBLE_DEVICES=0
taskset -c 8-15 "$PY" "$SCRIPT" --stack pi05 --device cuda:0 \
  --out "$OUT/formal_efficiency_pi05_20261017.json" 2>&1 | tee -a "$LOG/formal_pi05.log"

echo "==== molmo GPU2 $(date -Iseconds) ====" | tee -a "$LOG/serial.log"
export CUDA_VISIBLE_DEVICES=2
taskset -c 16-23 "$PY" "$SCRIPT" --stack molmoact2 --device cuda:0 \
  --out "$OUT/formal_efficiency_molmoact2_20261017.json" 2>&1 | tee -a "$LOG/formal_molmo.log"

echo "==== DONE $(date -Iseconds) ====" | tee -a "$LOG/serial.log"
