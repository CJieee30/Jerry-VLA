#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
SCRIPT="$ROOT/scripts/stage_nfe/measure_formal_deployment_efficiency.py"
OUT="$ROOT/results/stage_nfe/formal_deployment_20261017"
LOG="$OUT/logs"
PHYS=2
export CUDA_VISIBLE_DEVICES=$PHYS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUT/configs" "$LOG"

run_cfg() {
  local stack=$1 cfg=$2
  echo "==== $cfg $(date -Iseconds) ====" | tee -a "$LOG/serial.log"
  taskset -c 24-31 "$PY" "$SCRIPT" --stack "$stack" --config-id "$cfg" --physical-gpu "$PHYS" --out-dir "$OUT" \
    2>&1 | tee -a "$LOG/${cfg}.log"
}

for cfg in pi05_full_nfe10_eager pi05_pruned_nfe10_materialized pi05_full_nfe1_eager pi05_pruned_nfe1_materialized; do
  run_cfg pi05 "$cfg"
done
"$PY" "$SCRIPT" --stack pi05 --merge-only --physical-gpu "$PHYS" --out-dir "$OUT" 2>&1 | tee -a "$LOG/summary_pi05.log"

for cfg in molmo_full_nfe10_graph molmo_pruned_nfe10_materialized_graph molmo_full_nfe1_graph molmo_pruned_nfe1_materialized_graph; do
  run_cfg molmoact2 "$cfg"
done
"$PY" "$SCRIPT" --stack molmoact2 --merge-only --physical-gpu "$PHYS" --out-dir "$OUT" 2>&1 | tee -a "$LOG/summary_molmo.log"

echo "==== ALL DONE $(date -Iseconds) ====" | tee -a "$LOG/serial.log"
