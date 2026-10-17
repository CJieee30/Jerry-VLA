#!/usr/bin/env bash
# Molmo formal deployment latency (exclusive GPU). Prepared after graph diagnostic;
# uses per-obs warmup inside the timed region when graph is enabled.
set -euo pipefail
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
SCRIPT="$ROOT/scripts/stage_nfe/measure_formal_deployment_efficiency.py"
DIAG="$ROOT/scripts/stage_nfe/diagnose_molmo_cuda_graph_short.py"
PHYS="${1:?physical GPU index, e.g. 2 or 3}"
STAMP="${2:-$(date -u +%Y%m%d)}"
OUT="$ROOT/results/stage_nfe/formal_deployment_molmo_${STAMP}"
LOG="$OUT/logs"
export CUDA_VISIBLE_DEVICES="$PHYS"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUT/configs" "$LOG"

FREE_MIB=$(nvidia-smi --id="$PHYS" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if [ "${FREE_MIB:-0}" -lt 20000 ]; then
  echo "GPU $PHYS free ${FREE_MIB}MiB < 20GiB; abort (shared GPU not for formal Molmo latency)" | tee -a "$LOG/preflight.log"
  exit 1
fi

echo "==== graph diagnostic reference $(date -Iseconds) ====" | tee -a "$LOG/preflight.log"
taskset -c 0-7 "$PY" "$DIAG" --physical-gpu "$PHYS" --out-dir "$ROOT/results/stage_nfe/molmo_graph_diagnostic_${STAMP}" \
  2>&1 | tee -a "$LOG/graph_diagnostic.log" || true

run_cfg() {
  local cfg=$1
  echo "==== $cfg $(date -Iseconds) ====" | tee -a "$LOG/serial.log"
  taskset -c 0-7 "$PY" "$SCRIPT" --stack molmoact2 --config-id "$cfg" --physical-gpu "$PHYS" --out-dir "$OUT" \
    2>&1 | tee -a "$LOG/${cfg}.log"
}

for cfg in molmo_full_nfe10_graph molmo_pruned_nfe10_materialized_graph molmo_full_nfe1_graph molmo_pruned_nfe1_materialized_graph; do
  run_cfg "$cfg"
done
"$PY" "$SCRIPT" --stack molmoact2 --merge-only --physical-gpu "$PHYS" --out-dir "$OUT" \
  2>&1 | tee -a "$LOG/summary_molmo.log"
echo "==== MOLMO FORMAL DEPLOYMENT DONE $(date -Iseconds) out=$OUT ====" | tee -a "$LOG/serial.log"
