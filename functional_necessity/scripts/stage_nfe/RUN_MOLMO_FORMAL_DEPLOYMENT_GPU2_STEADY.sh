#!/usr/bin/env bash
# Molmo-only formal deployment latency on physical GPU2 (steady per-obs Graph protocol).
set -euo pipefail
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
SCRIPT="$ROOT/scripts/stage_nfe/measure_molmo_formal_deployment_steady.py"
PHYS=2
CPUSET=24-31
STAMP=20261017
OUT="$ROOT/results/stage_nfe/formal_deployment_molmo_steady_${STAMP}"
LOG="$OUT/logs"
DIAG="$ROOT/results/stage_nfe/molmo_graph_diagnostic_${STAMP}/diagnostic.json"
FLOCK="$ROOT/results/logs/molmo_formal_deploy_gpu2.flock"
export CUDA_VISIBLE_DEVICES=$PHYS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LOG" "$OUT/configs"

exec 8>"$FLOCK"
if ! flock -n 8; then
  echo "==== Molmo GPU2 formal deploy already running $(date -Iseconds) ====" | tee -a "$LOG/serial.log"
  exit 0
fi

FREE_MIB=$(nvidia-smi --id=$PHYS --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
OTHERS=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_gpu_memory --format=csv,noheader | awk -F', ' -v bus="$(nvidia-smi --id=$PHYS --query-gpu=pci.bus_id --format=csv,noheader | tr -d ' ')" '$1==bus' || true)
echo "==== MOLMO STEADY DEPLOY START $(date -Iseconds) gpu=$PHYS free=${FREE_MIB}MiB ====" | tee -a "$LOG/serial.log"
echo "loadavg: $(cat /proc/loadavg)" | tee -a "$LOG/serial.log"
if [ -n "$OTHERS" ]; then
  echo "WARN gpu$PHYS compute apps: $OTHERS" | tee -a "$LOG/serial.log"
fi
if [ "${FREE_MIB:-0}" -lt 18000 ]; then
  echo "ABORT: need ~18GiB free on GPU$PHYS" | tee -a "$LOG/serial.log"
  exit 1
fi

run_cfg() {
  local cfg=$1
  echo "==== $cfg $(date -Iseconds) ====" | tee -a "$LOG/serial.log"
  taskset -c "$CPUSET" "$PY" "$SCRIPT" --config-id "$cfg" --physical-gpu "$PHYS" --out-dir "$OUT" \
    --graph-diagnostic "$DIAG" 2>&1 | tee -a "$LOG/${cfg}.log"
}

for cfg in molmo_full_nfe10_graph molmo_pruned_nfe10_materialized_graph molmo_full_nfe1_graph molmo_pruned_nfe1_materialized_graph; do
  run_cfg "$cfg"
done

echo "==== anchor recheck Full10 $(date -Iseconds) ====" | tee -a "$LOG/serial.log"
taskset -c "$CPUSET" "$PY" "$SCRIPT" --anchor-recheck --physical-gpu "$PHYS" --out-dir "$OUT" \
  --graph-diagnostic "$DIAG" 2>&1 | tee -a "$LOG/anchor_recheck.log"

taskset -c "$CPUSET" "$PY" "$SCRIPT" --merge-only --physical-gpu "$PHYS" --out-dir "$OUT" \
  2>&1 | tee -a "$LOG/summary.log"

echo "==== MOLMO STEADY DEPLOY DONE $(date -Iseconds) out=$OUT ====" | tee -a "$LOG/serial.log"
