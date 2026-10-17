#!/usr/bin/env bash
# GPU3 serial: Molmo graph short diagnostic → formal FLOPs (π0.5 + Molmo) → Init Long if due.
set -euo pipefail
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
PHYS=3
CPUSET=0-7
STAMP=20261017
OUT_FLOPS="$ROOT/results/stage_nfe/formal_flops_${STAMP}"
OUT_GRAPH="$ROOT/results/stage_nfe/molmo_graph_diagnostic_${STAMP}"
LOG="$OUT_FLOPS/logs"
FLOCK="$ROOT/results/logs/efficiency_supplement_gpu3.flock"
MASTER="$LOG/serial.log"
export CUDA_VISIBLE_DEVICES=$PHYS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LOG" "$OUT_FLOPS/configs"

exec 9>"$FLOCK"
if ! flock -n 9; then
  echo "==== efficiency supplement already running $(date -Iseconds) ===="
  exit 0
fi

log_gpu() {
  nvidia-smi --id=$PHYS --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader | tee -a "$MASTER"
}

echo "==== EFFICIENCY SUPPLEMENT START $(date -Iseconds) gpu=$PHYS ====" | tee -a "$MASTER"
log_gpu

echo "==== graph diagnostic ====" | tee -a "$MASTER"
taskset -c "$CPUSET" "$PY" "$ROOT/scripts/stage_nfe/diagnose_molmo_cuda_graph_short.py" \
  --physical-gpu "$PHYS" --out-dir "$OUT_GRAPH" 2>&1 | tee -a "$LOG/graph_diagnostic.log"

for cfg in pi05_full_nfe10_eager pi05_pruned_nfe10_materialized pi05_full_nfe1_eager pi05_pruned_nfe1_materialized; do
  echo "==== FLOPs pi05 $cfg $(date -Iseconds) ====" | tee -a "$MASTER"
  log_gpu
  taskset -c "$CPUSET" "$PY" "$ROOT/scripts/stage_nfe/measure_formal_flops.py" \
    --stack pi05 --config-id "$cfg" --physical-gpu "$PHYS" --out-dir "$OUT_FLOPS" \
    2>&1 | tee -a "$LOG/${cfg}.log"
done
"$PY" "$ROOT/scripts/stage_nfe/measure_formal_flops.py" --stack pi05 --merge-only --physical-gpu "$PHYS" --out-dir "$OUT_FLOPS" \
  2>&1 | tee -a "$LOG/summary_pi05.log"

for cfg in molmo_full_nfe10_eager molmo_pruned_nfe10_materialized molmo_full_nfe1_eager molmo_pruned_nfe1_materialized; do
  echo "==== FLOPs molmo $cfg $(date -Iseconds) ====" | tee -a "$MASTER"
  log_gpu
  taskset -c "$CPUSET" "$PY" "$ROOT/scripts/stage_nfe/measure_formal_flops.py" \
    --stack molmoact2 --config-id "$cfg" --physical-gpu "$PHYS" --out-dir "$OUT_FLOPS" \
    2>&1 | tee -a "$LOG/${cfg}.log"
done
"$PY" "$ROOT/scripts/stage_nfe/measure_formal_flops.py" --stack molmoact2 --merge-only --physical-gpu "$PHYS" --out-dir "$OUT_FLOPS" \
  2>&1 | tee -a "$LOG/summary_molmoact2.log"

echo "==== Init Long queue check ====" | tee -a "$MASTER"
INIT_TAG=molmoact2_full_nfe10_init_10
INIT_OUT="$ROOT/results/eval_libero_plus/$INIT_TAG"
CLASS=/home/guorj/LIBERO-plus/libero/libero/benchmark/task_classification.json
CHECK="$PY $ROOT/scripts/molmoact2_libero/check_plus_eval_integrity.py --output-dir $INIT_OUT --suite libero_10 --category 'Robot Initial States' --classification-json $CLASS"
if pgrep -af "eval_plus_molmo.py.*${INIT_TAG}" >/dev/null 2>&1; then
  echo "[skip init] ${INIT_TAG} already running" | tee -a "$MASTER"
elif eval "$CHECK" >/tmp/init_check.json 2>/dev/null; then
  echo "[skip init] ${INIT_TAG} integrity ok" | tee -a "$MASTER"
  cat /tmp/init_check.json | tee -a "$MASTER"
else
  echo "[run init] ${INIT_TAG}" | tee -a "$MASTER"
  log_gpu
  bash "$ROOT/scripts/molmoact2_libero/run_eval_plus_lang.sh" "$INIT_TAG" libero_10 "Robot Initial States" 10 "$PHYS" \
    2>&1 | tee -a "$ROOT/results/logs/eval_${INIT_TAG}.log"
fi

echo "==== EFFICIENCY SUPPLEMENT DONE $(date -Iseconds) ====" | tee -a "$MASTER"
log_gpu
