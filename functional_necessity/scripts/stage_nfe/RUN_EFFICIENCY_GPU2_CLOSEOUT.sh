#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
PHYS=2
CPUSET=24-31
export CUDA_VISIBLE_DEVICES=$PHYS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
FLOPS_REF="$ROOT/results/stage_nfe/formal_flops_20261017"
FLOPS_LOG="$FLOPS_REF/logs/verify_global_ops_gpu2.log"
PI05_OUT="$ROOT/results/stage_nfe/formal_deployment_pi05_steady_20261017"
PI05_LOG="$PI05_OUT/logs"
mkdir -p "$PI05_LOG" "$PI05_OUT/configs"

echo "==== FLOPs Global-op verify $(date -Iseconds) ====" | tee "$FLOPS_LOG"
for cfg in pi05_full_nfe10_eager molmo_full_nfe10_eager; do
  stack=pi05; [[ "$cfg" == molmo* ]] && stack=molmoact2
  ref_med=$(python3 -c "import json; d=json.load(open('$FLOPS_REF/configs/$cfg.json')); print(d['summary_gflops']['median_gflops'])")
  taskset -c "$CPUSET" "$PY" "$ROOT/scripts/stage_nfe/measure_formal_flops.py" \
    --stack "$stack" --config-id "$cfg" --physical-gpu "$PHYS" --n-obs 2 \
    --out-dir /tmp/flops_verify_$$ 2>&1 | tee -a "$FLOPS_LOG"
  new_med=$(python3 -c "import json; d=json.load(open('/tmp/flops_verify_$$/configs/$cfg.json')); print(d['summary_gflops']['median_gflops'])")
  frac=$(python3 -c "print(abs($new_med-$ref_med)/$ref_med*100)")
  echo "$cfg ref=$ref_med verify=$new_med drift_pct=$frac tracked=$(python3 -c "import json; d=json.load(open('/tmp/flops_verify_$$/configs/$cfg.json')); print(d['per_observation'][0]['flop_detail']['tracked_fraction_of_total'])")" | tee -a "$FLOPS_LOG"
  rm -rf /tmp/flops_verify_$$
done

echo "==== pi05 steady deploy $(date -Iseconds) ====" | tee -a "$PI05_LOG/serial.log"
for cfg in pi05_full_nfe10_eager pi05_pruned_nfe10_materialized pi05_full_nfe1_eager pi05_pruned_nfe1_materialized; do
  taskset -c "$CPUSET" "$PY" "$ROOT/scripts/stage_nfe/measure_pi05_formal_deployment_steady.py" \
    --config-id "$cfg" --physical-gpu "$PHYS" --out-dir "$PI05_OUT" 2>&1 | tee -a "$PI05_LOG/${cfg}.log"
done
taskset -c "$CPUSET" "$PY" "$ROOT/scripts/stage_nfe/measure_pi05_formal_deployment_steady.py" \
  --anchor-recheck --physical-gpu "$PHYS" --out-dir "$PI05_OUT" 2>&1 | tee -a "$PI05_LOG/anchor_recheck.log"
"$PY" "$ROOT/scripts/stage_nfe/measure_pi05_formal_deployment_steady.py" \
  --merge-only --physical-gpu "$PHYS" --out-dir "$PI05_OUT" 2>&1 | tee -a "$PI05_LOG/summary.log"

echo "==== DONE pi05; chain Molmo Plus NFE1 on GPU2 ====" | tee -a "$PI05_LOG/serial.log"
bash "$ROOT/scripts/molmoact2_libero/run_eval_plus_lang_drop16_nfe1_gpu2.sh"
