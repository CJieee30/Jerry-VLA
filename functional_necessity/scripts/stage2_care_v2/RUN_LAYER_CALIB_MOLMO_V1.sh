#!/usr/bin/env bash
# Molmo prospective 400-obs layer calibration (rev2). Fails fast on errors.
set -euo pipefail

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/stage2_care_v2
OUT=/home/guorj/Skill_Efficient_VLA/functional_necessity/results/stage2_care_v2/layer_calibration_molmo_v1
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
GPU="${1:-}"
MODE="${2:-all}"

mkdir -p "$OUT"
echo "==== GPU snapshot $(date -Iseconds) ====" | tee -a "$OUT/gpu_check.log"
nvidia-smi | tee -a "$OUT/gpu_check.log"

cd "$ROOT"

if [[ "$MODE" == "cpu" || "$MODE" == "all" ]]; then
  echo "==== CPU selftest ====" | tee -a "$OUT/run.log"
  "$PY" selftest_layer_calibration_molmo_v1.py 2>&1 | tee -a "$OUT/selftest.log"

  echo "==== CPU: dump calib (skip if exists) ====" | tee -a "$OUT/run.log"
  "$PY" dump_calib_molmo_prospective.py 2>&1 | tee -a "$OUT/run.log"
fi

if [[ -z "$GPU" ]]; then
  echo "No GPU id; scoring skipped. Example:" | tee -a "$OUT/run.log"
  echo "  CUDA_VISIBLE_DEVICES=1 bash RUN_LAYER_CALIB_MOLMO_V1.sh 1 gpu" | tee -a "$OUT/run.log"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU"

if [[ "$MODE" == "smoke" || "$MODE" == "gpu" || "$MODE" == "all" ]]; then
  echo "==== GPU verify smoke (n=2) ====" | tee -a "$OUT/run.log"
  "$PY" verify_molmo_score_smoke.py --device cuda:0 2>&1 | tee -a "$OUT/score_smoke.log" || {
    echo "SMOKE FAILED — not running full score" | tee -a "$OUT/run.log"
    exit 1
  }
fi

if [[ "$MODE" == "gpu" || "$MODE" == "all" ]]; then
  echo "==== GPU frame scores (IGIA/Fisher/GateProbe) ====" | tee -a "$OUT/run.log"
  "$PY" score_molmo_prospective_layers.py --device cuda:0 2>&1 | tee -a "$OUT/score_run.log"

  echo "==== GPU Taylor groups (b=80, four strategies) ====" | tee -a "$OUT/run.log"
  "$PY" score_molmo_prospective_taylor_groups.py --device cuda:0 2>&1 | tee -a "$OUT/taylor_group_run.log"
fi

echo "==== CPU analyze ====" | tee -a "$OUT/run.log"
"$PY" analyze_layer_calibration_molmo_v1.py 2>&1 | tee -a "$OUT/run.log"

echo "==== done $(date -Iseconds) ====" | tee -a "$OUT/run.log"
