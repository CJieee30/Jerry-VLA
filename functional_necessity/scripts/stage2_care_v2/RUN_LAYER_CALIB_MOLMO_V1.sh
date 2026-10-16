#!/usr/bin/env bash
# Molmo prospective 400-obs layer calibration (IGIA + 3 metrics). Check GPU first.
set -u
set -o pipefail

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/stage2_care_v2
OUT=/home/guorj/Skill_Efficient_VLA/functional_necessity/results/stage2_care_v2/layer_calibration_molmo_v1
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
GPU="${1:-}"

mkdir -p "$OUT"
echo "==== GPU snapshot $(date -Iseconds) ====" | tee -a "$OUT/gpu_check.log"
nvidia-smi | tee -a "$OUT/gpu_check.log"

cd "$ROOT"
echo "==== CPU: dump calib ====" | tee -a "$OUT/run.log"
"$PY" dump_calib_molmo_prospective.py 2>&1 | tee -a "$OUT/run.log"

if [[ -z "$GPU" ]]; then
  echo "No GPU id passed; skip scoring. Example:" | tee -a "$OUT/run.log"
  echo "  CUDA_VISIBLE_DEVICES=3 bash RUN_LAYER_CALIB_MOLMO_V1.sh 3" | tee -a "$OUT/run.log"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU"
echo "==== GPU score on device $GPU ====" | tee -a "$OUT/run.log"
"$PY" score_molmo_prospective_layers.py --device cuda:0 2>&1 | tee -a "$OUT/score_run.log"

echo "==== CPU analyze ====" | tee -a "$OUT/run.log"
"$PY" analyze_layer_calibration_molmo_v1.py 2>&1 | tee -a "$OUT/run.log"
echo "==== done $(date -Iseconds) ====" | tee -a "$OUT/run.log"
