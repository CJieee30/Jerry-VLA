#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/stage2_care_v2
OUT=/home/guorj/Skill_Efficient_VLA/functional_necessity/results/stage2_care_v2/layer_calibration_molmo_v1
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
echo "==== FRAME GPU0 START $(date -Iseconds) ====" | tee -a "$OUT/score_run.log"
"$PY" score_molmo_prospective_layers.py --device cuda:0 2>&1 | tee -a "$OUT/score_run.log"
echo "==== FRAME GPU0 DONE $(date -Iseconds) ====" | tee -a "$OUT/score_run.log"
