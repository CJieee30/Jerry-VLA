#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/stage2_care_v2
OUT=/home/guorj/Skill_Efficient_VLA/functional_necessity/results/stage2_care_v2/layer_calibration_molmo_v1
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
echo "==== TAYLOR GPU2 START $(date -Iseconds) ====" | tee -a "$OUT/taylor_group_run.log"
"$PY" score_molmo_prospective_taylor_groups.py --device cuda:0 2>&1 | tee -a "$OUT/taylor_group_run.log"
echo "==== TAYLOR GPU2 DONE $(date -Iseconds) ====" | tee -a "$OUT/taylor_group_run.log"
