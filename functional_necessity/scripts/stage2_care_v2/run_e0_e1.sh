#!/usr/bin/env bash
# CARE v2 E0+E1. GPU1 only. Do not touch GPU0/2/3.
set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/guorj/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

source /home/guorj/.miniforge3/etc/profile.d/conda.sh
conda activate lerobot

HERE="/home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/stage2_care_v2"
OUT="/home/guorj/Skill_Efficient_VLA/functional_necessity/results/stage2_care_v2"
mkdir -p "$OUT/logs" "$OUT/scores"
cd "$HERE"

echo "[$(date -Is)] dump_shared" | tee -a "$OUT/logs/e0_e1.log"
python dump_shared.py 2>&1 | tee -a "$OUT/logs/dump_shared.log"

echo "[$(date -Is)] fetch_molmo" | tee -a "$OUT/logs/e0_e1.log"
python fetch_molmo.py 2>&1 | tee -a "$OUT/logs/fetch_molmo.log"

echo "[$(date -Is)] dump_calib pi05" | tee -a "$OUT/logs/e0_e1.log"
python dump_calib.py --stack pi05 2>&1 | tee -a "$OUT/logs/dump_pi05.log"

echo "[$(date -Is)] dump_calib smolvla" | tee -a "$OUT/logs/e0_e1.log"
python dump_calib.py --stack smolvla 2>&1 | tee -a "$OUT/logs/dump_smolvla.log"

echo "[$(date -Is)] dump_calib molmoact2" | tee -a "$OUT/logs/e0_e1.log"
python dump_calib.py --stack molmoact2 2>&1 | tee -a "$OUT/logs/dump_molmo.log"

echo "[$(date -Is)] score pi05" | tee -a "$OUT/logs/e0_e1.log"
python score_frame.py --stack pi05 --device cuda:0 2>&1 | tee -a "$OUT/logs/score_pi05.log"

echo "[$(date -Is)] score smolvla" | tee -a "$OUT/logs/e0_e1.log"
python score_frame.py --stack smolvla --device cuda:0 2>&1 | tee -a "$OUT/logs/score_smolvla.log"

echo "[$(date -Is)] score molmoact2" | tee -a "$OUT/logs/e0_e1.log"
python score_frame.py --stack molmoact2 --device cuda:0 2>&1 | tee -a "$OUT/logs/score_molmo.log"

echo "[$(date -Is)] analyze_e1" | tee -a "$OUT/logs/e0_e1.log"
python analyze_e1.py 2>&1 | tee -a "$OUT/logs/analyze_e1.log"

echo "[$(date -Is)] EXIT:0" | tee -a "$OUT/logs/e0_e1.log"
