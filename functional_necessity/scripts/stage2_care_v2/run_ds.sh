#!/usr/bin/env bash
# Discontinuous vs smooth timestamp composition. GPU3 only.
set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/guorj/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

source /home/guorj/.miniforge3/etc/profile.d/conda.sh
conda activate lerobot

HERE="/home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/stage2_care_v2"
ROOT="/home/guorj/Skill_Efficient_VLA/functional_necessity/results/stage2_care_v2/composition"
mkdir -p "$ROOT/logs" "$ROOT/disc" "$ROOT/smooth" "$ROOT/scores/disc" "$ROOT/scores/smooth"
cd "$HERE"

echo "[$(date -Is)] select_arms" | tee -a "$ROOT/logs/ds.log"
python select_arms.py 2>&1 | tee -a "$ROOT/logs/select_arms.log"

echo "[$(date -Is)] fetch_molmo" | tee -a "$ROOT/logs/ds.log"
python fetch_molmo.py 2>&1 | tee -a "$ROOT/logs/fetch_molmo.log"

for ARM in disc smooth; do
  SHARED="$ROOT/shared_${ARM}.pt"
  DEST="$ROOT/$ARM"
  SCORES="$ROOT/scores/$ARM"
  mkdir -p "$DEST" "$SCORES"
  for STACK in pi05 smolvla molmoact2; do
    echo "[$(date -Is)] dump $ARM $STACK" | tee -a "$ROOT/logs/ds.log"
    python dump_calib.py --stack "$STACK" --shared "$SHARED" --dest-dir "$DEST" \
      2>&1 | tee -a "$ROOT/logs/dump_${ARM}_${STACK}.log"
    echo "[$(date -Is)] score $ARM $STACK" | tee -a "$ROOT/logs/ds.log"
    python score_frame.py --stack "$STACK" --device cuda:0 \
      --calib-dir "$DEST" --score-dir "$SCORES" \
      2>&1 | tee -a "$ROOT/logs/score_${ARM}_${STACK}.log"
  done
done

echo "[$(date -Is)] analyze_ds" | tee -a "$ROOT/logs/ds.log"
python analyze_ds.py 2>&1 | tee -a "$ROOT/logs/analyze_ds.log"
echo "[$(date -Is)] EXIT:0" | tee -a "$ROOT/logs/ds.log"
