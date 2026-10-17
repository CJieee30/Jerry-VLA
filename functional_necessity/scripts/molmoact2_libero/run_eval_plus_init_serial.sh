#!/usr/bin/env bash
# MolmoAct2 Full NFE=10 LIBERO-plus Robot Initial States on GPU3:
# Spatial → Object → Goal → Long (same recipe as Language serial).
set -u
set -o pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
MASTER_LOG="$ROOT/results/logs/eval_molmoact2_full_nfe10_init_serial.log"
FLOCK="$ROOT/results/logs/eval_molmoact2_init_serial.flock"
CATEGORY="Robot Initial States"
NFE=10
GPU=3

exec 9>"$FLOCK"
if ! flock -n 9; then
  echo "==== INIT SERIAL already running $(date -Iseconds) ====" | tee -a "$MASTER_LOG"
  exit 0
fi

echo "==== INIT SERIAL START $(date -Iseconds) gpu=$GPU category=$CATEGORY nfe=$NFE ====" | tee -a "$MASTER_LOG"
for suite in libero_spatial libero_object libero_goal libero_10; do
  case "$suite" in
    libero_spatial) suffix=spatial ;;
    libero_object) suffix=object ;;
    libero_goal) suffix=goal ;;
    libero_10) suffix=10 ;;
  esac
  TAG="molmoact2_full_nfe10_init_${suffix}"
  bash "$DIR/run_eval_plus_lang.sh" "$TAG" "$suite" "$CATEGORY" "$NFE" "$GPU" \
    || { echo "==== FAIL $TAG ====" | tee -a "$MASTER_LOG"; exit 1; }
done
echo "==== INIT SERIAL DONE $(date -Iseconds) ====" | tee -a "$MASTER_LOG"
