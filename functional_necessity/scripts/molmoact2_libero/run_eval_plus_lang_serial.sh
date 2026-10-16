#!/usr/bin/env bash
# Full MolmoAct2 Plus Language NFE=10 on GPU3: Spatial → Object → Goal → Long.
set -u
set -o pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_LOG=/home/guorj/Skill_Efficient_VLA/functional_necessity/results/logs/eval_molmoact2_full_nfe10_lang_serial.log

echo "==== SERIAL START $(date -Iseconds) ====" | tee -a "$MASTER_LOG"
for suite in libero_spatial libero_object libero_goal libero_10; do
  case "$suite" in
    libero_spatial) suffix=spatial ;;
    libero_object) suffix=object ;;
    libero_goal) suffix=goal ;;
    libero_10) suffix=10 ;;
  esac
  TAG="molmoact2_full_nfe10_lang_${suffix}"
  bash "$DIR/run_eval_plus_lang.sh" "$TAG" "$suite" "Language Instructions" 10 3 \
    || { echo "==== FAIL $TAG ====" | tee -a "$MASTER_LOG"; exit 1; }
done
echo "==== SERIAL DONE $(date -Iseconds) ====" | tee -a "$MASTER_LOG"
