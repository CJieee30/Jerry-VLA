#!/usr/bin/env bash
# SmolVLA CARE v2 offline NFE probe. Shared GPU OK. Not closed-loop, not efficiency.
# Usage:
#   bash run_probe_smolvla_nfe.sh --gpu 1 --allow-shared
#   bash run_probe_smolvla_nfe.sh --gpu 1 --allow-shared --correctness-only
set -euo pipefail
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
SCRIPT="$FN/scripts/stage2_care_v2/probe_smolvla_nfe.py"
OUT="$FN/results/stage2_care_v2/nfe_probe_smolvla"
LOG="$OUT/run.log"
mkdir -p "$OUT"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/guorj/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "launch $(date -Is) $*" | tee -a "$LOG"
"$PY" "$SCRIPT" --out-dir "$OUT" "$@" 2>&1 | tee -a "$LOG"
echo "EXIT:${PIPESTATUS[0]} $(date -Is)" | tee -a "$LOG"
