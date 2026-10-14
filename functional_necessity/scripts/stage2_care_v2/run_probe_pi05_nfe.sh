#!/usr/bin/env bash
# π0.5 full-model offline NFE probe on CARE v2 480 frames.
# Usage: bash run_probe_pi05_nfe.sh GPU [--correctness-only|--analyze-only]
set -eu
GPU="${1:?gpu}"
MODE="${2:-}"
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
SCRIPT="$ROOT/scripts/stage2_care_v2/probe_pi05_nfe.py"
OUT="$ROOT/results/stage2_care_v2/nfe_probe_pi05"
LOG="$OUT/run.log"
mkdir -p "$OUT"

ARGS=(--gpu "$GPU" --allow-shared --min-free-mib 8000 --out-dir "$OUT")
case "$MODE" in
  --correctness-only) ARGS+=(--correctness-only --limit 1) ;;
  --analyze-only) ARGS+=(--analyze-only) ;;
  "") ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

echo "==== probe_pi05_nfe $(date -Iseconds) gpu=$GPU mode=${MODE:-full} ====" | tee -a "$LOG"
exec "$PY" "$SCRIPT" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
