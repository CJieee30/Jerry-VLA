#!/usr/bin/env bash
# CARE_event phase-2: build holdout + offline NFE probes + analysis.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/guorj/.miniforge3/envs/lerobot/bin/python3}"
SCRIPTS="$ROOT/functional_necessity/scripts/stage2_care_v2"
OUT="$ROOT/functional_necessity/results/stage2_care_v2/nfe_probe_care_event_holdout"
HOLDOUT="$OUT/holdout_shared.pt"

"$PY" "$SCRIPTS/holdout_validation.py" --build-only

SMOL_GPU="${SMOL_GPU:-2}"
PI05_GPU="${PI05_GPU:-3}"

"$PY" "$SCRIPTS/probe_smolvla_nfe.py" --gpu "$SMOL_GPU" --allow-shared --ignore-eval-drop \
  --shared-path "$HOLDOUT" \
  --out-dir "$OUT/smolvla" \
  --skip-analyze

"$PY" "$SCRIPTS/probe_pi05_nfe.py" --gpu "$PI05_GPU" --allow-shared \
  --shared-path "$HOLDOUT" \
  --out-dir "$OUT/pi05" \
  --skip-analyze

"$PY" "$SCRIPTS/holdout_validation.py" --analyze-only

echo "Phase-2 complete: $OUT/REPORT.md"
