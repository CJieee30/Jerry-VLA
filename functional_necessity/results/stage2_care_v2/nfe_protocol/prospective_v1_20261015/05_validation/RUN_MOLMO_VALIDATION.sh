#!/usr/bin/env bash
set -euo pipefail
RUN="$(cd "$(dirname "$0")/.." && pwd)"
SCR=/home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/stage2_care_v2
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MOLMO_PROBE_DATA_ROOT=molmo_root
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
WRAPPER="$SCR/.molmo_validation_probe_local.py"
cp "$RUN/05_validation/molmoact2/probe_molmoact2_nfe_validation.py" "$WRAPPER"
cd "$SCR"
$PY "$WRAPPER" --gpu 2 --allow-shared --ignore-eval-drop \
  --shared-path "$RUN/05_validation/probe_request.pt" \
  --out-dir "$RUN/05_validation/molmoact2/raw" --skip-analyze --nfes 1,2,3,10 \
  >> "$RUN/logs/molmo_validation_probe.log" 2>&1
rm -f "$WRAPPER"
$PY nfe_protocol_v1.py --config "$RUN/config.frozen.json" validation-import \
  --model molmoact2 --probe-dir "$RUN/05_validation/molmoact2/raw"
