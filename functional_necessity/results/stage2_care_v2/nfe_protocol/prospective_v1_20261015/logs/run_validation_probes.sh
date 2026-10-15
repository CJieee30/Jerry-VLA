#!/usr/bin/env bash
# Prospective validation probes — queued; avoids GPU3 (主线一闭环).
set -euo pipefail
RUN="/home/guorj/Skill_Efficient_VLA/functional_necessity/results/stage2_care_v2/nfe_protocol/prospective_v1_20261015"
PY="/home/guorj/.miniforge3/envs/lerobot/bin/python"
SCR="/home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/stage2_care_v2"
REQ="$RUN/05_validation/probe_request.pt"
CFG="$RUN/config.frozen.json"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MOLMO_PROBE_DATA_ROOT=molmo_root

log() { echo "[$(date -Iseconds)] $*" | tee -a "$RUN/logs/validation_queue.log"; }

import_model() {
  local model=$1 dir=$2
  $PY "$SCR/nfe_protocol_v1.py" --config "$CFG" validation-import --model "$model" --probe-dir "$dir"
  log "IMPORT $model"
}

log "phase1 parallel pi05@1 smolvla@2"
$PY "$SCR/probe_pi05_nfe.py" --gpu 1 --allow-shared --shared-path "$REQ" \
  --out-dir "$RUN/05_validation/pi05/raw" --skip-analyze --nfes 1,2,3,10 \
  >> "$RUN/logs/pi05_validation_probe.log" 2>&1 &
P1=$!
$PY "$SCR/probe_smolvla_nfe.py" --gpu 2 --allow-shared --ignore-eval-drop --shared-path "$REQ" \
  --out-dir "$RUN/05_validation/smolvla/raw" --skip-analyze --nfes 1,2,3,10 \
  >> "$RUN/logs/smolvla_validation_probe.log" 2>&1 &
P2=$!
wait "$P1" && log "DONE pi05" && import_model pi05 "$RUN/05_validation/pi05/raw"
wait "$P2" && log "DONE smolvla" && import_model smolvla "$RUN/05_validation/smolvla/raw"

log "phase2 molmoact2@2"
$PY "$SCR/probe_molmoact2_nfe.py" --gpu 2 --allow-shared --shared-path "$REQ" \
  --out-dir "$RUN/05_validation/molmoact2/raw" --skip-analyze --nfes 1,2,3,10 \
  >> "$RUN/logs/molmo_validation_probe.log" 2>&1
log "DONE molmoact2"
import_model molmoact2 "$RUN/05_validation/molmoact2/raw"
log "all validation probes imported"
