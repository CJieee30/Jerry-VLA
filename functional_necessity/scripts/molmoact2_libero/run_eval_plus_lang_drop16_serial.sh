#!/usr/bin/env bash
# Molmo Plus Language: Taylor Drop-16 materialized, same seeds/protocol as Full NFE=10.
# Usage: bash run_eval_plus_lang_drop16_serial.sh [NFE] [GPU]
set -u
set -o pipefail

NFE="${1:-10}"
GPU="${2:-0}"
DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="molmoact2_drop16_materialized_nfe${NFE}_lang"

for suite in spatial object goal; do
  TAG="${PREFIX}_${suite}"
  bash "$DIR/run_eval_plus_lang.sh" "$TAG" "libero_${suite}" "Language Instructions" "$NFE" "$GPU" drop16_materialized
done
TAG="${PREFIX}_10"
bash "$DIR/run_eval_plus_lang.sh" "$TAG" libero_10 "Language Instructions" "$NFE" "$GPU" drop16_materialized

# NFE=1 Language: use run_eval_plus_lang_drop16_nfe1_serial.sh [GPU] (cluster-wide flock; do not auto-chain here).
