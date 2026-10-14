#!/usr/bin/env bash
# π0.5 Fisher Drop-6 NFE=1 on LIBERO-plus Light Conditions + Background Textures.
# New tags only (plus_fisher6_nfe1_{light,background}_*); does not overwrite NFE=10 rows.
#
# Usage:
#   bash run_eval_plus_fisher6_nfe1_light_background_pair.sh          # GPU1 light + GPU3 background
#   bash run_eval_plus_fisher6_nfe1_light_background_pair.sh light
#   bash run_eval_plus_fisher6_nfe1_light_background_pair.sh background
set -u
set -o pipefail

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
DIM="$ROOT/scripts/pi05_libero/run_eval_plus_dim.sh"
DROP="1,13,14,15,16,17"
NFE=1
LIGHT_GPU=1
BACKGROUND_GPU=3
LIGHT_PREFIX=plus_fisher6_nfe1_light
BACKGROUND_PREFIX=plus_fisher6_nfe1_background

_run_light() {
  bash "$DIM" "$LIGHT_GPU" "$LIGHT_PREFIX" "$DROP" "Light Conditions" "$NFE"
}

_run_background() {
  bash "$DIM" "$BACKGROUND_GPU" "$BACKGROUND_PREFIX" "$DROP" "Background Textures" "$NFE"
}

case "${1:-both}" in
  light)
    _run_light
    ;;
  background)
    _run_background
    ;;
  both|"")
    _run_light &
    PID_LIGHT=$!
    _run_background &
    PID_BG=$!
    FAIL=0
    wait "$PID_LIGHT" || FAIL=1
    wait "$PID_BG" || FAIL=1
    exit "$FAIL"
    ;;
  *)
    echo "usage: $0 [both|light|background]" >&2
    exit 2
    ;;
esac
