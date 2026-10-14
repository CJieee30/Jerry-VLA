#!/usr/bin/env bash
# π0.5 Fisher Drop-6 NFE=1 on LIBERO-plus Language + Robot Initial States.
# Protocol matches eval_libero_plus.md except num_inference_steps=1.
# New output tags only (plus_fisher6_nfe1_*); does not overwrite NFE=10 rows.
#
# Usage:
#   bash run_eval_plus_fisher6_nfe1_pair.sh          # launch GPU1 lang + GPU3 init
#   bash run_eval_plus_fisher6_nfe1_pair.sh lang     # GPU1 only
#   bash run_eval_plus_fisher6_nfe1_pair.sh init     # GPU3 only
set -u
set -o pipefail

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
DIM="$ROOT/scripts/pi05_libero/run_eval_plus_dim.sh"
DROP="1,13,14,15,16,17"
NFE=1
LANG_GPU=1
INIT_GPU=3
LANG_PREFIX=plus_fisher6_nfe1_lang
INIT_PREFIX=plus_fisher6_nfe1_init

_run_lang() {
  bash "$DIM" "$LANG_GPU" "$LANG_PREFIX" "$DROP" "Language Instructions" "$NFE"
}

_run_init() {
  bash "$DIM" "$INIT_GPU" "$INIT_PREFIX" "$DROP" "Robot Initial States" "$NFE"
}

case "${1:-both}" in
  lang)
    _run_lang
    ;;
  init)
    _run_init
    ;;
  both|"")
    _run_lang &
    PID_LANG=$!
    _run_init &
    PID_INIT=$!
    FAIL=0
    wait "$PID_LANG" || FAIL=1
    wait "$PID_INIT" || FAIL=1
    exit "$FAIL"
    ;;
  *)
    echo "usage: $0 [both|lang|init]" >&2
    exit 2
    ;;
esac
