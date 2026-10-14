#!/usr/bin/env bash
# Resume LIBERO-plus Sensor Noise (Goal + Long). Skips suites with eval_info.json.
# GPU0 = full; GPU2 = Fisher Drop-6.
#
# Usage:
#   bash run_eval_plus_noise_resume_pair.sh
#   bash run_eval_plus_noise_resume_pair.sh full
#   bash run_eval_plus_noise_resume_pair.sh fisher6
set -u
set -o pipefail

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
DIM="$ROOT/scripts/pi05_libero/run_eval_plus_dim.sh"
FULL_GPU=0
FISHER_GPU=2
DROP="1,13,14,15,16,17"
NFE=10

_run_full() {
  bash "$DIM" "$FULL_GPU" plus_full_noise "" "Sensor Noise" "$NFE"
}

_run_fisher6() {
  bash "$DIM" "$FISHER_GPU" plus_fisher6_noise "$DROP" "Sensor Noise" "$NFE"
}

case "${1:-both}" in
  full)
    _run_full
    ;;
  fisher6|fisher)
    _run_fisher6
    ;;
  both|"")
    _run_full &
    PID_F=$!
    _run_fisher6 &
    PID_D=$!
    FAIL=0
    wait "$PID_F" || FAIL=1
    wait "$PID_D" || FAIL=1
    exit "$FAIL"
    ;;
  *)
    echo "usage: $0 [both|full|fisher6]" >&2
    exit 2
    ;;
esac
