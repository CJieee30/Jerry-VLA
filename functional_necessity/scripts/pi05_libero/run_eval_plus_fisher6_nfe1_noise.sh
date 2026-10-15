#!/usr/bin/env bash
# π0.5 Fisher Drop-6 NFE=1 on LIBERO-plus Sensor Noise (four suites).
# New tags only: plus_fisher6_nfe1_noise_{spatial,object,goal,10}
#
# Usage: bash run_eval_plus_fisher6_nfe1_noise.sh [GPU]
set -u
set -o pipefail

GPU="${1:-1}"
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
DIM="$ROOT/scripts/pi05_libero/run_eval_plus_dim.sh"
DROP="1,13,14,15,16,17"
NFE=1
PREFIX=plus_fisher6_nfe1_noise

exec bash "$DIM" "$GPU" "$PREFIX" "$DROP" "Sensor Noise" "$NFE"
