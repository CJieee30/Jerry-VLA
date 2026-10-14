#!/usr/bin/env bash
# π0.5 Fisher Drop-6 NFE=1 on LIBERO-plus Camera Viewpoints + Objects Layout.
# New tags only (plus_fisher6_nfe1_{camera,layout}_*); does not overwrite NFE=10 rows.
#
# Usage:
#   bash run_eval_plus_fisher6_nfe1_camera_layout_pair.sh          # GPU1 camera + GPU3 layout
#   bash run_eval_plus_fisher6_nfe1_camera_layout_pair.sh camera
#   bash run_eval_plus_fisher6_nfe1_camera_layout_pair.sh layout
set -u
set -o pipefail

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
DIM="$ROOT/scripts/pi05_libero/run_eval_plus_dim.sh"
DROP="1,13,14,15,16,17"
NFE=1
CAMERA_GPU=1
LAYOUT_GPU=3
CAMERA_PREFIX=plus_fisher6_nfe1_camera
LAYOUT_PREFIX=plus_fisher6_nfe1_layout

_run_camera() {
  bash "$DIM" "$CAMERA_GPU" "$CAMERA_PREFIX" "$DROP" "Camera Viewpoints" "$NFE"
}

_run_layout() {
  bash "$DIM" "$LAYOUT_GPU" "$LAYOUT_PREFIX" "$DROP" "Objects Layout" "$NFE"
}

case "${1:-both}" in
  camera)
    _run_camera
    ;;
  layout)
    _run_layout
    ;;
  both|"")
    _run_camera &
    PID_CAM=$!
    _run_layout &
    PID_LAY=$!
    FAIL=0
    wait "$PID_CAM" || FAIL=1
    wait "$PID_LAY" || FAIL=1
    exit "$FAIL"
    ;;
  *)
    echo "usage: $0 [both|camera|layout]" >&2
    exit 2
    ;;
esac
