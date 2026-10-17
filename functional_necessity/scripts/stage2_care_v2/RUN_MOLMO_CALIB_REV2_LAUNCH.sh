#!/usr/bin/env bash
# rev2: smoke GPU0 (--with-gate) → frame GPU0 + taylor GPU2 in parallel → analyze CPU
set -euo pipefail
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/stage2_care_v2
OUT=/home/guorj/Skill_Efficient_VLA/functional_necessity/results/stage2_care_v2/layer_calibration_molmo_v1
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
MASTER="$OUT/rev2_launch.log"

mkdir -p "$OUT"
{
  echo "==== REV2 LAUNCH $(date -Iseconds) ===="
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee -a "$MASTER"

for g in 0 2; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
  if [ "${used:-99999}" -gt 500 ]; then
    echo "GPU$g not idle (used=${used}MiB)" | tee -a "$MASTER"
    exit 2
  fi
done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
echo "==== SMOKE GPU0 --with-gate n=2 ====" | tee -a "$MASTER"
if ! "$PY" verify_molmo_score_smoke.py --device cuda:0 --n 2 --with-gate; then
  echo "SMOKE FAILED — abort formal run" | tee -a "$MASTER"
  exit 1
fi
echo "==== SMOKE OK ====" | tee -a "$MASTER"

bash "$ROOT/RUN_MOLMO_CALIB_FRAME_GPU0.sh" >>"$OUT/score_run.log" 2>&1 &
PID0=$!
bash "$ROOT/RUN_MOLMO_CALIB_TAYLOR_GPU2.sh" >>"$OUT/taylor_group_run.log" 2>&1 &
PID2=$!
echo "frame_pid=$PID0 taylor_pid=$PID2" | tee -a "$MASTER"
wait "$PID0"
RC0=$?
wait "$PID2"
RC2=$?
echo "frame_exit=$RC0 taylor_exit=$RC2" | tee -a "$MASTER"
if [ "$RC0" -ne 0 ] || [ "$RC2" -ne 0 ]; then
  exit 1
fi
"$PY" analyze_layer_calibration_molmo_v1.py | tee -a "$MASTER"
echo "==== REV2 DONE $(date -Iseconds) ====" | tee -a "$MASTER"
