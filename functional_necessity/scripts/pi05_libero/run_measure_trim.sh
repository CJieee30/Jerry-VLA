#!/usr/bin/env bash
# Formal exclusive-GPU efficiency: raw / trim-only / trim-grid.
# Refuses a shared card. Do not run closed-loop on the same GPU while this runs.
#
# Usage:
#   bash run_measure_trim.sh
#   bash run_measure_trim.sh --nfe 1
#   bash run_measure_trim.sh --nfe 10
#   bash run_measure_trim.sh --nfe 10 --gpu 1
set -u
set -o pipefail

NFE=1
GPU_REQ=""
while [ $# -gt 0 ]; do
  case "$1" in
    --nfe)
      NFE="${2:?}"
      shift 2
      ;;
    --gpu)
      GPU_REQ="${2:?}"
      shift 2
      ;;
    [0-9]*)
      GPU_REQ="$1"
      shift
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
SCRIPT_DIR="$ROOT/scripts/pi05_libero"
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="$ROOT/results/logs/measure_trim_nfe${NFE}_formal_${STAMP}.log"

mkdir -p "$(dirname "$LOG")"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/guorj/lerobot/src

cd "$SCRIPT_DIR"
if ! IDLE="$("$PY" -c "
from crop_vla import first_exclusive_gpu, gpu_is_exclusive
import sys
want = sys.argv[1]
if want == '':
    gpu, note = first_exclusive_gpu()
    print(note, file=sys.stderr)
    if gpu is None:
        raise SystemExit(2)
    print(gpu)
else:
    gpu = int(want)
    ok, why = gpu_is_exclusive(gpu)
    print(why, file=sys.stderr)
    if not ok:
        raise SystemExit(2)
    print(gpu)
" "$GPU_REQ")"; then
  echo "==== WAIT GPU not idle $(date -Iseconds) ====" | tee -a "$LOG"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$IDLE"
echo "==== FORMAL MEASURE nfe=$NFE gpu=$IDLE $(date -Iseconds) ====" | tee -a "$LOG"
"$PY" measure_crop_efficiency.py \
  --mode formal --gpu "$IDLE" --nfe "$NFE" --prefix-trim --selector grid \
  --groups raw,trim-only,trim-grid \
  --n-frames 16 --warmup 5 --repeats 3 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "==== EXIT measure exit=$RC $(date -Iseconds) ====" | tee -a "$LOG"
exit "$RC"
