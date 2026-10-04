#!/usr/bin/env bash
# Score lerobot/pi05_base on LIBERO calib and compare keep-sets to DTR Table 9.
# Usage: bash run_pi05_base.sh
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CKPT="${CKPT:-lerobot/pi05_base}"
CALIB="$ROOT/results/calib/pi05_base_libero_64x8_seed42.pt"
SCORES="$ROOT/results/scores_pi05_base"
RANKS="$ROOT/results/ranks_pi05_base"
LOG="$ROOT/results/logs/pi05_base.log"
mkdir -p "$SCORES" "$RANKS" "$(dirname "$LOG")"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x /home/guorj/.miniforge3/envs/lerobot/bin/python ]; then
    PYTHON=/home/guorj/.miniforge3/envs/lerobot/bin/python
  else
    PYTHON=python
  fi
fi

run_py() {
  local tag="$1"
  local script="$2"
  shift 2
  local out="$SCORES/${tag}.pt"
  if [ -f "$out" ]; then
    echo "[skip] $tag ($out exists)"
    return
  fi
  echo "==== $tag $(date -Iseconds) ===="
  local args=(--checkpoint "$CKPT" --save-file "$out")
  # Magnitude is data-independent and does not accept --calib-file.
  if [ "$tag" != "magnitude" ]; then
    args+=(--calib-file "$CALIB")
  fi
  "$PYTHON" "$script" "${args[@]}" "$@"
}

{
  echo "==== start $(date -Iseconds) ckpt=$CKPT gpu=$CUDA_VISIBLE_DEVICES ===="

  if [ -f "$CALIB" ]; then
    echo "[skip] dump calibration ($CALIB exists)"
  else
    echo "==== dump calib ===="
    "$PYTHON" dump_calib.py --checkpoint "$CKPT" --out "$CALIB"
  fi

  # DTR Table 3/9 order, minus CosSim-contiguous.
  run_py gateprobe profiling_gate.py --microbatch-size 2
  run_py taylor profiling_taylor.py --microbatch-size 2
  run_py fisher profiling_fisher.py --microbatch-size 2
  run_py igia profiling_igia.py --microbatch-size 2
  run_py hessian profiling_hessian.py --microbatch-size 1
  run_py magnitude profiling_magnitude.py
  run_py cossim profiling_cossim.py
  run_py ppl profiling_ppl.py

  echo "==== rank ===="
  "$PYTHON" rank.py \
    --scores-dir "$SCORES" \
    --out-dir "$RANKS" \
    --ks 9 12 16 17 \
    --checkpoint-label "$CKPT"

  echo "==== compare DTR Table 9 ===="
  "$PYTHON" compare_dtr.py --ranks-json "$RANKS/ranks.json" --out "$RANKS/DTR_COMPARE.md"

  echo "==== done $(date -Iseconds) ===="
} 2>&1 | tee -a "$LOG"
