#!/usr/bin/env bash
set -euo pipefail

# Local training protocol: Hub mirror. dump_calib also resolves the LIBERO
# snapshot under ~/.cache/huggingface/lerobot/hub so it does not call list_repo_refs.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CALIB="$ROOT/results/calib/pi05_v044_libero_64x8_seed42.pt"
SCORES="$ROOT/results/scores"

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
  echo "==== $tag ===="
  "$PYTHON" "$script" "$@"
}

if [ -f "$CALIB" ]; then
  echo "[skip] dump calibration ($CALIB exists)"
else
  echo "[1/11] dump calibration"
  "$PYTHON" dump_calib.py
fi

run_py magnitude profiling_magnitude.py
run_py cossim profiling_cossim.py
run_py cka profiling_cka.py
run_py gateprobe profiling_gate.py --microbatch-size 2
run_py igia profiling_igia.py --microbatch-size 2
run_py fisher profiling_fisher.py --microbatch-size 2
run_py taylor profiling_taylor.py --microbatch-size 2
run_py ppl profiling_ppl.py
run_py hessian profiling_hessian.py --microbatch-size 1

echo "==== rank ===="
"$PYTHON" rank.py

echo "done. scores -> $SCORES  ranks -> $ROOT/results/ranks"
