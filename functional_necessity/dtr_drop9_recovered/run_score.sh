#!/usr/bin/env bash
# Score 9 FN metrics on DTR Drop-9 recovered π0.5. Skip existing results/scores/*.pt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${ROOT}/.venv/bin/python"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TORCHDYNAMO_DISABLE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ ! -x "$PY" ]]; then
  echo "missing venv python: $PY" >&2
  exit 1
fi

mkdir -p "$ROOT/results/scores" "$ROOT/results/ranks" "$ROOT/results/logs"

run_one() {
  local name="$1"
  local script="$2"
  local out="$ROOT/results/scores/${name}.pt"
  local log="$ROOT/results/logs/${name}.log"
  if [[ -f "$out" ]]; then
    echo "[skip] $name already at $out"
    return 0
  fi
  echo "[run] $name on GPU $GPU"
  "$PY" "$ROOT/$script" --device cuda 2>&1 | tee "$log"
}

# data-independent first, then no-grad, then grad
run_one magnitude profiling_magnitude.py
run_one cossim profiling_cossim.py
run_one cka profiling_cka.py
run_one gateprobe profiling_gate.py
run_one fisher profiling_fisher.py
run_one igia profiling_igia.py
run_one taylor profiling_taylor.py
run_one hessian profiling_hessian.py
run_one ppl profiling_ppl.py

"$PY" "$ROOT/rank.py"
echo "done. ranks -> $ROOT/results/ranks/DROP_INDEX_LOOKUP.md"
