#!/usr/bin/env bash
# MolmoAct2 LIBERO-plus one-category eval (1 trial per variant).
# Usage: bash run_eval_plus_lang.sh TAG SUITE [CATEGORY] [NFE] [GPU] [VARIANT]
#   VARIANT: full (default) | drop16_materialized
#   TAG e.g. molmoact2_full_nfe10_lang_spatial
#   SUITE: libero_spatial | libero_object | libero_goal | libero_10
# GPU defaults to 3; override only for debugging.
set -u
set -o pipefail

TAG="${1:?tag}"
SUITE="${2:?suite}"
CATEGORY="${3:-Language Instructions}"
NFE="${4:-10}"
GPU="${5:-3}"
VARIANT="${6:-full}"

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
OUT="$ROOT/results/eval_libero_plus/$TAG"
LOG="$ROOT/results/logs/eval_${TAG}.log"
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
CLASS=/home/guorj/LIBERO-plus/libero/libero/benchmark/task_classification.json
SCRIPT="$ROOT/scripts/molmoact2_libero/eval_plus_molmo.py"

mkdir -p "$OUT" "$(dirname "$LOG")"
if [ -f "$OUT/eval_info.json" ]; then
  echo "[skip] $TAG exists" | tee -a "$LOG"
  exit 0
fi

# shellcheck disable=SC1091
source /home/guorj/.libero_plus/env.sh
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IDS="$("$PY" - "$CLASS" "$SUITE" "$CATEGORY" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
suite, cat = sys.argv[2], sys.argv[3]
ids = [item["id"] - 1 for item in payload[suite] if item["category"] == cat]
if not ids:
    raise SystemExit(f"no {cat} tasks in {suite}")
print(json.dumps(ids))
print(f"# {suite} {cat} n={len(ids)}", file=sys.stderr)
PY
)"

echo "==== START $TAG $(date -Iseconds) gpu=$GPU suite=$SUITE category=$CATEGORY nfe=$NFE n_ids=$(echo "$IDS" | $PY -c 'import json,sys; print(len(json.load(sys.stdin)))') ====" | tee -a "$LOG"
"$PY" "$SCRIPT" \
  --output_dir "$OUT" \
  --job_name "$TAG" \
  --suite "$SUITE" \
  --task_ids "$IDS" \
  --category "$CATEGORY" \
  --num_inference_steps "$NFE" \
  --variant "$VARIANT" \
  --classification_json "$CLASS" >>"$LOG" 2>&1
RC=$?
echo "==== EXIT $TAG exit=$RC $(date -Iseconds) ====" | tee -a "$LOG"
exit "$RC"
