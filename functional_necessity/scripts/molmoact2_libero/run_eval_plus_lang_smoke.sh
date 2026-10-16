#!/usr/bin/env bash
# Smoke: 2 Language-Instruction spatial variants, separate dir (not in official stats).
set -u
set -o pipefail

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
CLASS=/home/guorj/LIBERO-plus/libero/libero/benchmark/task_classification.json
TAG=molmoact2_smoke_full_nfe10_lang_spatial
OUT="$ROOT/results/eval_libero_plus/_smoke/$TAG"
LOG="$ROOT/results/logs/eval_${TAG}.log"
GPU=3

mkdir -p "$OUT" "$(dirname "$LOG")"
if [ -f "$OUT/eval_info.json" ]; then
  echo "[skip] smoke already done"
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

IDS="$("$PY" - "$CLASS" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
suite, cat = "libero_spatial", "Language Instructions"
ids = [item["id"] - 1 for item in payload[suite] if item["category"] == cat][:2]
print(json.dumps(ids))
print(f"# smoke n={len(ids)} ids={ids}", file=sys.stderr)
PY
)"

echo "==== SMOKE START $TAG $(date -Iseconds) ====" | tee "$LOG"
"$PY" "$ROOT/scripts/molmoact2_libero/eval_plus_molmo.py" \
  --output_dir "$OUT" \
  --job_name "$TAG" \
  --suite libero_spatial \
  --task_ids "$IDS" \
  --category "Language Instructions" \
  --num_inference_steps 10 \
  --classification_json "$CLASS" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "==== SMOKE EXIT $RC $(date -Iseconds) ====" | tee -a "$LOG"
exit "$RC"
