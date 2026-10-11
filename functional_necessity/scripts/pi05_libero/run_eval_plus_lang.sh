#!/usr/bin/env bash
# Isolated LIBERO-plus one-category eval (official 1 trial per variant).
# Usage: bash run_eval_plus_lang.sh GPU TAG DROP_CSV SUITE [CATEGORY]
#   DROP_CSV empty  → full model
#   DROP_CSV 1,13,14,15,16,17 → Fisher Drop-6
# SUITE: libero_spatial | libero_object | libero_goal | libero_10
# CATEGORY defaults to "Language Instructions".
set -u
set -o pipefail

GPU="${1:?gpu}"
TAG="${2:?tag}"
DROP="${3-}"
SUITE="${4:?suite}"
CATEGORY="${5:-Language Instructions}"

ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
OUT="$ROOT/results/eval_libero_plus/$TAG"
LOG="$ROOT/results/logs/eval_${TAG}.log"
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
CLASS=/home/guorj/LIBERO-plus/libero/libero/benchmark/task_classification.json

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
print("[" + ",".join(map(str, ids)) + "]")
print(f"# {suite} {cat} n={len(ids)}", file=sys.stderr)
PY
)"

if [ -z "${DROP}" ]; then
  METHOD=baseline
  K=0
  DROP_ARG=""
else
  METHOD=fisher
  K="${DROP//,/ }"
  K="$(set -- $K; echo $#)"
  DROP_ARG="$DROP"
fi

cd "$ROOT/scripts/pi05_libero"
echo "==== START $TAG $(date -Iseconds) gpu=$GPU suite=$SUITE category=$CATEGORY method=$METHOD k=$K drop=$DROP_ARG ====" | tee -a "$LOG"
"$PY" eval_drop.py \
  --method "$METHOD" --k "$K" --drop-layers "$DROP_ARG" \
  --output_dir "$OUT" --job_name "$TAG" \
  --policy.path=lerobot/pi05_libero_finetuned_v044 \
  --policy.n_action_steps=10 \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --env.type=libero_plus \
  --env.task="$SUITE" \
  --env.task_ids="$IDS" \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed=1000 >>"$LOG" 2>&1
RC=$?
echo "==== EXIT $TAG exit=$RC $(date -Iseconds) ====" | tee -a "$LOG"
exit "$RC"
