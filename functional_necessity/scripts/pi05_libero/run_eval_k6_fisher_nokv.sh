#!/usr/bin/env bash
# π0.5 Fisher Drop-6 without keep-KV. New dir; never touches k6_fisher.
# Spatial first. If Spatial pc_success is 0, remaining suites are marked 0.
# Usage: bash run_eval_k6_fisher_nokv.sh GPU
set -u
set -o pipefail

GPU="${1:?gpu}"
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
OUT="$ROOT/results/eval_drop/k6_fisher_nokv"
LOG="$ROOT/results/logs/eval_k6_fisher_nokv.log"
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
DROP="1,13,14,15,16,17"
SUITES=(libero_spatial libero_object libero_goal libero_10)

if [ -e "$ROOT/results/eval_drop/k6_fisher/eval_info.json" ]; then
  # Historical keep-KV run must stay untouched.
  :
fi
if [ -f "$OUT/eval_info.json" ]; then
  echo "[skip] k6_fisher_nokv already has eval_info.json" | tee -a "$LOG"
  exit 0
fi

mkdir -p "$OUT" "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/guorj/lerobot/src
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
cd "$ROOT/scripts/pi05_libero"

echo "==== START k6_fisher_nokv $(date -Iseconds) gpu=$GPU drop=$DROP skip=no_kv ====" | tee -a "$LOG"

echo "==== SMOKE no_kv $(date -Iseconds) ====" | tee -a "$LOG"
"$PY" eval_drop.py --smoke --skip-mode no_kv --drop-layers "$DROP" >>"$LOG" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "==== EXIT k6_fisher_nokv smoke fail exit=$RC $(date -Iseconds) ====" | tee -a "$LOG"
  exit "$RC"
fi

early=0
for suite in "${SUITES[@]}"; do
  suite_out="$OUT/$suite"
  if [ -f "$suite_out/eval_info.json" ]; then
    echo "[resume] $suite already finished" | tee -a "$LOG"
  else
    echo "==== SUITE $suite $(date -Iseconds) ====" | tee -a "$LOG"
    "$PY" eval_drop.py \
      --method fisher --k 6 --drop-layers "$DROP" --skip-mode no_kv \
      --output_dir "$suite_out" --job_name "k6_fisher_nokv_${suite}" \
      --policy.path=lerobot/pi05_libero_finetuned_v044 \
      --policy.n_action_steps=10 \
      --policy.compile_model=false \
      --policy.gradient_checkpointing=false \
      --env.type=libero \
      --env.task="$suite" \
      --env.control_mode=relative \
      --env.max_parallel_tasks=1 \
      --eval.batch_size=1 \
      --eval.n_episodes=10 \
      --seed=1000 >>"$LOG" 2>&1
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "==== EXIT k6_fisher_nokv $suite fail exit=$RC $(date -Iseconds) ====" | tee -a "$LOG"
      exit "$RC"
    fi
  fi

  if [ "$suite" = "libero_spatial" ]; then
    pc="$("$PY" - "$suite_out/eval_info.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
print(info["per_group"]["libero_spatial"]["pc_success"])
PY
)"
    echo "spatial pc_success=$pc" | tee -a "$LOG"
    if "$PY" -c "raise SystemExit(0 if float('$pc') <= 0.0 else 1)"; then
      echo "Spatial is 0; remaining suites marked 0" | tee -a "$LOG"
      early=1
      break
    fi
  fi
done

"$PY" - "$OUT" "$early" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
early = sys.argv[2] == "1"
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
combined = {"per_task": [], "per_group": {}, "overall": {}}
n_ep = 0
weighted = 0.0
for suite in suites:
    path = out / suite / "eval_info.json"
    if path.is_file():
        info = json.loads(path.read_text())
        combined["per_task"].extend(info.get("per_task", []))
        group = info.get("per_group", {}).get(suite, {})
        combined["per_group"][suite] = group
        n = int(group.get("n_episodes", 100))
        n_ep += n
        weighted += float(group.get("pc_success", 0.0)) * n
        continue
    combined["per_group"][suite] = {
        "avg_sum_reward": 0.0,
        "avg_max_reward": 0.0,
        "pc_success": 0.0,
        "n_episodes": 100,
        "marked_zero": True,
        "video_paths": [],
        "predicted_video_paths": [],
    }
    n_ep += 100
meta_src = next((out / s / "drop_meta.json" for s in suites if (out / s / "drop_meta.json").is_file()), None)
if meta_src is not None:
    (out / "drop_meta.json").write_text(meta_src.read_text())
combined["overall"] = {
    "avg_sum_reward": weighted / max(n_ep, 1) / 100.0,
    "avg_max_reward": weighted / max(n_ep, 1) / 100.0,
    "pc_success": weighted / max(n_ep, 1),
    "n_episodes": n_ep,
    "early_stop_spatial_zero": early,
    "video_paths": [],
    "predicted_video_paths": [],
}
(out / "eval_info.json").write_text(json.dumps(combined, indent=2) + "\n")
if early:
    stopped = {
        "stopped": True,
        "remaining_marked_zero": True,
        "method": "fisher",
        "k": 6,
        "drop": [1, 13, 14, 15, 16, 17],
        "operator": "identity_zero_kv",
        "keep_kv": False,
        "completed": "libero_spatial",
        "suite_success_pct_remaining_zero": {
            suite: float(combined["per_group"][suite]["pc_success"]) for suite in suites
        },
        "avg": float(combined["overall"]["pc_success"]),
        "n_episodes_denom": 400,
        "note": "Spatial pc_success==0; Object/Goal/Long marked 0. Do not overwrite k6_fisher.",
    }
    (out / "stopped.json").write_text(json.dumps(stopped, indent=2) + "\n")
print(json.dumps(combined["overall"], indent=2))
PY
MRC=$?
if [ "$MRC" -ne 0 ]; then
  echo "==== EXIT k6_fisher_nokv merge fail exit=$MRC $(date -Iseconds) ====" | tee -a "$LOG"
  exit "$MRC"
fi

echo "==== EXIT k6_fisher_nokv exit=0 $(date -Iseconds) ====" | tee -a "$LOG"
exit 0
