#!/usr/bin/env bash
# Eval one FN extra-drop config on Drop-9 recovered π0.5 (OpenPI serve + LIBERO client).
# Usage: bash run_eval_one.sh <gpu> <port> <method_tag>
#   Drop-3 tags: fisher_ppl | gateprobe | ... | drop9_recovered
#                greedy_ppl | extra_1_2_9  (PPL subset-search winner [1,2,9])
#   Drop-1 leave-one-out: l0 | l1 | l2 | l3 | l4 | l5 | l6 | l8 | l9
set -u
set -o pipefail

GPU="${1:?gpu}"
PORT="${2:?port}"
TAG="${3:?method tag, e.g. fisher_ppl or l9}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI="/home/guorj/VLADrop/models/pi0.5/openpi"
CKPT="$ROOT/checkpoints/VLADrop_pi05_LIBERO_Drop9Block_GateProbe"
RANKS="$ROOT/results/ranks/ranks.json"
if [[ "$TAG" =~ ^l[0-9]+$ ]]; then
  OUT="$ROOT/results/eval_drop1/$TAG"
  LOG_PREFIX="eval_drop1"
else
  OUT="$ROOT/results/eval_drop3/$TAG"
  LOG_PREFIX="eval_drop3"
fi
LOGDIR="$ROOT/results/logs"
DTR_PY="$ROOT/.venv/bin/python"
LER_PY="${LER_PY:-/home/guorj/.miniforge3/envs/lerobot/bin/python}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

export CUDA_VISIBLE_DEVICES="$GPU"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${OPENPI}/src:${OPENPI}/packages/openpi-client/src:${PYTHONPATH:-}"

mkdir -p "$OUT" "$LOGDIR"
if [[ -f "$OUT/eval_info.json" ]]; then
  complete="$("$DTR_PY" - <<PY
import json
from pathlib import Path
info = json.loads(Path("$OUT/eval_info.json").read_text())
ov = info.get("overall") or {}
print("1" if ov.get("n_suites") == 4 and ov.get("pc_success_mean") is not None else "0")
PY
)"
  if [[ "$complete" == "1" ]]; then
    echo "[skip] $TAG already has complete $OUT/eval_info.json"
    exit 0
  fi
  echo "[retry] $TAG previous eval_info.json is incomplete; rerunning"
fi

UNION="$("$DTR_PY" - <<PY
import json
from pathlib import Path
ranks = json.loads(Path("$RANKS").read_text())
tag = "$TAG"
stage1 = list(ranks["stage1_drop"])
remaining = list(ranks["remaining_layers"])
if tag in {"drop9_recovered", "baseline", "stage1"}:
    extra, keep, union = [], remaining, stage1
    methods = ["drop9_recovered"]
elif tag.startswith("l") and tag[1:].isdigit():
    layer = int(tag[1:])
    if layer not in remaining:
        raise SystemExit(f"{tag}: layer {layer} is not in remaining {remaining}")
    extra = [layer]
    keep = [i for i in remaining if i != layer]
    union = sorted(set(stage1) | set(extra))
    methods = [tag]
elif tag in {"greedy_ppl", "ppl_exhaustive", "extra_1_2_9"}:
    extra = [1, 2, 9]
    bad = [i for i in extra if i not in remaining]
    if bad:
        raise SystemExit(f"{tag}: extra {bad} not in remaining {remaining}")
    keep = [i for i in remaining if i not in extra]
    union = sorted(set(stage1) | set(extra))
    methods = ["greedy_ppl", "ppl_exhaustive"]
elif tag.startswith("extra_") and tag.split("_")[1:]:
    extra = [int(x) for x in tag.split("_")[1:]]
    bad = [i for i in extra if i not in remaining]
    if bad:
        raise SystemExit(f"{tag}: extra {bad} not in remaining {remaining}")
    keep = [i for i in remaining if i not in extra]
    union = sorted(set(stage1) | set(extra))
    methods = [tag]
else:
    alias = {"fisher_ppl": "fisher"}
    method = alias.get(tag, tag)
    entry = ranks["methods"][method]["drop"]["3"]
    extra, keep, union = list(entry["drop"]), list(entry["keep"]), list(entry["union_drop"])
    methods = ["fisher", "ppl"] if tag == "fisher_ppl" else [tag]
meta = {
    "tag": tag,
    "methods": methods,
    "checkpoint": "s1ghhh/VLADrop_pi05_LIBERO_Drop9Block_GateProbe",
    "stage1_drop": stage1,
    "extra_drop": extra,
    "keep": keep,
    "union_drop": union,
    "protocol": {
        "n_trials_per_task": 10,
        "seed": 7,
        "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        "replan_steps": 5,
        "action_horizon": 10,
        "num_steps_wait": 10,
        "resize": 224,
        "env_resolution": 256,
        "recovery": False,
        "backend": "openpi_serve_libero_client",
        "config": "pi05_libero_dropped",
    },
}
Path("$OUT/drop_meta.json").write_text(json.dumps(meta, indent=2))
print(",".join(str(i) for i in union))
print(json.dumps(meta, indent=2), file=__import__("sys").stderr)
PY
)"

echo "==== START $TAG gpu=$GPU port=$PORT union=[$UNION] $(date -Iseconds) ===="
cat "$OUT/drop_meta.json"

SERVER_LOG="$LOGDIR/${LOG_PREFIX}_${TAG}_server.log"
"$DTR_PY" "$ROOT/serve_fn_policy.py" \
  --dir "$CKPT" \
  --port "$PORT" \
  --llm-drop "$UNION" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID log=$SERVER_LOG"

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

ok=0
for i in $(seq 1 180); do
  if "$DTR_PY" - "$PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", port))
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
  then
    ok=1
    echo "server port $PORT is open after ${i} checks"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server died while loading; see $SERVER_LOG" >&2
    exit 1
  fi
  sleep 2
done
if [[ "$ok" != 1 ]]; then
  echo "server did not open port $PORT within timeout" >&2
  exit 1
fi
sleep 3

FAILED=0
for suite in "${SUITES[@]}"; do
  suite_json="$OUT/${suite}.json"
  if [[ -f "$suite_json" ]]; then
    echo "[skip suite] $suite"
    continue
  fi
  echo "---- $TAG $suite $(date -Iseconds) ----"
  if ! "$LER_PY" "$ROOT/eval_libero_client.py" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --task-suite "$suite" \
    --num-trials 10 \
    --seed 7 \
    --output "$suite_json" \
    2>&1 | tee -a "$LOGDIR/${LOG_PREFIX}_${TAG}_${suite}.log"; then
    echo "==== FAIL $TAG $suite ====" >&2
    FAILED=1
  fi
done

"$DTR_PY" - <<PY
import json
from pathlib import Path
out = Path("$OUT")
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
info = json.loads((out / "drop_meta.json").read_text())
per = {}
pcs = []
for name in suites:
    p = out / f"{name}.json"
    if not p.exists():
        per[name] = None
        continue
    payload = json.loads(p.read_text())
    per[name] = payload
    pcs.append(payload["pc_success"])
info["suites"] = per
info["overall"] = {
    "pc_success_mean": (sum(pcs) / len(pcs)) if pcs else None,
    "n_suites": len(pcs),
    "spatial": per.get("libero_spatial", {}).get("pc_success") if per.get("libero_spatial") else None,
    "object": per.get("libero_object", {}).get("pc_success") if per.get("libero_object") else None,
    "goal": per.get("libero_goal", {}).get("pc_success") if per.get("libero_goal") else None,
    "long": per.get("libero_10", {}).get("pc_success") if per.get("libero_10") else None,
}
(out / "eval_info.json").write_text(json.dumps(info, indent=2))
ov = info["overall"]
print(
    f"SUMMARY {info['tag']}: "
    f"{ov['spatial']}/{ov['object']}/{ov['goal']}/{ov['long']} avg={ov['pc_success_mean']}"
)
PY

if [[ "$FAILED" != 0 ]]; then
  echo "==== FAIL $TAG $(date -Iseconds) ===="
  exit 1
fi
echo "==== OK $TAG $(date -Iseconds) ===="
exit 0
