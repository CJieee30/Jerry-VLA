#!/usr/bin/env bash
# Closed-loop eval with extra_drop = f(task) on Drop-9 recovered π0.5.
# Usage:
#   bash run_eval_task_extra.sh <gpu> <port> <tag> <map.json>
# Example:
#   bash run_eval_task_extra.sh 3 18103 oracle_k1 results/task_layer/map_oracle_k1.json
#   bash run_eval_task_extra.sh 3 18103 oracle_k1_shuffled results/task_layer/map_oracle_k1_shuffled.json
set -u
set -o pipefail

GPU="${1:?gpu}"
PORT="${2:?port}"
TAG="${3:?tag, e.g. oracle_k1}"
MAP="${4:?path to extra_drop=f(task) JSON}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$MAP" != /* ]]; then
  MAP="$ROOT/$MAP"
fi
if [[ ! -f "$MAP" ]]; then
  echo "map not found: $MAP" >&2
  exit 2
fi
OPENPI="/home/guorj/VLADrop/models/pi0.5/openpi"
CKPT="$ROOT/checkpoints/VLADrop_pi05_LIBERO_Drop9Block_GateProbe"
OUT="$ROOT/results/eval_task_extra/$TAG"
LOGDIR="$ROOT/results/logs"
DTR_PY="$ROOT/.venv/bin/python"
LER_PY="${LER_PY:-/home/guorj/.miniforge3/envs/lerobot/bin/python}"
SUITES=(libero_spatial libero_object libero_goal libero_10)
STAGE1="7,10,11,12,13,14,15,16,17"

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

"$DTR_PY" - <<PY
import json
from pathlib import Path
import sys
sys.path.insert(0, "$ROOT")
from task_extra_map import load_task_extra_map, union_drop
from backend import STAGE1_DROP
mapping = load_task_extra_map(Path("$MAP"))
extras = sorted({tuple(t["extra_drop"]) for t in mapping["tasks"]})
meta = {
    "tag": "$TAG",
    "kind": mapping.get("kind"),
    "map_path": "$MAP",
    "checkpoint": "s1ghhh/VLADrop_pi05_LIBERO_Drop9Block_GateProbe",
    "stage1_drop": list(STAGE1_DROP),
    "k": mapping.get("k", 1),
    "leaky": mapping.get("leaky"),
    "n_tasks": len(mapping["tasks"]),
    "unique_extras": [list(x) for x in extras],
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
        "extra_drop": "f(task)",
    },
}
Path("$OUT/drop_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps({"kind": meta["kind"], "unique_extras": meta["unique_extras"]}, indent=2))
PY

echo "==== START $TAG gpu=$GPU port=$PORT map=$MAP $(date -Iseconds) ===="
cat "$OUT/drop_meta.json"

SERVER_LOG="$LOGDIR/eval_task_extra_${TAG}_server.log"
"$DTR_PY" "$ROOT/serve_fn_policy.py" \
  --dir "$CKPT" \
  --port "$PORT" \
  --llm-drop "$STAGE1" \
  --task-extra-map "$MAP" \
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
    --task-extra-map "$MAP" \
    --output "$suite_json" \
    2>&1 | tee -a "$LOGDIR/eval_task_extra_${TAG}_${suite}.log"; then
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
