#!/usr/bin/env bash
# Prefix-trim closed-loop LIBERO eval. Does not use run_eval_crop.sh (greedy / default NFE).
#
# Usage:
#   bash run_eval_trim.sh trim-only
#   bash run_eval_trim.sh trim-grid
#   bash run_eval_trim.sh trim-only --gpu 1
#   bash run_eval_trim.sh trim-grid --nfe 10 --gpu 1 --allow-shared
#
# Default remains exclusive GPU + NFE=1 so historical NFE=1 commands stay valid.
# --allow-shared bypasses the exclusive gate only; still requires --gpu, >=12 GiB
# free, and at most one of this project's eval_drop.py per GPU.
set -u
set -o pipefail

MODE="${1:-}"
if [ "$MODE" != "trim-only" ] && [ "$MODE" != "trim-grid" ]; then
  echo "usage: bash run_eval_trim.sh trim-only|trim-grid [--nfe N] [--gpu N] [--allow-shared]" >&2
  exit 2
fi
shift

NFE=1
ALLOW_SHARED=0
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
    --allow-shared)
      ALLOW_SHARED=1
      shift
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
DROP="1,13,14,15,16,17"
if [ "$MODE" = "trim-only" ]; then
  TAG="eval_trim_only_nfe${NFE}"
else
  TAG="eval_trim_grid_nfe${NFE}"
fi
OUT="$ROOT/results/stage_nfe/crop_vla/$TAG"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="$ROOT/results/logs/${TAG}_${STAMP}.log"
CHECK_JSON="$ROOT/results/stage_nfe/crop_vla/offline_check_trim_grid_nfe${NFE}.json"
CROP_FLAG=0
CROP_SELECTOR=""
if [ "$MODE" = "trim-grid" ]; then
  CROP_FLAG=1
  CROP_SELECTOR="grid"
fi

mkdir -p "$(dirname "$LOG")" "$OUT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH=/home/guorj/lerobot/src
export TRIM_SHARED_EVAL="$ALLOW_SHARED"

cd "$SCRIPT_DIR"

STATUS="$("$PY" -c "
from pathlib import Path
from eval_drop import eval_results_status
st, why = eval_results_status(
    Path('$OUT'),
    expected_nfe=int('$NFE'),
    prefix_trim=True,
    crop=bool(int('$CROP_FLAG')),
    crop_selector='$CROP_SELECTOR' or None,
)
print(st)
print(why)
")"
ST_KIND="$(printf '%s\n' "$STATUS" | sed -n '1p')"
ST_WHY="$(printf '%s\n' "$STATUS" | sed -n '2p')"
echo "[status] $TAG $ST_KIND $ST_WHY" | tee -a "$LOG"
if [ "$ST_KIND" = "complete" ]; then
  echo "[skip] $MODE NFE=$NFE already complete: $OUT" | tee -a "$LOG"
  exit 0
fi
if [ "$ST_KIND" = "mismatch" ]; then
  echo "[refuse] $OUT exists with a different config ($ST_WHY); will not overwrite" | tee -a "$LOG"
  exit 3
fi

PICK_RC=0
if [ "$ALLOW_SHARED" = "1" ]; then
  if [ -z "$GPU_REQ" ]; then
    echo "--allow-shared requires explicit --gpu N" | tee -a "$LOG"
    exit 2
  fi
  PICK_OUT="$("$PY" -c "
from crop_vla import shared_eval_ok
import sys
gpu = int(sys.argv[1])
ok, why, snap = shared_eval_ok(gpu)
print(why, file=sys.stderr)
if not ok:
    raise SystemExit(2)
print(gpu)
" "$GPU_REQ" 2>>"$LOG")" || PICK_RC=$?
else
  PICK_OUT="$("$PY" -c "
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
" "$GPU_REQ" 2>>"$LOG")" || PICK_RC=$?
fi
if [ "$PICK_RC" != "0" ]; then
  echo "==== WAIT GPU not eligible $(date -Iseconds) ====" | tee -a "$LOG"
  exit 2
fi
IDLE="$PICK_OUT"

export CUDA_VISIBLE_DEVICES="$IDLE"
"$PY" -c "
from crop_vla import gpu_shared_snapshot
from pathlib import Path
import json
snap = gpu_shared_snapshot(int('$IDLE'))
Path('$OUT/gpu_occupancy_at_start.json').write_text(json.dumps(snap, indent=2) + '\n')
print('wrote occupancy', snap['memory'], 'apps', len(snap['occupancy']))
" | tee -a "$LOG"

CHECK_OK=0
if [ -f "$CHECK_JSON" ]; then
  if "$PY" -c "
import json, sys
from pathlib import Path
p = Path('$CHECK_JSON')
d = json.loads(p.read_text())
ok = bool(d.get('all_ok')) and int(d.get('nfe') or -1) == int('$NFE') and d.get('selector') == 'grid' and d.get('prefix_trim') is True
sys.exit(0 if ok else 1)
"; then
    CHECK_OK=1
    echo "==== SKIP offline check (existing $CHECK_JSON all_ok nfe=$NFE) ====" | tee -a "$LOG"
  fi
fi
if [ "$CHECK_OK" != "1" ]; then
  CHECK_ARGS=(--gpu "$IDLE" --n-frames 4 --prefix-trim --selector grid --nfe "$NFE")
  if [ "$ALLOW_SHARED" = "1" ]; then
    CHECK_ARGS+=(--allow-shared)
  fi
  echo "==== OFFLINE CHECK $TAG gpu=$IDLE nfe=$NFE $(date -Iseconds) ====" | tee -a "$LOG"
  if ! "$PY" check_crop_vla.py "${CHECK_ARGS[@]}" 2>&1 | tee -a "$LOG"; then
    echo "==== FAIL offline check; skip 400-ep eval $(date -Iseconds) ====" | tee -a "$LOG"
    exit 1
  fi
fi

EVAL_EXTRA=(--prefix-trim)
if [ "$MODE" = "trim-grid" ]; then
  EVAL_EXTRA+=(--crop --crop-selector grid)
fi

echo "==== START $TAG gpu=$IDLE shared=$ALLOW_SHARED nfe=$NFE $(date -Iseconds) ====" | tee -a "$LOG"
"$PY" eval_drop.py \
  --method fisher --k 6 --drop-layers "$DROP" \
  "${EVAL_EXTRA[@]}" \
  --output_dir "$OUT" --job_name "$TAG" \
  --policy.path=lerobot/pi05_libero_finetuned_v044 \
  --policy.n_action_steps=10 \
  --policy.num_inference_steps="$NFE" \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --seed=1000 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "==== EXIT $TAG exit=$RC $(date -Iseconds) ====" | tee -a "$LOG"
exit "$RC"
