#!/usr/bin/env bash
# MolmoAct2 keep-KV skip [0] only. 400-ep protocol (10/task). Spatial first;
# if Spatial pc_success is 0, remaining suites marked 0. Does not overwrite
# historical 2000-ep dirs.
# Usage: bash run_l0_only.sh GPU
set -u
set -o pipefail

GPU="${1:?gpu}"
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
OUT="$FN/results/molmoact2_libero/l0_only"
LOG="$FN/results/molmoact2_libero/l0_only.log"
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
SUITES=(libero_spatial libero_object libero_goal libero_10)

if [ -f "$OUT/eval_info.json" ]; then
  echo "[skip] molmo l0_only already has eval_info.json" | tee -a "$LOG"
  exit 0
fi

mkdir -p "$OUT" "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="$GPU"
export MOLMO_DROP_TAG=l0_only
export MOLMO_N_EPISODES=10
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/home/guorj/lerobot/src
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "==== START molmo l0_only $(date -Iseconds) gpu=$GPU drop=[0] ====" | tee -a "$LOG"
cd /home/guorj/lerobot
early=0
for suite in "${SUITES[@]}"; do
  suite_out="$OUT/$suite"
  if [ -f "$suite_out/eval_info.json" ]; then
    echo "[resume] $suite already finished" | tee -a "$LOG"
  else
    echo "==== SUITE $suite $(date -Iseconds) ====" | tee -a "$LOG"
    export MOLMO_SUITE="$suite"
    set +e
    "$PY" "$FN/scripts/molmoact2_libero/run_l0_only.py" >>"$LOG" 2>&1
    RC=$?
    set -e
    if [ "$RC" -ne 0 ]; then
      echo "==== EXIT molmo l0_only $suite fail exit=$RC $(date -Iseconds) ====" | tee -a "$LOG"
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

set +e
"$PY" "$FN/scripts/stage1_recovery_free/merge_suite_eval.py" "$OUT" "$early" \
  "MolmoAct2 keep-KV skip [0] only. Spatial pc_success==0; Object/Goal/Long marked 0." | tee -a "$LOG"
MRC=$?
set -e
if [ "$MRC" -ne 0 ]; then
  echo "==== EXIT molmo l0_only merge fail exit=$MRC $(date -Iseconds) ====" | tee -a "$LOG"
  exit "$MRC"
fi
echo "==== EXIT molmo l0_only exit=0 $(date -Iseconds) ====" | tee -a "$LOG"
exit 0
