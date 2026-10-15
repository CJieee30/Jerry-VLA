#!/usr/bin/env bash
# MolmoAct2 Full vs materialized Taylor Drop-16, NFE=10, 400 ep — GPU 3 only.
set -euo pipefail
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
LOGDIR="$FN/results/stage_nfe/logs_nfe10_400_gpu3"
mkdir -p "$LOGDIR"

wait_gpu() {
  while true; do
    free=$(nvidia-smi -i 3 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [ "${free:-0}" -ge 18000 ]; then
      return 0
    fi
    echo "$(date -Is) GPU3 free ${free}MiB < 18GiB; waiting 120s" | tee -a "$LOGDIR/wait.log"
    sleep 120
  done
}

run_variant() {
  local variant=$1
  local tag=$2
  local out="$FN/results/stage_nfe/$tag"
  if [ -f "$out/eval_info.json" ]; then
    echo "SKIP already done $tag"
    return 0
  fi
  wait_gpu
  echo "$(date -Is) START $variant $tag" | tee -a "$LOGDIR/run.log"
  export MOLMO_EVAL_VARIANT="$variant"
  export MOLMO_DROP_TAG="$tag"
  export MOLMO_NFE=10
  export MOLMO_N_EPISODES=10
  if [ "$variant" = "drop16_materialized" ]; then
    export MOLMO_PREFLIGHT_OUT="$tag"
    "$PY" "$FN/scripts/stage_nfe/preflight_molmo_nfe10_materialized.py" \
      2>&1 | tee "$LOGDIR/preflight_${tag}.log"
  fi
  wait_gpu
  "$PY" "$FN/scripts/stage_nfe/run_molmo_nfe10_closed_loop.py" \
    2>&1 | tee "$LOGDIR/${tag}.log"
  echo "$(date -Is) DONE $tag exit=${PIPESTATUS[0]}" | tee -a "$LOGDIR/run.log"
}

run_variant full molmoact2_full_nfe10_400
run_variant drop16_materialized molmoact2_drop16_materialized_nfe10_400
echo "ALL DONE $(date -Is)" | tee -a "$LOGDIR/run.log"
