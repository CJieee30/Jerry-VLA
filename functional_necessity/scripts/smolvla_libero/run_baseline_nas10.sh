#!/usr/bin/env bash
# 400-ep SmolVLA LIBERO contrast at n_action_steps=10.
# Waits for the first idle GPU among 0-3 so it does not share GPU3 with nas=1.
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
LOG="$FN/results/smolvla_libero/baseline_nas10.log"
DONE="$FN/results/smolvla_libero/baseline_eval_nas10/eval_info.json"
SCRIPT="$FN/scripts/smolvla_libero/run_baseline_nas10.py"
LIMIT_MIB=800
POLL=30

mkdir -p "$(dirname "$LOG")"
if [ -f "$DONE" ]; then
  echo "ALREADY_DONE smolvla libero baseline nas10 $(date -Iseconds)" | tee -a "$LOG"
  exit 0
fi

pick_idle_gpu() {
  local g
  for g in 0 1 2 3; do
    local mem procs
    mem="$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    procs="$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$' || true)"
    if [ "${mem:-99999}" -lt "$LIMIT_MIB" ] && [ -z "$procs" ]; then
      echo "$g"
      return 0
    fi
  done
  return 1
}

echo "waiting for first idle GPU among 0-3 (<${LIMIT_MIB} MiB) $(date -Iseconds)" | tee -a "$LOG"
GPU=""
while true; do
  if GPU="$(pick_idle_gpu)"; then
    break
  fi
  echo "all GPUs busy $(date -Iseconds)" >>"$LOG"
  sleep "$POLL"
done

# Settle so we do not steal a card that just changed hands.
sleep 15
if ! GPU2="$(pick_idle_gpu)" || [ "$GPU2" != "$GPU" ]; then
  echo "GPU${GPU} no longer uniquely idle after settle; keep waiting $(date -Iseconds)" | tee -a "$LOG"
  while true; do
    if GPU="$(pick_idle_gpu)"; then
      sleep 15
      if GPU2="$(pick_idle_gpu)" && [ "$GPU2" = "$GPU" ]; then
        break
      fi
    fi
    sleep "$POLL"
  done
fi

if [ -f "$DONE" ]; then
  echo "ALREADY_DONE smolvla libero baseline nas10 $(date -Iseconds)" | tee -a "$LOG"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=device
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/home/guorj/lerobot/src:${PYTHONPATH:-}
echo "launching SmolVLA LIBERO baseline nas10 on GPU${CUDA_VISIBLE_DEVICES} $(date -Iseconds)" | tee -a "$LOG"
cd /home/guorj/lerobot
"$PY" "$SCRIPT" >>"$LOG" 2>&1
echo "EXIT:$? $(date -Iseconds)" >>"$LOG"
