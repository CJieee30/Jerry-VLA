#!/usr/bin/env bash
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
LOG="$FN/results/openvla_oft_libero/greedy_ppl_eval/gpu3_k3.log"
WAIT="$FN/scripts/wait_gpu_idle.sh"
PY=/home/guorj/.miniforge3/envs/openvla-oft/bin/python
SCRIPT="$FN/scripts/openvla_oft_libero/run_greedy_closed_loop.py"
INFO4="$FN/results/openvla_oft_libero/greedy_ppl_eval/eval/k3_greedy_ppl/libero_10/eval_info.json"

mkdir -p "$(dirname "$LOG")"
chmod +x "$WAIT"
bash "$WAIT" 3 800 30
if [ -f "$INFO4" ]; then
  echo "ALREADY_DONE greedy-ppl drop-3 $(date -Iseconds)" | tee -a "$LOG"
  exit 0
fi
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/guorj/openvla-oft:/home/guorj/openvla-oft/LIBERO
export LIBERO_CONFIG_PATH=/home/guorj/openvla-oft/libero_config
echo "launching OpenVLA-OFT greedy-PPL Drop-3 on GPU3 $(date -Iseconds)" | tee -a "$LOG"
cd /home/guorj
"$PY" "$SCRIPT" 3 >>"$LOG" 2>&1
echo "EXIT:$? $(date -Iseconds)" >>"$LOG"
