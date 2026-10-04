#!/usr/bin/env bash
# Full Drop-4 closed loop for this GPU's share of the eight unique sets.
# Usage: bash run_drop4.sh <gpu>
set -u
set -o pipefail

GPU="${1:?gpu index}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=/home/guorj/.miniforge3/envs/openvla-oft/bin/python
LOGDIR="$FN_ROOT/results/openvla_oft_libero/drop4/logs"
mkdir -p "$LOGDIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OPENVLA_SAVE_ROLLOUT_VIDEO=0
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/home/guorj/openvla-oft/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="/home/guorj/openvla-oft/libero_config"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$SCRIPT_DIR"
exec "$PY" run_drop4.py "$GPU"
