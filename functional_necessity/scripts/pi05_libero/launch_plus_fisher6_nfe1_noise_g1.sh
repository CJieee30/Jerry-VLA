#!/usr/bin/env bash
set -eu
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
RUN="$ROOT/scripts/pi05_libero/run_eval_plus_fisher6_nfe1_noise.sh"
LOG_DIR="$ROOT/results/logs"
NAME=plus_fisher6_nfe1_noise_gpu1
mkdir -p "$LOG_DIR"

if tmux has-session -t "$NAME" 2>/dev/null; then
  echo "tmux session exists: $NAME"
  exit 0
fi

tmux new-session -d -s "$NAME" \
  "bash '$RUN' 1 2>&1 | tee -a '$LOG_DIR/${NAME}.log'; echo DONE exit=\$?; sleep 5"
echo "started tmux $NAME on GPU1 (Fisher Drop-6 NFE=1 Sensor Noise)"
