#!/usr/bin/env bash
set -eu
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PAIR="$ROOT/scripts/pi05_libero/run_eval_plus_noise_resume_pair.sh"
LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"

_start() {
  local name="$1"
  local mode="$2"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "tmux session exists: $name"
    return 0
  fi
  tmux new-session -d -s "$name" \
    "bash '$PAIR' '$mode' 2>&1 | tee -a '$LOG_DIR/${name}.log'; echo DONE exit=\$?; sleep 5"
  echo "started tmux $name ($mode)"
}

_start plus_full_noise_resume_gpu0 full
_start plus_fisher6_noise_resume_gpu2 fisher6
tmux ls 2>/dev/null | rg 'plus_.*noise_resume' || true
