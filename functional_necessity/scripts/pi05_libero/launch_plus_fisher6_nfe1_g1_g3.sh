#!/usr/bin/env bash
# Start tmux jobs: GPU1 Language Instructions, GPU3 Robot Initial States.
set -eu
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
PAIR="$ROOT/scripts/pi05_libero/run_eval_plus_fisher6_nfe1_pair.sh"
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

_start plus_fisher6_nfe1_lang_gpu1 lang
_start plus_fisher6_nfe1_init_gpu3 init
tmux ls | rg 'plus_fisher6_nfe1' || true
