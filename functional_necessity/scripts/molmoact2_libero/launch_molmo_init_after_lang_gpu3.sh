#!/usr/bin/env bash
# Wait for molmoact2_full_nfe10_lang_10 to finish, then run Init serial on GPU3.
# Safe to nohup while lang_serial is still running (one-shot lock).
set -u
set -o pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
LOG="$ROOT/results/logs/eval_molmoact2_chain_lang_to_init.log"
LOCK="$ROOT/results/logs/eval_molmoact2_init_chain.lock"
LANG10_OUT="$ROOT/results/eval_libero_plus/molmoact2_full_nfe10_lang_10"
LANG10_EVAL="$LANG10_OUT/eval_info.json"
POLL_SEC="${POLL_SEC:-60}"

mkdir -p "$(dirname "$LOG")"

exec >>"$LOG" 2>&1
echo "==== CHAIN WATCH START $(date -Iseconds) ===="

if [ -f "$LOCK" ]; then
  echo "lock exists ($LOCK); another chain already started — exit 0"
  exit 0
fi

echo "waiting for $LANG10_EVAL ..."
while [ ! -f "$LANG10_EVAL" ]; do
  sleep "$POLL_SEC"
done
echo "lang_10 eval_info present $(date -Iseconds)"

# Ensure the lang_10 python job has exited (eval_info is written at end).
while pgrep -f "eval_plus_molmo.py.*molmoact2_full_nfe10_lang_10" >/dev/null 2>&1; do
  echo "lang_10 python still running; sleep ${POLL_SEC}s"
  sleep "$POLL_SEC"
done
echo "lang_10 process gone $(date -Iseconds)"

touch "$LOCK"
echo "starting Init serial on GPU3 (flock dedupes vs lang_serial chain)"
bash "$DIR/run_eval_plus_init_serial.sh"
RC=$?
echo "==== CHAIN WATCH EXIT rc=$RC $(date -Iseconds) ===="
exit "$RC"
