#!/usr/bin/env bash
# SmolVLA Drop-1 on GPU2. Five named methods collapse to two unique sets:
# [15] Taylor / GateProbe / PPL; [14] IGIA / Fisher.
set -u
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
LOG="$FN/results/smolvla_libero/drop1_queue_g2.log"
echo "queue SmolVLA Drop-1 GPU2 start $(date -Iseconds)" >>"$LOG"
for method in taylor igia; do
  echo "start $method $(date -Iseconds)" >>"$LOG"
  bash "$FN/scripts/smolvla_libero/run_drop4.sh" 2 "$method" 1
  echo "end $method exit=$? $(date -Iseconds)" >>"$LOG"
done
echo "queue SmolVLA Drop-1 GPU2 done $(date -Iseconds)" >>"$LOG"
