#!/usr/bin/env bash
# SmolVLA Drop-3 on GPU0, one after another. Fisher shares the IGIA set.
set -u
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
LOG="$FN/results/smolvla_libero/drop3_queue_g0.log"
echo "queue SmolVLA Drop-3 GPU0 start $(date -Iseconds)" >>"$LOG"
for method in taylor igia gateprobe; do
  echo "start $method $(date -Iseconds)" >>"$LOG"
  bash "$FN/scripts/smolvla_libero/run_drop4.sh" 0 "$method" 3
  echo "end $method exit=$? $(date -Iseconds)" >>"$LOG"
done
echo "queue SmolVLA Drop-3 GPU0 done $(date -Iseconds)" >>"$LOG"
