#!/usr/bin/env bash
# SmolVLA Drop-2 on GPU3, one after another. Taylor / IGIA / Fisher / GateProbe
# are four distinct sets at k=2 (Fisher does not share IGIA).
set -u
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
LOG="$FN/results/smolvla_libero/drop2_queue_g3.log"
echo "queue SmolVLA Drop-2 GPU3 start $(date -Iseconds)" >>"$LOG"
for method in taylor igia fisher gateprobe; do
  echo "start $method $(date -Iseconds)" >>"$LOG"
  bash "$FN/scripts/smolvla_libero/run_drop4.sh" 3 "$method" 2
  echo "end $method exit=$? $(date -Iseconds)" >>"$LOG"
done
echo "queue SmolVLA Drop-2 GPU3 done $(date -Iseconds)" >>"$LOG"
