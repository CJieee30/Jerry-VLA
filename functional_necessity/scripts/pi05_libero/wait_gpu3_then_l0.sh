#!/usr/bin/env bash
# Wait until SmolVLA Drop-2 on GPU3 has written eval_info, then π0.5 skip L0.
set -u
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
DONE="$FN/results/smolvla_libero/drop2_gateprobe/eval_info.json"
LOG="$FN/results/logs/eval_k1_l0_wait.log"
mkdir -p "$(dirname "$LOG")"
echo "wait GPU3 Drop-2 then π0.5 L0-only $(date -Iseconds)" | tee -a "$LOG"

gpu3_has_guorj_smolvla() {
  local pids p user cmd
  pids=$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  for p in $pids; do
    [ -z "$p" ] && continue
    user=$(ps -p "$p" -o user= 2>/dev/null || true)
    cmd=$(ps -p "$p" -o cmd= 2>/dev/null || true)
    if [ "$user" = guorj ] && echo "$cmd" | grep -q smolvla; then
      return 0
    fi
  done
  return 1
}

while [ ! -f "$DONE" ]; do
  sleep 30
done
echo "drop2_gateprobe eval_info present $(date -Iseconds)" | tee -a "$LOG"
sleep 20
while gpu3_has_guorj_smolvla; do
  echo "GPU3 still has SmolVLA $(date -Iseconds)" >>"$LOG"
  sleep 20
done
echo "GPU3 free, start k1_l0 $(date -Iseconds)" | tee -a "$LOG"
bash "$FN/scripts/pi05_libero/run_eval_named.sh" 3 k1_l0 l0 1 0
echo "k1_l0 done exit=$? $(date -Iseconds)" | tee -a "$LOG"
