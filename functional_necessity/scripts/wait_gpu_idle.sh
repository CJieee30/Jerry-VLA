#!/usr/bin/env bash
# Wait until GPU $1 has no compute process and used memory is below $2 MiB.
set -eu
GPU="${1:?gpu index}"
LIMIT_MIB="${2:-800}"
POLL="${3:-30}"

gpu_idle() {
  local mem procs
  mem="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  procs="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$' || true)"
  [ "${mem:-99999}" -lt "$LIMIT_MIB" ] && [ -z "$procs" ]
}

echo "waiting for GPU${GPU} idle (<${LIMIT_MIB} MiB, no compute apps) $(date -Iseconds)"
while ! gpu_idle; do
  mem="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  echo "GPU${GPU} busy mem=${mem}MiB $(date -Iseconds)"
  sleep "$POLL"
done
# Recheck after a short settle so we do not steal a card that just changed hands.
sleep 15
if ! gpu_idle; then
  echo "GPU${GPU} became busy again during settle; keep waiting $(date -Iseconds)"
  while ! gpu_idle; do
    sleep "$POLL"
  done
fi
echo "GPU${GPU} idle $(date -Iseconds)"
