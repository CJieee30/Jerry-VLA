#!/usr/bin/env bash
# One LIBERO eval per GPU. Do not colocate two closed-loop evals.
# yany TimesNet on GPU1 is not a LIBERO eval and may share the card.
set -eu
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
LOG="$FN/results/smolvla_libero/drop4_dispatch.log"
METHODS=(magnitude cossim cka ppl taylor fisher igia gateprobe hessian)
GPUS=(2 3)
POLL=30

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "dispatch SmolVLA Drop-4 start $(date -Iseconds)"

gpu_has_libero_eval() {
  local gpu="$1" pid cmd
  while IFS= read -r pid; do
    [ -z "${pid:-}" ] && continue
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [ -z "$cmd" ] && continue
    case "$cmd" in
      *timesnet*) continue ;;
    esac
    if echo "$cmd" | grep -Eq 'lerobot.scripts.lerobot_eval|lerobot-eval|molmoact2_libero|smolvla_libero/run_|openvla_oft|pi05_libero|cogact_libero|eval_drop|run_drop'; then
      return 0
    fi
  done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$' || true)
  return 1
}

method_done() {
  local method="$1"
  [ -f "$FN/results/smolvla_libero/drop4_${method}/eval_info.json" ]
}

method_running() {
  local method="$1"
  tmux has-session -t "fn-smolvla-drop4-${method}" 2>/dev/null
}

next_method() {
  local method
  for method in "${METHODS[@]}"; do
    if method_done "$method"; then
      continue
    fi
    if method_running "$method"; then
      continue
    fi
    echo "$method"
    return 0
  done
  return 1
}

all_done() {
  local method
  for method in "${METHODS[@]}"; do
    method_done "$method" || return 1
  done
  return 0
}

launch_on() {
  local gpu="$1" method="$2"
  local session="fn-smolvla-drop4-${method}"
  echo "assign $method -> GPU${gpu} $(date -Iseconds)"
  tmux new-session -d -s "$session" "bash $FN/scripts/smolvla_libero/run_drop4.sh $gpu $method"
}

while ! all_done; do
  for gpu in "${GPUS[@]}"; do
    if all_done; then
      break
    fi
    if gpu_has_libero_eval "$gpu"; then
      continue
    fi
    method="$(next_method || true)"
    if [ -z "${method:-}" ]; then
      break
    fi
    launch_on "$gpu" "$method"
    # Give the eval process time to attach to the GPU before the next assign.
    sleep 15
    if ! gpu_has_libero_eval "$gpu"; then
      echo "WARN GPU${gpu} did not show a LIBERO process after launching $method $(date -Iseconds)"
    fi
  done
  if all_done; then
    break
  fi
  sleep "$POLL"
done

echo "dispatch SmolVLA Drop-4 all assigned/done $(date -Iseconds)"
