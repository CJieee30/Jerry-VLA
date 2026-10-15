#!/usr/bin/env bash
set -euo pipefail
FN=/home/guorj/Skill_Efficient_VLA/functional_necessity
RUN=prospective_v1_20261015
LOG=$FN/results/stage2_care_v2/nfe_protocol/$RUN/logs
REQ=$FN/results/stage2_care_v2/nfe_protocol/$RUN/03_probe/probe_request.pt
OUT=$FN/results/stage2_care_v2/nfe_protocol/$RUN/03_probe
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
PROBE=$FN/scripts/stage2_care_v2
CFG=$FN/scripts/stage2_care_v2/nfe_protocol_config.prospective_prefreeze.json

try_pi05() {
  [ -f "$OUT/pi05/raw/DONE" ] && return 0
  pgrep -f "probe_pi05_nfe.py.*$RUN" >/dev/null && return 0
  for g in 1 2 0; do
    free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    [ "${free:-0}" -lt 6000 ] && continue
    if $PY -c "
import subprocess
raw=subprocess.check_output(['nvidia-smi','-i','$g','--query-compute-apps=pid','--format=csv,noheader'], text=True)
for line in raw.strip().splitlines():
    pid=int(line.strip())
    u=subprocess.check_output(['ps','-o','user=','-p',str(pid)], text=True).strip()
    if u not in ('guorj',):
        raise SystemExit(1)
" 2>/dev/null; then
      echo "$(date -Is) starting pi05 on GPU $g free=${free}MiB" | tee -a "$LOG/queue.log"
      CUDA_VISIBLE_DEVICES=$g nohup $PY $PROBE/probe_pi05_nfe.py --gpu "$g" --allow-shared \
        --shared-path "$REQ" --out-dir "$OUT/pi05/raw" --skip-analyze --nfes 1,2,3,10 \
        >> "$LOG/pi05_probe.log" 2>&1 &
      return 0
    fi
  done
  return 1
}

try_molmo() {
  [ -f "$OUT/molmoact2/raw/DONE" ] && return 0
  pgrep -f "probe_molmoact2_nfe.py.*$RUN" >/dev/null && return 0
  [ ! -f "$OUT/smolvla/raw/DONE" ] && return 1
  for g in 0 2 1; do
    [ "$g" = 3 ] && continue
    free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    [ "${free:-0}" -lt 14000 ] && continue
    if $PY -c "
import subprocess
raw=subprocess.check_output(['nvidia-smi','-i','$g','--query-compute-apps=pid','--format=csv,noheader'], text=True)
for line in raw.strip().splitlines():
    pid=int(line.strip())
    u=subprocess.check_output(['ps','-o','user=','-p',str(pid)], text=True).strip()
    if u not in ('guorj',):
        raise SystemExit(1)
" 2>/dev/null; then
      echo "$(date -Is) starting molmo on GPU $g" | tee -a "$LOG/queue.log"
      CUDA_VISIBLE_DEVICES=$g nohup $PY $PROBE/probe_molmoact2_nfe.py --gpu "$g" --allow-shared \
        --shared-path "$REQ" --out-dir "$OUT/molmoact2/raw" --skip-analyze --nfes 1,2,3,10 --allow-shared \
        >> "$LOG/molmo_probe.log" 2>&1 &
      return 0
    fi
  done
  return 1
}

finalize() {
  for model in pi05 smolvla molmoact2; do
    raw="$OUT/$model/raw"
    [ -f "$raw/DONE" ] || continue
    sealed="$OUT/$model/sealed_union.npz"
    [ -f "$sealed" ] && continue
    echo "$(date -Is) probe-import $model" | tee -a "$LOG/queue.log"
    $PY $FN/scripts/stage2_care_v2/nfe_protocol_v1.py --config "$CFG" probe-import --model "$model" --probe-dir "$raw" \
      >> "$LOG/import_${model}.log" 2>&1 || true
  done
  if [ -f "$OUT/pi05/sealed_union.npz" ] && [ -f "$OUT/smolvla/sealed_union.npz" ] && [ -f "$OUT/molmoact2/sealed_union.npz" ]; then
    if [ ! -f "$FN/results/stage2_care_v2/nfe_protocol/$RUN/04_recommend/recommendations.json" ]; then
      echo "$(date -Is) recommend" | tee -a "$LOG/queue.log"
      $PY $FN/scripts/stage2_care_v2/nfe_protocol_v1.py --config "$CFG" recommend >> "$LOG/recommend.log" 2>&1
    fi
  fi
}

while true; do
  try_pi05 || true
  try_molmo || true
  finalize
  if [ -f "$FN/results/stage2_care_v2/nfe_protocol/$RUN/04_recommend/recommendations.json" ]; then
    echo "$(date -Is) ALL DONE" | tee -a "$LOG/queue.log"
    break
  fi
  sleep 120
done
