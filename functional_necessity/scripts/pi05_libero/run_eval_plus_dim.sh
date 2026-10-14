#!/usr/bin/env bash
# Four LIBERO suites, one Plus category, full or Fisher Drop-6.
# Usage: bash run_eval_plus_dim.sh GPU TAG_PREFIX DROP_CSV CATEGORY [NFE]
#   TAG_PREFIX plus_full_init → plus_full_init_{spatial,object,goal,10}
#   DROP_CSV empty, or 1,13,14,15,16,17
# NFE defaults to 10; pass 1 for Fisher Drop-6 NFE=1 Plus cells.
# Does not overwrite an existing eval_info.json. Stops on the first suite failure.
set -u
set -o pipefail

GPU="${1:?gpu}"
PREFIX="${2:?tag prefix}"
DROP="${3-}"
CATEGORY="${4:?category}"
NFE="${5:-10}"
ROOT=/home/guorj/Skill_Efficient_VLA/functional_necessity
ONE="$ROOT/scripts/pi05_libero/run_eval_plus_lang.sh"
SUITES=(libero_spatial:spatial libero_object:object libero_goal:goal libero_10:10)

echo "==== QUEUE $PREFIX $(date -Iseconds) gpu=$GPU category=$CATEGORY drop=${DROP:-<full>} nfe=$NFE ===="
for item in "${SUITES[@]}"; do
  suite="${item%%:*}"
  short="${item##*:}"
  tag="${PREFIX}_${short}"
  bash "$ONE" "$GPU" "$tag" "$DROP" "$suite" "$CATEGORY" "$NFE"
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "==== QUEUE $PREFIX stop at $tag exit=$RC $(date -Iseconds) ===="
    exit "$RC"
  fi
done
echo "==== QUEUE $PREFIX done $(date -Iseconds) ===="
exit 0
