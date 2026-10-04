# Skill-aware independent Drop-9 (igia, calib seed 42)

Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Full-model Stage-1 redundancy detector (no skip(U), not extra-3). Calib: AtomicVLA verb pools `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/calib/per_skill_32x8_seed42` 32×8. Scoring seed=42, microbatch=2.

- Global igia Drop-9 (64×8 concat, not re-searched): `[1, 2, 4, 5, 13, 14, 15, 16, 17]`
- Unique skill Drop-9 sets: **5**
- Skills differing from global: `['place', 'open', 'close', 'turn']`

| Skill | Drop-9 | vs global symdiff | Jaccard | seconds |
| --- | --- | --- | ---: | ---: |
| `pick` | `[1, 2, 4, 5, 13, 14, 15, 16, 17]` | `[]` | 1.000 | 124.1 |
| `place` | `[1, 2, 4, 11, 13, 14, 15, 16, 17]` | `[5, 11]` | 0.800 | 123.3 |
| `open` | `[1, 2, 4, 5, 11, 13, 15, 16, 17]` | `[11, 14]` | 0.800 | 123.5 |
| `close` | `[1, 2, 3, 4, 5, 7, 15, 16, 17]` | `[3, 7, 13, 14]` | 0.636 | 123.5 |
| `turn` | `[2, 4, 5, 10, 11, 13, 15, 16, 17]` | `[1, 10, 11, 14]` | 0.636 | 123.1 |

## Drop-9 histogram

| Drop-9 | Skills |
| --- | --- |
| `[1, 2, 4, 5, 13, 14, 15, 16, 17]` | `pick` |
| `[1, 2, 4, 11, 13, 14, 15, 16, 17]` | `place` |
| `[1, 2, 4, 5, 11, 13, 15, 16, 17]` | `open` |
| `[1, 2, 3, 4, 5, 7, 15, 16, 17]` | `close` |
| `[2, 4, 5, 10, 11, 13, 15, 16, 17]` | `turn` |

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop9_skill/igia_seed42/drop9.json`

