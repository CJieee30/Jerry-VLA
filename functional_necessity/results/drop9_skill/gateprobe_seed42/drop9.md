# Skill-aware independent Drop-9 (gateprobe, calib seed 42)

Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Full-model Stage-1 redundancy detector (no skip(U), not extra-3). Calib: AtomicVLA verb pools `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/calib/per_skill_32x8_seed42` 32×8. Scoring seed=42, microbatch=2.

- Global gateprobe Drop-9 (64×8 concat, not re-searched): `[2, 4, 9, 10, 13, 14, 15, 16, 17]`
- Unique skill Drop-9 sets: **5**
- Skills differing from global: `['pick', 'place', 'open', 'close', 'turn']`

| Skill | Drop-9 | vs global symdiff | Jaccard | seconds |
| --- | --- | --- | ---: | ---: |
| `pick` | `[2, 3, 9, 10, 13, 14, 15, 16, 17]` | `[3, 4]` | 0.800 | 103.1 |
| `place` | `[2, 3, 4, 12, 13, 14, 15, 16, 17]` | `[3, 9, 10, 12]` | 0.636 | 103.6 |
| `open` | `[2, 3, 4, 10, 12, 14, 15, 16, 17]` | `[3, 9, 12, 13]` | 0.636 | 103.4 |
| `close` | `[2, 3, 4, 9, 11, 14, 15, 16, 17]` | `[3, 10, 11, 13]` | 0.636 | 103.5 |
| `turn` | `[2, 3, 5, 10, 11, 14, 15, 16, 17]` | `[3, 4, 5, 9, 11, 13]` | 0.500 | 103.6 |

## Drop-9 histogram

| Drop-9 | Skills |
| --- | --- |
| `[2, 3, 9, 10, 13, 14, 15, 16, 17]` | `pick` |
| `[2, 3, 4, 12, 13, 14, 15, 16, 17]` | `place` |
| `[2, 3, 4, 10, 12, 14, 15, 16, 17]` | `open` |
| `[2, 3, 4, 9, 11, 14, 15, 16, 17]` | `close` |
| `[2, 3, 5, 10, 11, 14, 15, 16, 17]` | `turn` |

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop9_skill/gateprobe_seed42/drop9.json`

