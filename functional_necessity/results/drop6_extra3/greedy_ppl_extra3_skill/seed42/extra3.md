# v044 Drop-6 + extra-3 (greedy_ppl, skill pool, seed 42)

Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Universal skip `U=[1, 13, 14, 15, 16, 17]`. Candidates `[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`. Greedy PPL on skip(U), frozen ξ, extra-k=3. Pools are AtomicVLA verbs after a 10-frame boundary filter.

- Calib: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/calib/per_skill_32x8_seed42` (32×8)
- Global greedy control (not re-searched): `[2, 7, 11]`
- Gate A this seed (≥2 skills differ): **True**

| Skill | extra-3 | vs `{2,7,11}` symdiff | Jaccard | Δloss | seconds |
| --- | --- | --- | ---: | ---: | ---: |
| `pick` | `[2, 7, 11]` | `[]` | 1.000 | 0.272193 | 511.0 |
| `place` | `[2, 7, 11]` | `[]` | 1.000 | 0.264146 | 513.0 |
| `open` | `[2, 7, 11]` | `[]` | 1.000 | 0.249902 | 514.6 |
| `close` | `[2, 7, 10]` | `[10, 11]` | 0.500 | 0.206603 | 515.2 |
| `turn` | `[2, 9, 11]` | `[7, 9]` | 0.500 | 0.308919 | 514.4 |

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop6_extra3/greedy_ppl_extra3_skill/seed42/extra3.json`

