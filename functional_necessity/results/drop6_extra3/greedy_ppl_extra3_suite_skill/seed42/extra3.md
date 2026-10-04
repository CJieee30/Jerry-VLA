# v044 Drop-6 + extra-3 (greedy_ppl, suite×skill, seed 42)

Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Universal skip `U=[1, 13, 14, 15, 16, 17]`. Candidates `[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`. Greedy PPL on skip(U), frozen ξ, extra-k=3. Goal-pick excludes Goal5.

- Calib: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/calib/suite_skill_32x8_seed42` (32×8)
- Global greedy control: `[2, 7, 11]`

| Cell | extra-3 | vs `{2,7,11}` | Jaccard | n_source | Δloss | seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Spatial/pick` | `[2, 7, 9]` | `[9, 11]` | 0.500 | 20662 | 0.386573 | 510.8 |
| `Spatial/place` | `[2, 7, 8]` | `[8, 11]` | 0.500 | 14932 | 0.310799 | 512.0 |
| `Object/pick` | `[2, 10, 11]` | `[7, 10]` | 0.500 | 27566 | 0.188439 | 512.1 |
| `Object/place` | `[2, 5, 8]` | `[5, 7, 8, 11]` | 0.200 | 21179 | 0.227009 | 511.8 |
| `Goal/pick` | `[2, 7, 11]` | `[]` | 1.000 | 12411 | 0.309053 | 512.0 |
| `Goal/place` | `[2, 4, 11]` | `[4, 7]` | 0.500 | 8750 | 0.279519 | 512.1 |
| `Goal/turn` | `[2, 7, 11]` | `[]` | 1.000 | 3454 | 0.351120 | 512.6 |
| `Long/pick` | `[2, 7, 11]` | `[]` | 1.000 | 37270 | 0.274084 | 513.0 |
| `Long/place` | `[2, 7, 10]` | `[10, 11]` | 0.500 | 28590 | 0.274142 | 512.7 |
| `Long/turn` | `[2, 7, 9]` | `[9, 11]` | 0.500 | 4366 | 0.279098 | 513.2 |

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop6_extra3/greedy_ppl_extra3_suite_skill/seed42/extra3.json`

