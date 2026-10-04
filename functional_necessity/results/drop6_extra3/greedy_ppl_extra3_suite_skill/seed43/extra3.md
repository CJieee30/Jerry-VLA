# v044 Drop-6 + extra-3 (greedy_ppl, suite×skill, seed 43)

Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Universal skip `U=[1, 13, 14, 15, 16, 17]`. Candidates `[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`. Greedy PPL on skip(U), frozen ξ, extra-k=3. Goal-pick excludes Goal5.

- Calib: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/calib/suite_skill_32x8_seed43` (32×8)
- Global greedy control: `[2, 7, 11]`

| Cell | extra-3 | vs `{2,7,11}` | Jaccard | n_source | Δloss | seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Spatial/pick` | `[2, 4, 9]` | `[4, 7, 9, 11]` | 0.200 | 20662 | 0.390569 | 505.5 |
| `Spatial/place` | `[2, 7, 11]` | `[]` | 1.000 | 14932 | 0.306403 | 506.5 |
| `Object/pick` | `[2, 9, 10]` | `[7, 9, 10, 11]` | 0.200 | 27566 | 0.214444 | 506.4 |
| `Object/place` | `[2, 8, 11]` | `[7, 8]` | 0.500 | 21179 | 0.195842 | 506.4 |
| `Goal/pick` | `[2, 10, 11]` | `[7, 10]` | 0.500 | 12411 | 0.343891 | 506.4 |
| `Goal/place` | `[2, 7, 11]` | `[]` | 1.000 | 8750 | 0.323124 | 506.4 |
| `Goal/turn` | `[2, 7, 10]` | `[10, 11]` | 0.500 | 3454 | 0.317773 | 506.6 |
| `Long/pick` | `[2, 7, 10]` | `[10, 11]` | 0.500 | 37270 | 0.274925 | 507.0 |
| `Long/place` | `[2, 7, 11]` | `[]` | 1.000 | 28590 | 0.300883 | 506.9 |
| `Long/turn` | `[2, 9, 11]` | `[7, 9]` | 0.500 | 4366 | 0.264930 | 507.0 |

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop6_extra3/greedy_ppl_extra3_suite_skill/seed43/extra3.json`

