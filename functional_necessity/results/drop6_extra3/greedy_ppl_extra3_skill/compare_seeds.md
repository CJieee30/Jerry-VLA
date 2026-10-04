# P0 skill extra-3: seed 42 vs seed 43

- Global greedy control: `[2, 7, 11]`
- Gate A (≥2 skills differ from global on **both** seeds): **True**
- Skills differing on both seeds: `['close', 'turn']`

| Skill | seed42 | seed43 | same across seeds | both ≠ `{2,7,11}` |
| --- | --- | --- | --- | --- |
| `pick` | `[2, 7, 11]` | `[2, 7, 10]` | False | False |
| `place` | `[2, 7, 11]` | `[2, 7, 11]` | True | False |
| `open` | `[2, 7, 11]` | `[2, 7, 11]` | True | False |
| `close` | `[2, 7, 10]` | `[2, 7, 9]` | False | True |
| `turn` | `[2, 9, 11]` | `[2, 5, 9]` | False | True |

## Closed-loop implication

Gate A passed. Step 3 should run Goal0+Goal7 first. Add close / pick-place only if those masks differ from `{2,7,11}`.

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop6_extra3/greedy_ppl_extra3_skill/compare_seeds.json`

