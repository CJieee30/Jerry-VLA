# Extra Drop-3 PPL subset search (Drop-9 recovered π0.5)

Protocol: same as `profiling_ppl.py` (64×8 calib, frozen noise/time, Stage-1 drop on).
Baseline loss: **0.013392**.

## Greedy PPL

- Extra drop: `[1, 2, 9]`
- Keep: `[0, 3, 4, 5, 6, 8]`
- Loss: 0.027664 (Δ +0.014272)

- Step 1: pick L2 → extra `[2]` loss=0.016696
- Step 2: pick L9 → extra `[2, 9]` loss=0.021164
- Step 3: pick L1 → extra `[1, 2, 9]` loss=0.027664

## Exhaustive (84 triples)

- Winner: `[1, 2, 9]` loss=0.027664 (Δ +0.014272)
- Greedy matches winner: **True**

| Rank | Extra | Loss | Δ vs baseline |
| ---: | --- | ---: | ---: |
| 1 | [1, 2, 9] | 0.027664 | +0.014272 |
| 2 | [2, 4, 9] | 0.029579 | +0.016187 |
| 3 | [2, 3, 9] | 0.029893 | +0.016501 |
| 4 | [2, 5, 9] | 0.030290 | +0.016898 |
| 5 | [1, 4, 9] | 0.030466 | +0.017075 |
| 6 | [2, 8, 9] | 0.030611 | +0.017219 |
| 7 | [1, 2, 8] | 0.031176 | +0.017784 |
| 8 | [1, 3, 9] | 0.031364 | +0.017972 |
| 9 | [1, 5, 9] | 0.031696 | +0.018304 |
| 10 | [1, 2, 6] | 0.031961 | +0.018570 |
| 11 | [1, 2, 5] | 0.032077 | +0.018685 |
| 12 | [2, 6, 9] | 0.032418 | +0.019026 |
| 13 | [2, 4, 8] | 0.032664 | +0.019272 |
| 14 | [2, 5, 8] | 0.032715 | +0.019323 |
| 15 | [3, 5, 9] | 0.032834 | +0.019442 |

Reference triples:

- Independent PPL / Fisher `[2,4,9]`: rank 2/84, loss=0.029579, Δ=+0.016187
- LOO-SR cheapest `[1,2,5]`: rank 11/84, loss=0.032077, Δ=+0.018685
- Greedy extra: rank 1/84, loss=0.027664, Δ=+0.014272

Full table: `/home/guorj/Skill_Efficient_VLA/functional_necessity/dtr_drop9_recovered/results/ppl_subset_search/drop3_search.json`
