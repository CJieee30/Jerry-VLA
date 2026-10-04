# Skill greedy extra-k calibration curves (Wave B1)

Source: existing `per_skill/*.json` (no rescoring). Skip graph is already `skip(U)`.
U = `[1, 13, 14, 15, 16, 17]`. Global greedy extra-3 control = `[2, 7, 11]`.

`k_hat(skill, τ) = max { k | greedy-prefix Δloss(k) < τ }`, with frozen τ ∈ {0.12, 0.20, 0.28}.
This table is a **hypothesis**, not a closed-loop budget lock.

## Greedy order and cumulative Δloss

| Skill | seed | k=1 layer | k=2 extra | k=3 extra | Δ(k=1) | Δ(k=2) | Δ(k=3) | baseline loss |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `pick` | 42 | 2 | `[2, 7]` | `[2, 7, 11]` | 0.0648 | 0.1535 | 0.2722 | 0.2520 |
| `pick` | 43 | 2 | `[2, 10]` | `[2, 7, 10]` | 0.0699 | 0.1738 | 0.2818 | 0.2982 |
| `place` | 42 | 7 | `[2, 7]` | `[2, 7, 11]` | 0.0758 | 0.1667 | 0.2641 | 0.2458 |
| `place` | 43 | 7 | `[2, 7]` | `[2, 7, 11]` | 0.0782 | 0.1696 | 0.2748 | 0.2761 |
| `open` | 42 | 2 | `[2, 11]` | `[2, 7, 11]` | 0.0707 | 0.1554 | 0.2499 | 0.2667 |
| `open` | 43 | 2 | `[2, 11]` | `[2, 7, 11]` | 0.0746 | 0.1581 | 0.2411 | 0.2653 |
| `close` | 42 | 2 | `[2, 7]` | `[2, 7, 10]` | 0.0473 | 0.1160 | 0.2066 | 0.2461 |
| `close` | 43 | 2 | `[2, 9]` | `[2, 7, 9]` | 0.0430 | 0.1073 | 0.1963 | 0.2519 |
| `turn` | 42 | 2 | `[2, 9]` | `[2, 9, 11]` | 0.0910 | 0.1817 | 0.3089 | 0.2611 |
| `turn` | 43 | 9 | `[2, 9]` | `[2, 5, 9]` | 0.0808 | 0.1647 | 0.2873 | 0.2676 |

## Frozen-τ k_hat (hypothesis only)

| Skill | seed | k_hat(0.12) | k_hat(0.20) | k_hat(0.28) | vs global extra-3 |
| --- | ---: | ---: | ---: | ---: | --- |
| `pick` | 42 | 1 | 2 | 3 | `{2,7,11}` |
| `pick` | 43 | 1 | 2 | 2 | [2, 7, 10] |
| `place` | 42 | 1 | 2 | 3 | `{2,7,11}` |
| `place` | 43 | 1 | 2 | 3 | `{2,7,11}` |
| `open` | 42 | 1 | 2 | 3 | `{2,7,11}` |
| `open` | 43 | 1 | 2 | 3 | `{2,7,11}` |
| `close` | 42 | 2 | 2 | 3 | [2, 7, 10] |
| `close` | 43 | 2 | 3 | 3 | [2, 7, 9] |
| `turn` | 42 | 1 | 2 | 2 | [2, 9, 11] |
| `turn` | 43 | 1 | 2 | 2 | [2, 5, 9] |

## Readout (pre-registered, not tuned on closed-loop)

- τ=0.12: almost all skills extra-1.
- τ=0.20: extra-2 if Δ(k=2)<0.20 else extra-1.
- τ=0.28: extra-3 unless Turn-like Δ(k=3) exceeds 0.28.
- Compare later to Wave A Goal7 k=1/2/3. Do **not** export k_hat(turn) onto Long2.

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop6_extra3/greedy_ppl_extra3_skill/k_curve.json`
