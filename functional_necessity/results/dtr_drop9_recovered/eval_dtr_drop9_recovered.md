# Closed-loop LIBERO results (Drop-9 recovered π0.5：extra Drop-1 / Drop-3, no second recovery)

Importance metrics / leave-one-out on remaining LLM blocks of `s1ghhh/VLADrop_pi05_LIBERO_Drop9Block_GateProbe` (Stage-1 Drop-9 + 30K recovery). Format follows DTR Table 3 (Spatial / Object / Goal / Long / Avg.).

**Protocol (not DTR’s 20-trial + recovery FT).** OpenPI serve `pi05_libero_dropped` + LIBERO client；4 suites × 10 tasks × 10 trials（400 episodes）；`seed=7`；replan=5；action horizon=10；wait=10；render 256、flip 180°、pad-resize 224。Skip is DTR Appendix B at runtime (keep prefix K/V, drop Q / attention / O / MLP). No second recovery. Long = LIBERO-10. Extra drops are taken only from remaining `[0, 1, 2, 3, 4, 5, 6, 8, 9]`; Stage-1 drop is always `[7, 10, 11, 12, 13, 14, 15, 16, 17]`. Independent Fisher and independent PPL share the same extra Drop-3 set `[2, 4, 9]` and are merged into one row. Greedy PPL (which matches C(9,3) exhaustive) yields a different extra set `[1, 2, 9]` and is a separate row. Extra Drop-1 is leave-one-out over all 9 remaining layers (not only each metric’s top-1).

Source: `dtr_drop9_recovered/results/eval_drop3/{drop9_recovered,greedy_ppl,fisher_ppl,hessian,gateprobe,cka,cossim,igia,taylor,magnitude}/`；`dtr_drop9_recovered/results/eval_drop1/l{0,1,2,3,4,5,6,8,9}/`。Ranks: `dtr_drop9_recovered/results/ranks/ranks.json`。PPL subset search: `dtr_drop9_recovered/results/ppl_subset_search/drop3_search.md`.

Table: Extra Drop-1 / Drop-3 on Drop-9 recovered π0.5, no second recovery. 10 episodes per task, seed 7. Long = LIBERO-10.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Drop-9 recovered | — | 96.0 | 100.0 | 91.0 | 93.0 | 95.00 |
| Extra Drop-1 (10/18) | L2 | 93.0 | 96.0 | 81.0 | 78.0 | 87.00 |
|  | L1 | 90.0 | 94.0 | 78.0 | 58.0 | 80.00 |
|  | L5 | 76.0 | 97.0 | 72.0 | 60.0 | 76.25 |
|  | L6 | 74.0 | 92.0 | 58.0 | 49.0 | 68.25 |
|  | L4 | 73.0 | 78.0 | 46.0 | 64.0 | 65.25 |
|  | L9 | 66.0 | 89.0 | 56.0 | 48.0 | 64.75 |
|  | L3 | 68.0 | 81.0 | 39.0 | 53.0 | 60.25 |
|  | L8 | 60.0 | 86.0 | 47.0 | 39.0 | 58.00 |
|  | L0 | 0.0 | 0.0 | 0.0 | 0.0† | 0.00 |
| Extra Drop-3 (12/18) | Greedy / exhaustive PPL | 29.0 | 23.0 | 18.0 | 5.0 | 18.75 |
|  | Fisher / PPL | 17.0 | 13.0 | 11.0 | 1.0 | 10.50 |
|  | Hessian | 12.0 | 0.0 | 9.0 | 0.0 | 5.25 |
|  | GateProbe | 5.0 | 5.0 | 3.0 | 0.0 | 3.25 |
|  | CKA | 6.0 | 0.0 | 3.0 | 0.0 | 2.25 |
|  | CosSim | 4.0 | 4.0 | 0.0 | 0.0 | 2.00 |
|  | IGIA | 3.0 | 0.0 | 3.0 | 0.0 | 1.50 |
|  | Taylor | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Magnitude | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |

† `L0` Long 为终止后记 0：Spatial / Object / Goal 实测 0/100，Long 进行中亦为 0%，未跑完 100 episodes。其余 Extra Drop-1 / Drop-3 均为 400 episodes。

Δ Avg. vs Drop-9 recovered（95.00）:

- Extra Drop-1: L2 −8.00, L1 −15.00, L5 −18.75, L6 −26.75, L4 −29.75, L9 −30.25, L3 −34.75, L8 −37.00, L0 −95.00.
- Extra Drop-3: Greedy / exhaustive PPL −76.25, Fisher / PPL −84.50, Hessian −89.75, GateProbe −91.75, CKA −92.75, CosSim −93.00, IGIA −93.50, Taylor / Magnitude −95.00.

各指标 extra Drop-1 top-1 对应：PPL→L2（87.00）；Taylor / Magnitude→L0（0.00）；CosSim→L4（65.25）；GateProbe / CKA→L3（60.25）；Fisher / IGIA / Hessian→L9（64.75）。即时必要性（ΔAvg 从大到小）：`0 ≫ 8 > 3 > 9 ≈ 4 > 6 > 5 > 1 > 2`。

论文同 checkpoint Drop-9 GateProbe（50 trials/task）：97.0 / 98.2 / 92.0 / 88.8 / 94.0。本机 10-trial 锚点为 95.00。

## Extra drop / keep indices (PaliGemma 18 layers, 0–17)

### Extra Drop-1

| Metric | Extra drop | Remaining keep | Union drop |
| --- | --- | --- | --- |
| —（baseline） | [] | [0, 1, 2, 3, 4, 5, 6, 8, 9] | [7, 10, 11, 12, 13, 14, 15, 16, 17] |
| L0 | [0] | [1, 2, 3, 4, 5, 6, 8, 9] | [0, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| L1 | [1] | [0, 2, 3, 4, 5, 6, 8, 9] | [1, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| L2 | [2] | [0, 1, 3, 4, 5, 6, 8, 9] | [2, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| L3 | [3] | [0, 1, 2, 4, 5, 6, 8, 9] | [3, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| L4 | [4] | [0, 1, 2, 3, 5, 6, 8, 9] | [4, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| L5 | [5] | [0, 1, 2, 3, 4, 6, 8, 9] | [5, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| L6 | [6] | [0, 1, 2, 3, 4, 5, 8, 9] | [6, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| L8 | [8] | [0, 1, 2, 3, 4, 5, 6, 9] | [7, 8, 10, 11, 12, 13, 14, 15, 16, 17] |
| L9 | [9] | [0, 1, 2, 3, 4, 5, 6, 8] | [7, 9, 10, 11, 12, 13, 14, 15, 16, 17] |

### Extra Drop-3

| Metric | Extra drop | Remaining keep | Union drop |
| --- | --- | --- | --- |
| —（baseline） | [] | [0, 1, 2, 3, 4, 5, 6, 8, 9] | [7, 10, 11, 12, 13, 14, 15, 16, 17] |
| Greedy / exhaustive PPL | [1, 2, 9] | [0, 3, 4, 5, 6, 8] | [1, 2, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17] |
| Fisher / PPL | [2, 4, 9] | [0, 1, 3, 5, 6, 8] | [2, 4, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17] |
| Hessian | [2, 5, 9] | [0, 1, 3, 4, 6, 8] | [2, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17] |
| GateProbe | [3, 8, 9] | [0, 1, 2, 4, 5, 6] | [3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] |
| CKA | [3, 4, 6] | [0, 1, 2, 5, 8, 9] | [3, 4, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| CosSim | [3, 4, 5] | [0, 1, 2, 6, 8, 9] | [3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| IGIA | [5, 8, 9] | [0, 1, 2, 3, 4, 6] | [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] |
| Taylor | [0, 2, 9] | [1, 3, 4, 5, 6, 8] | [0, 2, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17] |
| Magnitude | [0, 5, 9] | [1, 2, 3, 4, 6, 8] | [0, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17] |

## Checks

Drop-9 recovered baseline、9 组 unique Extra Drop-3（8 个独立指标集合 + greedy/exhaustive PPL `[1, 2, 9]`）、以及 Extra Drop-1 中 L1–L9 均写有 `eval_info.json`；每项 4 suites × 100 episodes，`pc_success` 与 raw `successes` 一致。只跳 LLM attention+MLP；expert / vision 不跳。`L0` 的 Spatial / Object / Goal 为实测 0/100，Long 按上述规则记 0。Greedy PPL 与独立 PPL 闭环差 8.25 Avg（18.75 vs 10.50）：loss 更优的三元组闭环也更好，但仍远低于 recovered 锚点 95.00。
