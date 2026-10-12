# Closed-loop LIBERO results (OpenVLA-OFT Drop-4 / Drop-8, no recovery)

Importance metrics for whole-layer dropping on `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` / LIBERO.
Format follows DTR Table 3 (Spatial / Object / Goal / Long / Avg.).

**Protocol.** Official OFT eval: seed 7, center crop, L1 action head, 2 images + proprio, chunk 8, four suites × 10 tasks × 50 episodes (500 episodes), no video, no recovery fine-tune. Skip is a whole Llama decoder identity (no Q/K/V/O/MLP). Layers are Llama-2 0–31. High score = keep. Drop-k is the k lowest scores. Identical drop sets share one 500-episode run (†). Greedy PPL is stepwise conditional ΔL1 on 512 calibration samples; Drop-k is the first k picks, not the independent-PPL lowest-k.

Anchor is this machine's seed-7 full model: Spatial 466/500, Object 496/500, Goal 486/500, Long 471/500 (93.2 / 99.2 / 97.2 / 94.2, avg 95.95). Not the paper's 3-seed 96.8.

Source: `results/openvla_oft_libero/drop{4,8}/eval/k{4,8}_*/eval_info.json`. Ranks: `results/openvla_oft_libero/drop{4,8}/ranks/ranks.json`. Scores: `results/openvla_oft_libero/drop8/scores/*.pt`. Layer-0-excluded Drop-4: `results/openvla_oft_libero/drop4_nol0/eval/`. Greedy PPL: `results/openvla_oft_libero/greedy_ppl_eval/eval/k{3,4,8}_greedy_ppl/*/eval_info.json` and `results/openvla_oft_libero/ppl_subset_search/oft_drop8_greedy.json`.

Drop-4 CKA Long was stopped on GPU 1 before 500 episodes. The table marks that cell 0.0 together with the three finished CKA suites, which are measured 0/500.

Table: success rate (%). Long = LIBERO-10.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 93.2 | 99.2 | 97.2 | 94.2 | 95.95 |
| Drop-4 (4/32) | Fisher | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | IGIA | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Greedy PPL | 89.8 | 98.0 | 91.8 | 87.6 | 91.80 |
|  | PPL † | 85.2 | 99.2 | 73.0 | 89.0 | 86.60 |
|  | GateProbe | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Hessian | 11.6 | 0.6 | 5.6 | 7.2 | 6.25 |
|  | CKA | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | CosSim † | 85.2 | 99.2 | 73.0 | 89.0 | 86.60 |
|  | Taylor | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Magnitude | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| Drop-8 (8/32) | Fisher † | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | IGIA † | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Greedy PPL | 72.2 | 92.2 | 55.8 | 55.0 | 68.80 |
|  | CosSim | 69.0 | 96.6 | 39.0 | 63.0 | 66.90 |
|  | PPL | 60.8 | 72.8 | 8.6 | 25.6 | 41.95 |
|  | GateProbe | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Hessian | 3.8 | 0.0 | 0.4 | 1.0 | 1.30 |
|  | CKA | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Taylor | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Magnitude | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |

† Drop-4 PPL and CosSim are the same set `[3, 4, 5, 6]`. Drop-8 Fisher and IGIA are the same set `[0, 3, 4, 5, 28, 29, 30, 31]`. Greedy PPL Drop-4 `[3, 4, 5, 29]` and Drop-8 `[3, 4, 5, 19, 23, 26, 28, 29]` are not the independent-PPL sets.

Δ Avg. vs this-machine anchor (95.95), completed rows only:

- Greedy PPL Drop-3: −1.90.
- Drop-4: Greedy PPL −4.15, CosSim/PPL −9.35, Hessian −89.70, Fisher / IGIA / GateProbe / CKA / Taylor / Magnitude −95.95.
- Drop-8: Greedy PPL −27.15, CosSim −29.05, PPL −54.00, Hessian −94.65, Fisher/IGIA / GateProbe / CKA / Taylor / Magnitude −95.95.

## Greedy PPL prefixes

Pick order `[3, 5, 29, 4, 23, 26, 19, 28]` from `oft_drop8_greedy.json` (baseline L1 0.037880; greedy Drop-8 ΔL1 +0.054602). All three prefixes are 4/4 suites.

| k | Drop | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | [3, 5, 29] | 90.8 | 97.6 | 95.6 | 92.2 | 94.05 |
| 4 | [3, 4, 5, 29] | 89.8 | 98.0 | 91.8 | 87.6 | 91.80 |
| 8 | [3, 4, 5, 19, 23, 26, 28, 29] | 72.2 | 92.2 | 55.8 | 55.0 | 68.80 |

Greedy Drop-4 vs independent PPL Drop-4: +5.20 Avg. Greedy Drop-8 vs independent PPL Drop-8: +26.85 Avg. Neither prefix contains layer 0.

## Drop-4 excluding layer 0

Same protocol. The set is the four lowest scores after removing layer 0 from the redundancy order. Fisher and IGIA finished 4/4 suites on GPU 0 and GPU 1. GateProbe is the same set as CosSim/PPL, so it reuses that run. Magnitude and Taylor were stopped after Spatial and Object.

| Metric | Drop | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fisher | [3, 4, 30, 31] | 25.8 | 11.8 | 7.0 | 15.4 | 15.00 |
| IGIA | [3, 29, 30, 31] | 15.2 | 1.4 | 6.4 | 7.6 | 7.65 |
| GateProbe | [3, 4, 5, 6] | 85.2 | 99.2 | 73.0 | 89.0 | 86.60 |
| Magnitude | [1, 2, 3, 6] | 0.0 | 0.0 |  |  |  |
| Taylor | [1, 2, 3, 4] | 0.0 | 0.0 |  |  |  |

Δ Avg. vs anchor for the finished rows: GateProbe −9.35, Fisher −80.95, IGIA −88.30.

## Drop indices (Llama 0–31)

Recomputed from `scores/*.pt` with the same high-score-keep ranking. Each `eval_info.json` and `drop.json` matches that list. Greedy PPL lists come from the subset search, not `ranks.json`.

### Drop-4

| Metric | Drop |
| --- | --- |
| Greedy PPL | [3, 4, 5, 29] |
| Fisher | [0, 3, 4, 30] |
| IGIA | [0, 29, 30, 31] |
| PPL | [3, 4, 5, 6] |
| GateProbe | [0, 3, 4, 5] |
| Hessian | [28, 29, 30, 31] |
| CKA | [0, 2, 18, 19] |
| CosSim | [3, 4, 5, 6] |
| Taylor | [0, 1, 2, 3] |
| Magnitude | [0, 1, 3, 6] |

### Drop-8

| Metric | Drop |
| --- | --- |
| Greedy PPL | [3, 4, 5, 19, 23, 26, 28, 29] |
| Fisher | [0, 3, 4, 5, 28, 29, 30, 31] |
| IGIA | [0, 3, 4, 5, 28, 29, 30, 31] |
| PPL | [2, 3, 4, 5, 6, 23, 26, 29] |
| GateProbe | [0, 3, 4, 5, 6, 23, 26, 27] |
| Hessian | [24, 25, 26, 27, 28, 29, 30, 31] |
| CKA | [0, 2, 15, 17, 18, 19, 23, 24] |
| CosSim | [3, 4, 5, 6, 19, 20, 23, 28] |
| Taylor | [0, 1, 2, 3, 4, 5, 30, 31] |
| Magnitude | [0, 1, 2, 3, 4, 6, 7, 8] |

## Checks

Finished jobs: Drop-8 32/32 and Drop-4 31/32. Drop-4 CKA Long has no `eval_info.json`; GPU 1 was stopped during that suite, and the table records 0.0 for it. Greedy PPL Drop-3 / Drop-4 / Drop-8 are all 4/4 suites (`EXIT:0`). Drop-3 Long is 461/500. Each finished log has one `EVAL-*.txt`, 10 tasks, 50 episodes per task, 500 `Success` lines, and `Total episodes` / `Total successes` equal to `eval_info.json`. EGL destructor tracebacks occur after the 500-episode total is written and do not change the count.

Hessian importance is 0 on all 32 layers, so its Drop-4/Drop-8 lists are the stable tie-break (highest indices), not a measured ranking. CKA `1−CKA` for almost every layer sits near `1e-5`, so that ranking is not resolved except that layer 1 is kept. Both were still evaluated on the lists above.
