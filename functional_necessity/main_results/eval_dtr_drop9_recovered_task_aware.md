# Closed-loop LIBERO results (Drop-9 recovered π0.5：task-aware extra Drop-1, no second recovery)

Task-conditional skip `extra_drop = f(task)` on remaining LLM blocks of `s1ghhh/VLADrop_pi05_LIBERO_Drop9Block_GateProbe` (Stage-1 Drop-9 + 30K recovery). Format follows DTR Table 3 (Spatial / Object / Goal / Long / Avg.).

**Protocol (same as extra Drop-1 / Drop-3; not DTR’s 20-trial + recovery FT).** OpenPI serve `pi05_libero_dropped` + LIBERO client；4 suites × 10 tasks × 10 trials（400 episodes）；`seed=7`；replan=5；action horizon=10；wait=10；render 256、flip 180°、pad-resize 224。Skip is DTR Appendix B at runtime (keep prefix K/V, drop Q / attention / O / MLP). No second recovery. Long = LIBERO-10. Stage-1 drop is always `[7, 10, 11, 12, 13, 14, 15, 16, 17]`; each task adds exactly one extra remaining layer from `[0, 1, 2, 3, 4, 5, 6, 8, 9]`. Serve-time skip updates both `llm_drop_*_list` and HuggingFace prefix `GemmaDecoderLayer.drop_attn` / `drop_mlp`.

**Oracle map is leaky** (layer chosen on the same seed-7 10-trial LOO SR). Default extra is L2; switch only if another remaining layer has **strictly** more successes (ties keep L2). Shuffled map is a Fisher–Yates permutation of the same 40 extras (seed=7): same layer multiset, wrong tasks. Neither row is a fair method number.

Source: `dtr_drop9_recovered/results/eval_task_extra/{oracle_k1,oracle_k1_shuffled}/`（this machine）；other-server suite totals reported below（same maps / protocol；per-task JSON not copied here）。Maps: `dtr_drop9_recovered/results/task_layer/{map_oracle_k1,map_oracle_k1_shuffled,heatmap}.json`；LOO table: `dtr_drop9_recovered/results/eval_drop1/l{0,1,2,3,4,5,6,8,9}/`；full-skeleton and global L2: `eval_dtr_drop9_recovered.md`.

Table: Task-aware extra Drop-1 on Drop-9 recovered π0.5, no second recovery. 10 episodes per task, seed 7. Long = LIBERO-10. Recovered / global L2 anchors are this-machine extra Drop-1 jobs. Task-aware rows are two independent closed-loop repeats of the same leaky maps.

| Setting | Run | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Drop-9 recovered | this machine | 96.0 | 100.0 | 91.0 | 93.0 | 95.00 |
| Extra Drop-1, global L2 | this machine | 93.0 | 96.0 | 81.0 | 78.0 | 87.00 |
| Extra Drop-1, task-aware oracle k=1 | this machine | 91.0 | 94.0 | 88.0 | 78.0 | 87.75 |
| Extra Drop-1, task-aware oracle k=1 | other server | 88.0 | 99.0 | 90.0 | 82.0 | 89.75 |
| Extra Drop-1, shuffled oracle k=1 | this machine | 82.0 | 96.0 | 73.0 | 78.0 | 82.25 |
| Extra Drop-1, shuffled oracle k=1 | other server | 88.0 | 98.0 | 70.0 | 75.0 | 82.75 |

LOO-reconstructed oracle (not a new eval; sum of per-task best from the nine extra Drop-1 jobs): **94.75%**. Neither closed-loop oracle reproduces that stitching (87.75 / 89.75).

Δ Avg. vs Drop-9 recovered (95.00): global L2 −8.00；oracle −7.25 / −5.25；shuffled −12.75 / −12.25（this machine / other server）。
Δ Avg. vs global L2 (87.00): oracle **+0.75 / +2.75**；shuffled **−4.75 / −4.25**。
Oracle − shuffled: **+5.50 / +7.00** Avg. Pairing replicates on both machines; compression vs global L2 does not.

Oracle extra histogram (40 tasks): L2×26, L1×7, L5×3, L9/L6/L4/L3×1. Unique extras served: `{1,2,3,4,5,6,9}` (never L0/L8). 14/40 tasks leave L2.

## Closed-loop vs LOO table on the 14 switch tasks

Cell = successes / 10. `LOO extra` / `LOO L2` come from the original extra Drop-1 jobs. `CL oracle` / `CL shuffled` are **this machine** only; the other server reported suite totals, not per-task bits.

| id | extra | LOO extra | LOO L2 | CL oracle | CL shuffled | task |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Spa5 | 1 | 9 | 4 | 9 | 6 | pick up the black bowl on the ramekin and place it on the plate |
| Obj1 | 1 | 10 | 9 | 8 | 8 | pick up the cream cheese and place it in the basket |
| Obj2 | 5 | 10 | 9 | 10 | 10 | pick up the salad dressing and place it in the basket |
| Obj5 | 5 | 10 | 9 | 7 | 10 | pick up the tomato sauce and place it in the basket |
| Obj9 | 1 | 10 | 9 | 10 | 10 | pick up the orange juice and place it in the basket |
| Goal0 | 9 | 9 | 1 | 8 | 0 | open the middle drawer of the cabinet |
| Goal6 | 1 | 10 | 9 | 10 | 10 | put the cream cheese in the bowl |
| Goal7 | 1 | 9 | 6 | 7 | 9 | turn on the stove |
| Goal9 | 1 | 8 | 7 | 6 | 10 | put the wine bottle on the rack |
| Long2 | 1 | 10 | 9 | 9 | 9 | turn on the stove and put the moka pot on it |
| Long3 | 5 | 10 | 9 | 10 | 9 | put the black bowl in the bottom drawer of the cabinet and close it |
| Long5 | 6 | 10 | 9 | 10 | 10 | pick up the book and place it in the back compartment of the caddy |
| Long6 | 4 | 9 | 4 | 8 | 5 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| Long7 | 3 | 10 | 9 | 7 | 10 | put both the alphabet soup and the cream cheese box in the basket |

Predicted gain of the 14 switches vs LOO L2: **+31** successes. Closed-loop realized **+16** on those tasks vs the same LOO L2 numbers. The 26 stay-L2 tasks scored 232/260 here vs 245/260 in the dedicated L2 job (−13). Net closed-loop vs global L2 is therefore only +3 successes / 400 (**+0.75 Avg.**).

Pairing is real on the headline tasks: Goal0 extra L9 is 8/10 vs shuffled extra L2 0/10；Spa5 extra L1 is 9/10 vs shuffled L2 6/10；Long6 extra L4 is 8/10 vs shuffled L2 5/10.

## Extra / union indices (examples)

Stage-1 ∪ extra；only LLM Q/attn/O/MLP skipped.

| Task | Extra | Union drop |
| --- | ---: | --- |
| default (26 tasks) | [2] | [2, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| Spa5 / several Object+Goal | [1] | [1, 7, 10, 11, 12, 13, 14, 15, 16, 17] |
| Goal0 | [9] | [7, 9, 10, 11, 12, 13, 14, 15, 16, 17] |
| Long6 | [4] | [4, 7, 10, 11, 12, 13, 14, 15, 16, 17] |

Full 40-row map: `dtr_drop9_recovered/results/task_layer/map_oracle_k1.json`.

## Checks

This machine: both jobs wrote complete `eval_info.json` (`n_suites=4`). Each suite 100 episodes; `pc_success` matches raw successes. Every `per_task` row records `extra_drop`. Server logs show prefix flags tracking the extra (`Goal0`: `extra_drop=[9] prefix_flags=[7, 9, 10, …, 17]`).

Other server: suite totals 88/99/90/82 avg=89.75 and 88/98/70/75 avg=82.75 (means check: (88+99+90+82)/4=89.75, (88+98+70+75)/4=82.75). Same leaky maps and protocol; raw per-task JSON not stored in this repo.

Client seed=7, `num_trials=10`, no second recovery. Oracle Avg 87.75 / 89.75 is a **wiring / pairing** result, not a fair skill-aware compressor. Fair E4 must re-select `extra_drop` from calibration bypass Δloss (or another score split from this SR), then shuffle that map.
