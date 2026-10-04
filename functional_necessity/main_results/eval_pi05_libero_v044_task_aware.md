# Closed-loop LIBERO results (v044 Drop-6 + task-aware extra Drop-3, no recovery)

Task-conditional skip `drop = U ∪ extra3(task)` on `lerobot/pi05_libero_finetuned_v044`. Format follows DTR Table 3 (Spatial / Object / Goal / Long / Avg.).

**Protocol (same as this-machine Drop-6 / Drop-9; not DTR’s 20-trial + recovery FT).** Official v044 eval: `n_action_steps=10`, `compile_model=false`, `seed=1000`, four suites × 10 tasks × 10 episodes (400 episodes), `batch_size=1`, no video. Skip is DTR Appendix B at runtime (keep prefix K/V, drop Q / attention / O / MLP). No recovery fine-tune. Long = LIBERO-10.

**Universal skip is locked** `U = [1, 13, 14, 15, 16, 17]` (global Fisher Drop-6). Each task adds exactly three extra layers from the remaining 12 `[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`. Extra-3 is chosen on the already-skipped Drop-6 graph (per-task 16×8 calib, seed 42): Fisher OBD lowest-3, or greedy PPL (frozen ξ). Shuffled maps are Fisher–Yates permutations of the same 40 triples (`shuffle_seed=1000`): same multiset, wrong tasks. Maps are calibration scores, not eval SR.

Source: `results/eval_task_extra/{fisher_extra3,fisher_extra3_shuffled,greedy_ppl_extra3,greedy_ppl_extra3_shuffled,greedy_ppl_extra3_global,greedy_ppl_extra3_global_r2,greedy_ppl_extra3_suite,greedy_ppl_extra3_suite_r2}/`. Maps: `results/drop6_extra3/maps/`. Anchors: `eval_pi05_libero_v044.md`.

Table: Task-aware extra Drop-3 on Drop-6 π0.5-LIBERO, no recovery. 10 episodes per task, seed 1000. Long = LIBERO-10.

| Setting | Spatial | Object | Goal | Long | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 97.0 | 100.0 | 95.0 | 97.0 | 97.25 |
| Drop-6 Fisher (`U` only) | 97.0 | 100.0 | 96.0 | 98.0 | 97.75 |
| Drop-9 Fisher (`U ∪ {2,3,10}`) | 89.0 | 87.0 | 53.0 | 78.0 | 76.75 |
| Drop-9 Greedy PPL (global 9-set) | 50.0 | 72.0 | 22.0 | 44.0 | 47.00 |
| Extra-3 Fisher `f(task)` | 61.0 | 60.0 | 30.0 | 21.0 | 43.00 |
| Extra-3 Fisher shuffled | 58.0 | 65.0 | 32.0 | 23.0 | 44.50 |
| Extra-3 greedy PPL `f(task)` | 79.0 | 91.0 | 63.0 | 84.0 | 79.25 |
| Extra-3 greedy PPL shuffled | 84.0 | 89.0 | 49.0 | 62.0 | 71.00 |
| Extra-3 greedy PPL global (r1) | 85.0 | 86.0 | 59.0 | 86.0 | 79.00 |
| Extra-3 greedy PPL global (r2) | 86.0 | 87.0 | 64.0 | 86.0 | 80.75 |
| Extra-3 greedy PPL global (mean) | 85.5 | 86.5 | 61.5 | 86.0 | 79.88 |
| Extra-3 greedy PPL suite-aware (r1) | 89.0 | 93.0 | 60.0 | 85.0 | 81.75 |
| Extra-3 greedy PPL suite-aware (r2) | 85.0 | 94.0 | 63.0 | 87.0 | 82.25 |
| Extra-3 greedy PPL suite-aware (mean) | 87.0 | 93.5 | 61.5 | 86.0 | 82.00 |

Raw: Fisher `f(task)` 172/400；Fisher shuffled 178/400；greedy `f(task)` 317/400；greedy shuffled 284/400；greedy global 316/400 and 323/400；greedy suite 327/400 and 329/400.

Same maps, same `seed=1000`. Global r2 − r1 = **+1.75** Avg；suite r2 − r1 = **+0.50**. Per-task n/10 differs on 12/40 (global) and 17/40 (suite). Object suite − global is **+7** on both runs.

Δ Avg. vs Drop-9 Fisher (76.75), using two-run means: greedy global **+3.13**；greedy suite **+5.25**.

Δ Avg. vs Drop-6 Fisher (97.75): greedy global mean −17.87；greedy suite mean −15.75.

`f(task)` − shuffled: Fisher **−1.50**；greedy **+8.25** (Goal +14, Long +22; Spatial −5).

Greedy pooling vs `f(task)` 79.25: global mean **+0.63**；suite mean **+2.75**. Suite mean − global mean = **+2.12**.

## Extra-3 histograms

Fisher (6 triples; L0 never; L7 on 3/40): `{3,4,12}`×31, `{3,4,7}`×3, `{4,10,12}`×3, `{3,10,12}` / `{4,9,12}` / `{3,4,9}`×1. No task uses global Drop-9 extra `{2,3,10}`. Shuffled same multiset, 23/40 tasks keep the original triple.

Greedy PPL (18 triples; L0 never; L7 on 22/40): mode `{2,7,11}`×12. Shuffled same multiset, 5/40 tasks keep the original triple.

## Default union (9 layers)

| Map | Default extra-3 | Union drop |
| --- | --- | --- |
| Fisher 31/40 | `[3, 4, 12]` | `[1, 3, 4, 12, 13, 14, 15, 16, 17]` |
| Greedy 12/40 | `[2, 7, 11]` | `[1, 2, 7, 11, 13, 14, 15, 16, 17]` |
| Greedy global 40/40 | `[2, 7, 11]` | `[1, 2, 7, 11, 13, 14, 15, 16, 17]` |
| Greedy suite Object | `[2, 8, 11]` | `[1, 2, 8, 11, 13, 14, 15, 16, 17]` |
| Global Fisher Drop-9 | `[2, 3, 10]` | `[1, 2, 3, 10, 13, 14, 15, 16, 17]` |

## Checks

Task-aware four jobs plus greedy global / suite (r1 and r2) all wrote complete `eval_info.json` and ended `EXIT=0`. Each suite 100 episodes, overall 400. `pc_success` matches raw success bits. Every `per_task` row records `extra_drop` and `union_drop` equal to its map; every union is `U ∪ extra` (length 9); extras stay in the remaining 12. `compile_model=false`, `n_action_steps=10`, `seed=1000`, `recovery=false`, no videos. Skip is runtime `skip_llm_layers` on prefix `layers[i].forward` (DTR Appendix B). Shuffled extras are a permutation of the `f(task)` extras (Fisher 23/40 identical by chance; greedy 5/40). Greedy global uses `{2,7,11}` on all 40 tasks; suite uses that triple on Spatial/Goal/Long and `{2,8,11}` on Object. Repeat jobs used the same maps and seed; output dirs are `*_r2`.

Fisher task-aware is **not** a routing gain on the 76.75% Drop-9 set: the conditional extra-3 is a different 9-layer graph, and shuffled is not worse. Greedy pairing is real vs its own shuffle. Constant `{2,7,11}` two-run mean **79.88** matches `f(task)` 79.25 within eval noise. Suite-aware two-run mean **82.00** is the best extra-3 so far (Object +7 vs global both runs), still about −16 vs Drop-6. Same-seed repeats move Avg by ≤1.75; do not treat a 1–2 point suite−global gap as significant except the Object +7.
