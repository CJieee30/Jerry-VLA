# LIBERO-plus：π0.5 full vs Fisher Drop-6

π0.5-LIBERO-finetuned (`lerobot/pi05_libero_finetuned_v044`) vs Stage-1 Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`, skip keep-KV. Official Plus recipe on this machine: 1 trial per variant, `seed=1000`, `n_action_steps=10`, `num_inference_steps=10`, `compile_model=false`. In-distribution LIBERO 400-ep anchors: full **97.25**, Drop-6 **97.75** (`eval_pi05_libero_v044.md`). Do not mix those with this table.

Avg. is the unweighted mean of four suite rates (Spatial / Object / Goal / Long). Sensor Noise is still running on GPU0/2 (Spatial json exists; Object is in progress). Background Textures finished on GPU1/3.

## Core: seven Plus dimensions (Avg. %)

| Dimension | Full | Drop-6 | Δ |
| --- | ---: | ---: | ---: |
| Language Instructions | 89.20 | 85.24 | −3.96 |
| Robot Initial States | 77.69 | 74.18 | −3.51 |
| Camera Viewpoints | 66.85 | 60.68 | −6.18 |
| Sensor Noise |  |  |  |
| Objects Layout | 86.52 | 82.41 | −4.11 |
| Light Conditions | 98.47 | 95.01 | −3.46 |
| Background Textures | 96.62 | 92.36 | −4.26 |

Finished six-dim mean (not a Plus overall): full **85.89**, Drop-6 **81.65**, **−4.24**. Drop-6 does not collapse on any finished slice. Camera is the largest dim gap (−6.18). Largest single-suite holes: Background Goal (−14.59), Light Long (−13.50), Layout Long (−12.82).

## Core: finished dims by suite (%)

| Dimension | Setting | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Language | Full | 99.74 | 91.24 | 73.90 | 91.91 | 89.20 |
| Language | Drop-6 | 95.64 | 89.27 | 71.71 | 84.33 | 85.24 |
| Init | Full | 86.57 | 72.61 | 77.26 | 74.30 | 77.69 |
| Init | Drop-6 | 88.00 | 68.84 | 70.42 | 69.47 | 74.18 |
| Camera | Full | 74.73 | 73.99 | 66.67 | 52.03 | 66.85 |
| Camera | Drop-6 | 70.48 | 63.13 | 64.95 | 44.15 | 60.68 |
| Layout | Full | 96.88 | 88.83 | 68.71 | 91.67 | 86.52 |
| Layout | Drop-6 | 96.36 | 87.84 | 66.59 | 78.85 | 82.41 |
| Light | Full | 100.00 | 99.33 | 97.85 | 96.72 | 98.47 |
| Light | Drop-6 | 99.66 | 99.33 | 97.85 | 83.21 | 95.01 |
| Background | Full | 99.61 | 98.79 | 93.95 | 94.12 | 96.62 |
| Background | Drop-6 | 99.61 | 98.79 | 79.36 | 91.70 | 92.36 |

Counts: Language 1367/1537 vs 1306/1537; Init 1200/1550 vs 1143/1550; Camera 1064/1599 vs 965/1599; Layout 1309/1525 vs 1254/1525; Light 1125/1142 vs 1087/1142; Background 1038/1076 vs 990/1076.

Per-dimension protocol, disagreement, and jobs are in the sections below. Language 89.20 / 85.24 and Init 77.69 / 74.18 are not overwritten.

---

# LIBERO-plus Language Instructions（π0.5 Fisher Drop-6 vs full）

π0.5-LIBERO-finetuned (`lerobot/pi05_libero_finetuned_v044`) on the official **Language Instructions** slice of LIBERO-plus. This is not in-distribution LIBERO, and it is not the other six Plus dimensions (background, camera, light, layout, noise, init state).

**Protocol.** Official Plus language-dim recipe on this machine: 1 trial per language variant (`n_episodes=1`), `seed=1000`, `n_action_steps=10`, `num_inference_steps=10`, `compile_model=false`, `batch_size=1`, `control_mode=relative`, no video, no recovery. Task ids come from `LIBERO-plus/.../task_classification.json` category `Language Instructions` (1-based ids minus 1). Skip is DTR Appendix B keep-KV (`dtr_appendix_b_prefix_kv`). Fisher Drop-6 is the Stage-1 set `[1, 13, 14, 15, 16, 17]`. GPU0 = full; GPU2 = Fisher Drop-6. All eight jobs `EXIT:0`.

In-distribution LIBERO 400-ep anchors (same checkpoint, same skip, `eval_pi05_libero_v044.md`): full **97.25**, Fisher Drop-6 **97.75**. Do not mix those 400-ep suite rates with this Plus table.

Source: `results/eval_libero_plus/plus_{full,fisher6}_lang_{spatial,object,goal,10}/{eval_info.json,drop_meta.json}`. Script: `scripts/pi05_libero/run_eval_plus_lang.sh`. Smoke `plus_smoke_full_lang_spatial10` (10 Spatial language tasks, 100%) is not in the table.

Table: Language-instruction success rate (%). Avg. is the unweighted mean of four suite rates (LIBERO table convention). Long = LIBERO-10. n is the number of language variants in that suite.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full | — | 99.74 | 91.24 | 73.90 | 91.91 | 89.20 |
| Drop-6 (6/18) | Fisher | 95.64 | 89.27 | 71.71 | 84.33 | 85.24 |

Counts (successes / language variants):

| Setting | Spatial | Object | Goal | Long | All |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 389/390 | 323/354 | 303/410 | 352/383 | 1367/1537 |
| Fisher Drop-6 | 373/390 | 316/354 | 294/410 | 323/383 | 1306/1537 |

Δ vs this-machine Plus full (macro Avg. **89.20** → **85.24**, **−3.96**):

- Spatial **−4.10** (389/390 → 373/390)
- Object **−1.98** (323/354 → 316/354)
- Goal **−2.20** (303/410 → 294/410)
- Long **−7.57** (352/383 → 323/383)
- Micro (all 1537 variants): 88.94 → 84.97, **−3.97**
- Spatial+Object+Goal only (before Long finished): macro **−2.76**

Drop-6 does not collapse on Plus language. The gap is a few points, concentrated on Long. Same operator on in-distribution LIBERO is +0.50 Avg.

## Drop / keep

| Setting | Drop | Keep |
| --- | --- | --- |
| Full | [] | [0–17] |
| Fisher Drop-6 | [1, 13, 14, 15, 16, 17] | [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |

## Same-variant disagreement (full vs Fisher Drop-6)

Each suite uses the same 1-based-minus-1 language task ids under `seed=1000`.

| Suite | n | both success | both fail | full✓ drop✗ | full✗ drop✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 390 | 373 | 1 | 16 | 0 |
| Object | 354 | 303 | 18 | 20 | 13 |
| Goal | 410 | 279 | 92 | 24 | 15 |
| Long | 383 | 318 | 26 | 34 | 5 |

Net extra failures under Drop-6: Spatial 16, Object 7, Goal 9, Long 29.

## Jobs

All `EXIT:0`. Times are UTC from `results/logs/eval_plus_*.log`.

| Job | GPU | Start | End |
| --- | ---: | --- | --- |
| plus_full_lang_spatial | 0 | 2026-10-10T03:00:57Z | 2026-10-10T04:34:33Z |
| plus_full_lang_object | 0 | 2026-10-10T04:34:33Z | 2026-10-10T06:12:10Z |
| plus_full_lang_goal | 0 | 2026-10-10T06:12:10Z | 2026-10-10T08:18:51Z |
| plus_full_lang_10 | 0 | 2026-10-10T08:18:51Z | 2026-10-10T11:15:02Z |
| plus_fisher6_lang_spatial | 2 | 2026-10-10T03:00:57Z | 2026-10-10T04:41:43Z |
| plus_fisher6_lang_object | 2 | 2026-10-10T04:41:43Z | 2026-10-10T06:22:43Z |
| plus_fisher6_lang_goal | 2 | 2026-10-10T06:22:43Z | 2026-10-10T08:22:40Z |
| plus_fisher6_lang_10 | 2 | 2026-10-10T08:22:40Z | 2026-10-10T11:26:55Z |

`eval_info.json` wall-clock (`eval_s`): full Spatial 5430 / Object 5665 / Goal 7416 / Long 10367; Drop-6 Spatial 5859 / Object 5873 / Goal 7015 / Long 10866. These are MuJoCo-dominated; do not use them as efficiency.

## Reading

Plus language is a paraphrased-instruction OOD slice, 1 trial per variant. Fisher Drop-6 keeps the same keep-KV operator that is in-distribution neutral (97.75 vs 97.25). Here the cost is **−3.96** Avg., **−7.57** on Long, and Spatial/Object/Goal stay within about 2–4 points. That is a measured language-OOD gap, not a closed-loop collapse, and it is not a new selector. Do not put Plus rates into the in-distribution operating-point table.

Per-dimension sections follow. Sensor Noise is empty in the core table until all four suite json files exist.

---

# LIBERO-plus Robot Initial States（π0.5 Fisher Drop-6 vs full）

Same checkpoint, skip, seed, and 1-trial-per-variant recipe as Language. Category is official **Robot Initial States** (1550 variants). GPU0 = full; GPU2 = Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`. All eight jobs `EXIT:0`. Do not overwrite Language 89.20 / 85.24. Do not mix with in-distribution 400-ep 97.25 / 97.75.

Source: `results/eval_libero_plus/plus_{full,fisher6}_init_{spatial,object,goal,10}/{eval_info.json,drop_meta.json}`. Script: `scripts/pi05_libero/run_eval_plus_dim.sh` (entry `run_eval_plus_lang.sh`, category `Robot Initial States`).

Table: Init-state success rate (%). Avg. is the unweighted mean of four suite rates. Long = LIBERO-10.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full | — | 86.57 | 72.61 | 77.26 | 74.30 | 77.69 |
| Drop-6 (6/18) | Fisher | 88.00 | 68.84 | 70.42 | 69.47 | 74.18 |

Counts (successes / init-state variants):

| Setting | Spatial | Object | Goal | Long | All |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 303/350 | 289/398 | 316/409 | 292/393 | 1200/1550 |
| Fisher Drop-6 | 308/350 | 274/398 | 288/409 | 273/393 | 1143/1550 |

Δ vs this-machine Plus-init full (macro Avg. **77.69** → **74.18**, **−3.51**):

- Spatial **+1.43** (303/350 → 308/350)
- Object **−3.77** (289/398 → 274/398)
- Goal **−6.85** (316/409 → 288/409)
- Long **−4.83** (292/393 → 273/393)
- Micro (all 1550 variants): 77.42 → 73.74, **−3.68**
- Spatial+Object+Goal only: macro **78.81** → **75.75**, **−3.06**

Init is harder than Language for both settings (full 89.20 → 77.69). Drop-6 does not collapse. The gap is similar in size to Language (−3.96), concentrated on Goal and Long; Spatial is slightly *up*. Same operator on in-distribution LIBERO is +0.50 Avg.

## Drop / keep

Same as Language: Full `[]`; Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`. Skip keep-KV.

## Same-variant disagreement (full vs Fisher Drop-6)

Each suite uses the same 1-based-minus-1 init-state task ids under `seed=1000`.

| Suite | n | both success | both fail | full✓ drop✗ | full✗ drop✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 350 | 283 | 22 | 20 | 25 |
| Object | 398 | 245 | 80 | 44 | 29 |
| Goal | 409 | 273 | 78 | 43 | 15 |
| Long | 393 | 236 | 64 | 56 | 37 |

Net extra failures under Drop-6: Spatial **−5** (Drop-6 succeeds more), Object 15, Goal 28, Long 19. All 1550: 57 extra failures (1200 → 1143).

## Jobs

All `EXIT:0`. Times are UTC from `results/logs/eval_plus_*.log`.

| Job | GPU | Start | End |
| --- | ---: | --- | --- |
| plus_full_init_spatial | 0 | 2026-10-11T01:59:23Z | 2026-10-11T03:45:11Z |
| plus_full_init_object | 0 | 2026-10-11T03:45:11Z | 2026-10-11T06:05:11Z |
| plus_full_init_goal | 0 | 2026-10-11T06:05:12Z | 2026-10-11T08:10:43Z |
| plus_full_init_10 | 0 | 2026-10-11T08:10:43Z | 2026-10-11T14:09:51Z |
| plus_fisher6_init_spatial | 2 | 2026-10-11T01:59:23Z | 2026-10-11T03:44:03Z |
| plus_fisher6_init_object | 2 | 2026-10-11T03:44:04Z | 2026-10-11T06:05:58Z |
| plus_fisher6_init_goal | 2 | 2026-10-11T06:05:58Z | 2026-10-11T08:17:12Z |
| plus_fisher6_init_10 | 2 | 2026-10-11T08:17:12Z | 2026-10-11T14:19:39Z |

`eval_info.json` wall-clock (`eval_s`): full Spatial 6152 / Object 8213 / Goal 7327 / Long 21346; Drop-6 Spatial 6086 / Object 8311 / Goal 7673 / Long 21546. MuJoCo-dominated; do not use as efficiency.

## Reading

Init-state OOD is a larger absolute drop from in-distribution than paraphrased language (full 97.25 → 77.69, vs Language 89.20). Fisher Drop-6 costs **−3.51** Avg. here, not a collapse, and Spatial even gains 5 variants. Goal is the weakest suite for Drop-6 (−6.85). Do not put these rates into the in-distribution operating-point table.

---

# LIBERO-plus Camera Viewpoints（π0.5 Fisher Drop-6 vs full）

Same checkpoint, skip, seed, and 1-trial-per-variant recipe as Language. Category is official **Camera Viewpoints** (1599 variants). GPU1 = full; GPU3 = Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`. All eight jobs finished (`eval_info.json` present; Spatial/Object/Goal `EXIT:0`; Long json mtime is the end time because the `==== EXIT` line was not flushed). Do not overwrite Language 89.20 / 85.24 or Init 77.69 / 74.18.

Source: `results/eval_libero_plus/plus_{full,fisher6}_camera_{spatial,object,goal,10}/{eval_info.json,drop_meta.json}`. Script: `scripts/pi05_libero/run_eval_plus_dim.sh`, category `Camera Viewpoints`.

Table: Camera-viewpoint success rate (%). Avg. is the unweighted mean of four suite rates. Long = LIBERO-10.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full | — | 74.73 | 73.99 | 66.67 | 52.03 | 66.85 |
| Drop-6 (6/18) | Fisher | 70.48 | 63.13 | 64.95 | 44.15 | 60.68 |

Counts (successes / camera variants):

| Setting | Spatial | Object | Goal | Long | All |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 281/376 | 293/396 | 272/408 | 218/419 | 1064/1599 |
| Fisher Drop-6 | 265/376 | 250/396 | 265/408 | 185/419 | 965/1599 |

Δ vs this-machine Plus-camera full (macro Avg. **66.85** → **60.68**, **−6.18**):

- Spatial **−4.26** (281/376 → 265/376)
- Object **−10.86** (293/396 → 250/396)
- Goal **−1.72** (272/408 → 265/408)
- Long **−7.88** (218/419 → 185/419)
- Micro (all 1599 variants): 66.54 → 60.35, **−6.19**
- Spatial+Object+Goal only: macro **71.80** → **66.19**, **−5.61**

Camera is the weakest Plus slice so far for both settings (full 66.85 vs Language 89.20 / Init 77.69). Drop-6 does not collapse, but the gap is larger than Language (−3.96) and Init (−3.51), concentrated on Object and Long. Same operator on in-distribution LIBERO is +0.50 Avg.

## Drop / keep

Same as Language: Full `[]`; Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`. Skip keep-KV.

## Same-variant disagreement (full vs Fisher Drop-6)

Each suite uses the same 1-based-minus-1 camera task ids under `seed=1000`.

| Suite | n | both success | both fail | full✓ drop✗ | full✗ drop✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 376 | 237 | 67 | 44 | 28 |
| Object | 396 | 210 | 63 | 83 | 40 |
| Goal | 408 | 245 | 116 | 27 | 20 |
| Long | 419 | 162 | 178 | 56 | 23 |

Net extra failures under Drop-6: Spatial 16, Object 43, Goal 7, Long 33. All 1599: 99 extra failures (1064 → 965).

## Jobs

Spatial/Object/Goal `EXIT:0`. Long end times are `eval_info.json` mtime (UTC).

| Job | GPU | Start | End |
| --- | ---: | --- | --- |
| plus_full_camera_spatial | 1 | 2026-10-11T01:59:23Z | 2026-10-11T04:09:56Z |
| plus_full_camera_object | 1 | 2026-10-11T04:09:56Z | 2026-10-11T06:29:01Z |
| plus_full_camera_goal | 1 | 2026-10-11T06:29:01Z | 2026-10-11T08:46:16Z |
| plus_full_camera_10 | 1 | 2026-10-11T08:46:16Z | 2026-10-11T16:31:04Z |
| plus_fisher6_camera_spatial | 3 | 2026-10-11T01:59:23Z | 2026-10-11T04:13:58Z |
| plus_fisher6_camera_object | 3 | 2026-10-11T04:13:58Z | 2026-10-11T06:39:01Z |
| plus_fisher6_camera_goal | 3 | 2026-10-11T06:39:01Z | 2026-10-11T08:58:04Z |
| plus_fisher6_camera_10 | 3 | 2026-10-11T08:58:04Z | 2026-10-11T16:41:05Z |

`eval_info.json` wall-clock (`eval_s`): full Spatial 7639 / Object 8139 / Goal 8033 / Long 27669; Drop-6 Spatial 7886 / Object 8499 / Goal 8137 / Long 27555. MuJoCo-dominated; do not use as efficiency.

## Reading

Camera-view OOD is harder than language or init-state (full 97.25 → 66.85). Fisher Drop-6 costs **−6.18** Avg., **−10.86** on Object, **−7.88** on Long — still not a collapse. Do not put these rates into the in-distribution operating-point table.

---

# LIBERO-plus Objects Layout（π0.5 Fisher Drop-6 vs full）

Same protocol. Category **Objects Layout** (1525 variants). GPU1 = full; GPU3 = Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`. Spatial/Object/Goal `EXIT:0`. Long json mtime is the end time (`==== EXIT` not flushed). Do not overwrite Language 89.20 / 85.24, Init 77.69 / 74.18, or Camera 66.85 / 60.68.

Source: `results/eval_libero_plus/plus_{full,fisher6}_layout_{spatial,object,goal,10}/{eval_info.json,drop_meta.json}`. Script: `scripts/pi05_libero/run_eval_plus_dim.sh`, category `Objects Layout`.

Table: Layout success rate (%). Avg. is the unweighted mean of four suite rates.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full | — | 96.88 | 88.83 | 68.71 | 91.67 | 86.52 |
| Drop-6 (6/18) | Fisher | 96.36 | 87.84 | 66.59 | 78.85 | 82.41 |

Counts (successes / layout variants):

| Setting | Spatial | Object | Goal | Long | All |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 373/385 | 358/403 | 292/425 | 286/312 | 1309/1525 |
| Fisher Drop-6 | 371/385 | 354/403 | 283/425 | 246/312 | 1254/1525 |

Δ vs this-machine Plus-layout full (macro Avg. **86.52** → **82.41**, **−4.11**):

- Spatial **−0.52** (373/385 → 371/385)
- Object **−0.99** (358/403 → 354/403)
- Goal **−2.12** (292/425 → 283/425)
- Long **−12.82** (286/312 → 246/312)
- Micro (all 1525 variants): 85.84 → 82.23, **−3.61**
- Spatial+Object+Goal only: macro **84.81** → **83.60**, **−1.21**

Layout Spatial/Object stay near in-distribution. The Drop-6 cost is concentrated on Long. That is not a collapse.

## Drop / keep

Same as Language. Skip keep-KV.

## Same-variant disagreement (full vs Fisher Drop-6)

| Suite | n | both success | both fail | full✓ drop✗ | full✗ drop✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 385 | 364 | 5 | 9 | 7 |
| Object | 403 | 340 | 31 | 18 | 14 |
| Goal | 425 | 265 | 115 | 27 | 18 |
| Long | 312 | 237 | 17 | 49 | 9 |

Net extra failures under Drop-6: Spatial 2, Object 4, Goal 9, Long 40. All 1525: 55 extra failures (1309 → 1254).

## Jobs

Spatial/Object/Goal `EXIT:0`. Long end times are `eval_info.json` mtime (UTC).

| Job | GPU | Start | End |
| --- | ---: | --- | --- |
| plus_full_layout_spatial | 1 | 2026-10-11T16:31:47Z | 2026-10-11T18:48:04Z |
| plus_full_layout_object | 1 | 2026-10-11T18:48:04Z | 2026-10-11T21:08:11Z |
| plus_full_layout_goal | 1 | 2026-10-11T21:08:11Z | 2026-10-11T23:38:11Z |
| plus_full_layout_10 | 1 | 2026-10-11T23:38:11Z | 2026-10-12T02:23:29Z |
| plus_fisher6_layout_spatial | 3 | 2026-10-11T16:41:48Z | 2026-10-11T18:59:06Z |
| plus_fisher6_layout_object | 3 | 2026-10-11T18:59:06Z | 2026-10-11T21:19:04Z |
| plus_fisher6_layout_goal | 3 | 2026-10-11T21:19:04Z | 2026-10-11T23:54:17Z |
| plus_fisher6_layout_10 | 3 | 2026-10-11T23:54:17Z | 2026-10-12T02:50:51Z |

`eval_s`: full Spatial 7877 / Object 8200 / Goal 8818 / Long 9739; Drop-6 Spatial 7966 / Object 8189 / Goal 9133 / Long 10415. MuJoCo-dominated; do not use as efficiency.

## Reading

Layout Avg. **86.52 → 82.41 (−4.11)**. Spatial/Object/Goal stay within about 0.5–2 points; Long is **−12.82**. Sensor Noise is still running on GPU0/2 and is empty in the core table.

---

# LIBERO-plus Light Conditions（π0.5 Fisher Drop-6 vs full）

Same protocol. Category **Light Conditions** (1142 variants). GPU1 = full; GPU3 = Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`. All eight jobs `EXIT:0`. Do not overwrite Language 89.20 / 85.24, Init 77.69 / 74.18, Camera 66.85 / 60.68, or Layout 86.52 / 82.41.

Source: `results/eval_libero_plus/plus_{full,fisher6}_light_{spatial,object,goal,10}/{eval_info.json,drop_meta.json}`. Script: `scripts/pi05_libero/run_eval_plus_dim.sh`, category `Light Conditions`.

Table: Light success rate (%). Avg. is the unweighted mean of four suite rates. Long = LIBERO-10.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full | — | 100.00 | 99.33 | 97.85 | 96.72 | 98.47 |
| Drop-6 (6/18) | Fisher | 99.66 | 99.33 | 97.85 | 83.21 | 95.01 |

Counts (successes / light variants):

| Setting | Spatial | Object | Goal | Long | All |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 292/292 | 295/297 | 273/279 | 265/274 | 1125/1142 |
| Fisher Drop-6 | 291/292 | 295/297 | 273/279 | 228/274 | 1087/1142 |

Δ vs this-machine Plus-light full (macro Avg. **98.47** → **95.01**, **−3.46**):

- Spatial **−0.34** (292/292 → 291/292)
- Object **0.00** (295/297 → 295/297)
- Goal **0.00** (273/279 → 273/279)
- Long **−13.50** (265/274 → 228/274)
- Micro (all 1142 variants): 98.51 → 95.18, **−3.33**
- Spatial+Object+Goal only: macro **99.06** → **98.94**, **−0.11**

Light is the easiest Plus slice for both settings (full 98.47, close to in-distribution 97.25). Spatial/Object/Goal are unchanged under Drop-6. The whole gap is Long.

## Drop / keep

Same as Language. Skip keep-KV.

## Same-variant disagreement (full vs Fisher Drop-6)

| Suite | n | both success | both fail | full✓ drop✗ | full✗ drop✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 292 | 291 | 0 | 1 | 0 |
| Object | 297 | 293 | 0 | 2 | 2 |
| Goal | 279 | 267 | 0 | 6 | 6 |
| Long | 274 | 225 | 6 | 40 | 3 |

Net extra failures under Drop-6: Spatial 1, Object 0, Goal 0, Long 37. All 1142: 38 extra failures (1125 → 1087).

## Jobs

All `EXIT:0`. Times are UTC from `results/logs/eval_plus_*.log`.

| Job | GPU | Start | End |
| --- | ---: | --- | --- |
| plus_full_light_spatial | 1 | 2026-10-12T03:31:29Z | 2026-10-12T04:53:07Z |
| plus_full_light_object | 1 | 2026-10-12T04:53:07Z | 2026-10-12T06:30:55Z |
| plus_full_light_goal | 1 | 2026-10-12T06:30:55Z | 2026-10-12T07:53:13Z |
| plus_full_light_10 | 1 | 2026-10-12T07:53:13Z | 2026-10-12T10:22:20Z |
| plus_fisher6_light_spatial | 3 | 2026-10-12T03:31:29Z | 2026-10-12T04:54:15Z |
| plus_fisher6_light_object | 3 | 2026-10-12T04:54:16Z | 2026-10-12T06:22:42Z |
| plus_fisher6_light_goal | 3 | 2026-10-12T06:22:42Z | 2026-10-12T07:44:04Z |
| plus_fisher6_light_10 | 3 | 2026-10-12T07:44:04Z | 2026-10-12T10:16:49Z |

`eval_s`: full Spatial 4683 / Object 5680 / Goal 4733 / Long 8739; Drop-6 Spatial 4747 / Object 5121 / Goal 4672 / Long 8963. MuJoCo-dominated; do not use as efficiency.

## Reading

Light Avg. **98.47 → 95.01 (−3.46)**. Spatial/Object/Goal are within 0.4 points (S+O+G macro −0.11). Long is **−13.50**, 40 vs 3 one-sided failures. This repeats the Layout pattern (Long −12.82 with S/O/G near flat): under OOD shifts that leave short tasks easy, the Drop-6 cost shows up on long-horizon tasks. Not a collapse. Do not put these rates into the in-distribution operating-point table.

---

# LIBERO-plus Background Textures（π0.5 Fisher Drop-6 vs full）

Same protocol. Category **Background Textures** (1076 variants). GPU1 = full; GPU3 = Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`. All eight jobs `EXIT:0`. Do not overwrite Language 89.20 / 85.24, Init 77.69 / 74.18, Camera 66.85 / 60.68, Layout 86.52 / 82.41, or Light 98.47 / 95.01.

Source: `results/eval_libero_plus/plus_{full,fisher6}_background_{spatial,object,goal,10}/{eval_info.json,drop_meta.json}`. Script: `scripts/pi05_libero/run_eval_plus_dim.sh`, category `Background Textures`.

Table: Background success rate (%). Avg. is the unweighted mean of four suite rates. Long = LIBERO-10.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full | — | 99.61 | 98.79 | 93.95 | 94.12 | 96.62 |
| Drop-6 (6/18) | Fisher | 99.61 | 98.79 | 79.36 | 91.70 | 92.36 |

Counts (successes / background variants):

| Setting | Spatial | Object | Goal | Long | All |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 257/258 | 245/248 | 264/281 | 272/289 | 1038/1076 |
| Fisher Drop-6 | 257/258 | 245/248 | 223/281 | 265/289 | 990/1076 |

Δ vs this-machine Plus-background full (macro Avg. **96.62** → **92.36**, **−4.26**):

- Spatial **0.00** (257/258 → 257/258)
- Object **0.00** (245/248 → 245/248)
- Goal **−14.59** (264/281 → 223/281)
- Long **−2.42** (272/289 → 265/289)
- Micro (all 1076 variants): 96.47 → 92.01, **−4.46**
- Spatial+Object+Goal only: macro **97.45** → **92.59**, **−4.86**

Background Spatial/Object match full. The Drop-6 cost is concentrated on Goal (−14.59), not Long. That is not a collapse, but Goal is the weakest suite on this slice.

## Drop / keep

Same as Language. Skip keep-KV.

## Same-variant disagreement (full vs Fisher Drop-6)

| Suite | n | both success | both fail | full✓ drop✗ | full✗ drop✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 258 | 256 | 0 | 1 | 1 |
| Object | 248 | 242 | 0 | 3 | 3 |
| Goal | 281 | 216 | 10 | 48 | 7 |
| Long | 289 | 252 | 4 | 20 | 13 |

Net extra failures under Drop-6: Spatial 0, Object 0, Goal 41, Long 7. All 1076: 48 extra failures (1038 → 990).

## Jobs

All `EXIT:0`. Times are UTC from `results/logs/eval_plus_*.log`.

| Job | GPU | Start | End |
| --- | ---: | --- | --- |
| plus_full_background_spatial | 1 | 2026-10-12T11:14:48Z | 2026-10-12T12:23:08Z |
| plus_full_background_object | 1 | 2026-10-12T12:23:08Z | 2026-10-12T13:29:40Z |
| plus_full_background_goal | 1 | 2026-10-12T13:29:41Z | 2026-10-12T14:53:43Z |
| plus_full_background_10 | 1 | 2026-10-12T14:53:43Z | 2026-10-12T18:02:38Z |
| plus_fisher6_background_spatial | 3 | 2026-10-12T11:14:48Z | 2026-10-12T12:22:39Z |
| plus_fisher6_background_object | 3 | 2026-10-12T12:22:40Z | 2026-10-12T13:29:32Z |
| plus_fisher6_background_goal | 3 | 2026-10-12T13:29:32Z | 2026-10-12T15:01:25Z |
| plus_fisher6_background_10 | 3 | 2026-10-12T15:01:25Z | 2026-10-12T18:17:30Z |

`eval_s`: full Spatial 3894 / Object 3808 / Goal 4859 / Long 11119; Drop-6 Spatial 3861 / Object 3831 / Goal 5327 / Long 11550. MuJoCo-dominated; do not use as efficiency.

## Reading

Background Avg. **96.62 → 92.36 (−4.26)**. Spatial/Object are identical to full (net 0 extra failures). Goal is **−14.59** (48 vs 7 one-sided failures), larger than Light Long (−13.50) and Layout Long (−12.82). Long is only **−2.42**. The Drop-6 OOD hole is not always long-horizon: here it is Goal. Not a collapse. Do not put these rates into the in-distribution operating-point table.
