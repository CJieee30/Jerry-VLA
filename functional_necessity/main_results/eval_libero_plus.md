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

Plus language is a paraphrased-instruction OOD slice, 1 trial per variant. Fisher Drop-6 keeps the same keep-KV operator that is in-distribution neutral (97.75 vs 97.25). Here the cost is **−3.96** Avg., **−7.57** on Long, and Spatial/Object/Goal stay within about 2–4 points. That is a measured language-OOD gap, not a closed-loop collapse, and it is not a new selector. Do not put Plus rates into the in-distribution operating-point table. Other Plus dimensions were not run.
