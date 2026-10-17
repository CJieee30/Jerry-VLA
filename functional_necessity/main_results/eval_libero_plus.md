# LIBERO-plus：π0.5 与 MolmoAct2

**π0.5** — `lerobot/pi05_libero_finetuned_v044` vs Stage-1 Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`, skip keep-KV. In-distribution LIBERO 400-ep anchors: full **97.25**, Drop-6 **97.75** (`eval_pi05_libero_v044.md`). Do not mix those with Plus tables.

**MolmoAct2** — `allenai/MolmoAct2-LIBERO-LeRobot`（本机 `checkpoints/MolmoAct2-LIBERO-LeRobot` + `MolmoAct2-LIBERO` 结构）。Plus **Language Instructions × Full × NFE=10** 四轮已于 **2026-10-16 UTC** 全部 `EXIT:0`（见 Molmo 专节）；无剪枝、无 NFE=1，**不**写入上方 π0.5 七维核心表。

两栈共用 Plus 环境配方：每语言/变体 **1** 次试验（`n_episodes=1`），`seed=1000`，`n_action_steps=10`，`num_inference_steps=10`，`compile_model=false`，`batch_size=1`，`control_mode=relative`，无视频、无 recovery。任务 id 来自 `LIBERO-plus/.../task_classification.json`（JSON 内 **1-based id 减 1**）。

**Macro Avg.**（本文 π0.5 主表统一口径）：每个 Plus 维度内，对四个套件分别从 `eval_info.json` 的 `per_task` 重算成功率（**非**对表中四舍五入后的百分数再平均）；维度 **Avg.** = 四套件率的**算术平均**；七维总分行 = 七个维度 Avg. 的**算术平均**。**Micro**（成功数/变体总数）仅作补充，可与 Macro 因套件规模不同而略有偏差。

**NFE=1 batch (Fisher Drop-6, `num_inference_steps=1`, 2026-10-13–14):** GPU1 = Language + Camera + **Light** (24 suites, all `EXIT:0`); GPU3 = Init + Layout + **Background** (24 suites, all `EXIT:0`). Tags: `plus_fisher6_nfe1_{lang,init,camera,layout,light,background,noise}_*`. **Sensor Noise NFE=10:** π0.5 full/Drop-6 **四轮完成**（**2026-10-16–17 UTC** `EXIT:0`）；见 § Sensor Noise。**Drop-6 NFE=1 Sensor Noise**（GPU1）：**Spatial / Object 已完成**；**Goal 进行中**；**Long 未开始**（截至 **2026-10-17 UTC**）。

## Core: seven Plus dimensions (Avg. %)

Drop-6 = Fisher `[1,13,14,15,16,17]` keep-KV, **`num_inference_steps=10`** (official Plus). **Drop-6 NFE=1** = same skip, **`num_inference_steps=1`** (`plus_fisher6_nfe1_{lang,init,camera,layout,light,background}_*`; Sensor Noise not run at NFE=1).

| Dimension | Full (NFE=10) | Drop-6 (NFE=10) | Δ (D6−Full) | Drop-6 (NFE=1) | Δ (NFE1−D6) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Language Instructions | 89.20 | 85.24 | −3.96 | **84.67** | −0.57 |
| Robot Initial States | 77.69 | 74.18 | −3.51 | **74.47** | +0.29 |
| Camera Viewpoints | 66.85 | 60.68 | −6.18 | **61.24** | +0.56 |
| Sensor Noise | 93.83 | 89.08 | −4.75 | — | — |
| Objects Layout | 86.52 | 82.41 | −4.11 | **80.98** | −1.43 |
| Light Conditions | 98.47 | 95.01 | −3.46 | **95.90** | +0.89 |
| Background Textures | 96.62 | 92.36 | −4.26 | **90.96** | −1.40 |

All **seven** Plus dimensions (NFE=10 full vs Drop-6): macro **87.03** vs **82.71**, **−4.32**（自 `eval_info` 重算，见上）。On six finished NFE=1 slices (excl. Sensor Noise), macro Δ vs Drop-6 NFE=10 is **−0.57 / +0.29 / +0.56 / −1.43 / +0.89 / −1.40**—none match full-vs-Drop-6 NFE=10 gaps. **Background Goal** at NFE=1 vs NFE=10 Drop-6 is **−3.91 pp** (75.4448 vs 79.3594), not −4.51 pp; **Layout Goal** is **−4.47 pp** (62.12 vs 66.59)—**Layout Goal** is the larger Goal-suite NFE=1 vs Drop-6 NFE=10 regression among finished dims. Light **Long** is **+2.19** at NFE=1 (vs **−13.50** at NFE=10 full vs Drop-6). Camera remains the largest **dimension-level** Drop-6 NFE=10 gap (−6.18). Among **suite-level** full vs Drop-6 NFE=10 gaps, **Light Long (−13.50 pp)** exceeds Sensor Noise **Long (−7.57 pp**, largest **within** the Noise dimension only).

## Core: finished dims by suite (%)

| Dimension | Setting | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Language | Full | 99.74 | 91.24 | 73.90 | 91.91 | 89.20 |
| Language | Drop-6 (NFE=10) | 95.64 | 89.27 | 71.71 | 84.33 | 85.24 |
| Language | Drop-6 (NFE=1) | 96.92 | 88.98 | 69.76 | 83.03 | 84.67 |
| Init | Full | 86.57 | 72.61 | 77.26 | 74.30 | 77.69 |
| Init | Drop-6 (NFE=10) | 88.00 | 68.84 | 70.42 | 69.47 | 74.18 |
| Init | Drop-6 (NFE=1) | 87.14 | 68.59 | 70.66 | 71.50 | 74.47 |
| Camera | Full | 74.73 | 73.99 | 66.67 | 52.03 | 66.85 |
| Camera | Drop-6 | 70.48 | 63.13 | 64.95 | 44.15 | 60.68 |
| Layout | Full | 96.88 | 88.83 | 68.71 | 91.67 | 86.52 |
| Layout | Drop-6 | 96.36 | 87.84 | 66.59 | 78.85 | 82.41 |
| Light | Full | 100.00 | 99.33 | 97.85 | 96.72 | 98.47 |
| Light | Drop-6 (NFE=10) | 99.66 | 99.33 | 97.85 | 83.21 | 95.01 |
| Light | Drop-6 (NFE=1) | 100.00 | 100.00 | 98.21 | 85.40 | 95.90 |
| Background | Full | 99.61 | 98.79 | 93.95 | 94.12 | 96.62 |
| Background | Drop-6 (NFE=10) | 99.61 | 98.79 | 79.36 | 91.70 | 92.36 |
| Background | Drop-6 (NFE=1) | 98.84 | 97.18 | 75.44 | 92.39 | 90.96 |
| Sensor Noise | Full | 98.58 | 96.92 | 92.08 | 87.75 | 93.83 |
| Sensor Noise | Drop-6 (NFE=10) | 91.45 | 95.26 | 89.45 | 80.18 | 89.08 |

Counts: Language 1367/1537 vs 1306/1537; Init 1200/1550 vs 1143/1550; Camera 1064/1599 vs 965/1599; Layout 1309/1525 vs 1254/1525; Light 1125/1142 vs 1087/1142; Background 1038/1076 vs 990/1076; **Sensor Noise 1498/1601 vs 1422/1601** (351+422+379+449 noise variants).

## Drop-6 NFE=1 micro counts (six finished Plus dims)

| Dimension | Drop-6 NFE=10 micro | Drop-6 NFE=1 micro |
| --- | --- | --- |
| Language Instructions | 1306/1537 | **1297/1537** |
| Robot Initial States | 1143/1550 | **1148/1550** |
| Camera Viewpoints | 965/1599 | **974/1599** |
| Objects Layout | 1254/1525 | **1229/1525** |
| Light Conditions | 1087/1142 | **1097/1142** |
| Background Textures | 990/1076 | **975/1076** |

NFE=1 does not collapse these slices vs Drop-6 NFE=10 (see core table). In-distribution π0.5 Fisher Drop-6 NFE=1 on LIBERO-40 is **96.50%** (400 ep) and **95.70% / 96.25%** (2000 ep, same seed, two runs; `eval_nfe_libero.md` §1/§7)—not comparable to Plus variant counts. (Do not confuse with SmolVLA Drop-4 NFE=3 **76.00** in `eval_nfe_libero.md` §4.)

Per-dimension protocol, disagreement, and jobs are in the sections below. Language **89.20 / 85.24** (full / Drop-6 NFE=10) and Init **77.69 / 74.18** are not overwritten.

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

Per-dimension sections follow. Sensor Noise four-suite results are in § Sensor Noise and the core tables above.

---

# LIBERO-plus Language Instructions — MolmoAct2 Full NFE=10

MolmoAct2 on the official **Language Instructions** slice only (not the other six Plus dimensions). **Full** model: no VLM skip, no materialize. In-distribution Molmo anchor: **97.50** Avg. on standard LIBERO 2000-ep (`eval_molmoact2_libero.md`); do not mix with Plus variant counts.

**Protocol (aligned with π0.5 Plus Language except Molmo env/camera/postprocess).** `inference_action_mode=continuous`, `dtype=bfloat16`, `num_inference_steps=10`, `n_action_steps=10`, `eval_seed=1000`, `per_episode_seed=true`, `num_steps_wait=50`, `enable_inference_cuda_graph=false`. Dual-camera mapping `agentview_image→image`, `robot0_eye_in_hand_image→wrist_image`. Category **Language Instructions** from installed `task_classification.json`: Spatial **390**, Object **354**, Goal **410**, Long **383** → **1537** variants total.

**Entry.** `scripts/molmoact2_libero/eval_plus_molmo.py`; shell `run_eval_plus_lang.sh` (per suite) / `run_eval_plus_lang_serial.sh` (Spatial→Object→Goal→Long, then chains **Robot Initial States** via `run_eval_plus_init_serial.sh`). **GPU3 only** (`CUDA_VISIBLE_DEVICES=3`). Serial one process at a time. **Run in tmux** (e.g. `tmux attach -t molmo_plus_lang_nfe10_g3`); do not rely on IDE background shells. Aggregator: `scripts/molmoact2_libero/summarize_plus_lang.py`.

**Robot Initial States (auto after Language Long).** Same recipe as `molmoact2_full_nfe10_lang_10` (`num_inference_steps=10`, `seed=1000`, 1 trial/variant, category from `task_classification.json`). Tags: `molmoact2_full_nfe10_init_{spatial,object,goal,10}` → `results/eval_libero_plus/molmoact2_full_nfe10_init_*`. Master log: `results/logs/eval_molmoact2_full_nfe10_init_serial.log`. One-shot watcher for an already-running Language serial: `launch_molmo_init_after_lang_gpu3.sh` (log: `results/logs/eval_molmoact2_chain_lang_to_init.log`).

**Smoke (not in table).** `run_eval_plus_lang_smoke.sh` → `results/eval_libero_plus/_smoke/molmoact2_smoke_full_nfe10_lang_spatial/` — 2 Spatial language variants, **2/2** success (`EXIT:0`, 2026-10-16). Log: `results/logs/eval_molmoact2_smoke_full_nfe10_lang_spatial.log`.

**Formal outputs (reuse if present; do not overwrite `eval_info.json`).**

| Suite | Tag | Directory |
| --- | --- | --- |
| Spatial | `molmoact2_full_nfe10_lang_spatial` | `results/eval_libero_plus/molmoact2_full_nfe10_lang_spatial/` |
| Object | `molmoact2_full_nfe10_lang_object` | `results/eval_libero_plus/molmoact2_full_nfe10_lang_object/` |
| Goal | `molmoact2_full_nfe10_lang_goal` | `results/eval_libero_plus/molmoact2_full_nfe10_lang_goal/` |
| Long | `molmoact2_full_nfe10_lang_10` | `results/eval_libero_plus/molmoact2_full_nfe10_lang_10/` |

Each dir: `eval_info.json`, `run_config.json`, `task_manifest.json`, `plus_meta.json`. Shared eval ckpt symlinks: `results/eval_libero_plus/_molmo_shared/eval_ckpt/`. Per-suite logs: `results/logs/eval_molmoact2_full_nfe10_lang_*.log`; serial master: `results/logs/eval_molmoact2_full_nfe10_lang_serial.log`.

**Code pins (`run_config.json` → `code_sha256`).** `eval_plus_molmo.py`, `load_policy.py`, `layer_skip.py` at job start — same trio intended for later Drop-16 materialized / keep-KV Plus runs.

### Results — MolmoAct2 Full NFE=10 Language

**Status: complete** — four suites `eval_info.json` + `==== EXIT … exit=0` in `results/logs/eval_molmoact2_full_nfe10_lang_*.log` (serial **`molmo_plus_lang_nfe10_g3`**, GPU3). Aggregator: `python scripts/molmoact2_libero/summarize_plus_lang.py`.

| Setting | Spatial | Object | Goal | Long | Avg. | Micro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full NFE=10 | 92.82 | 93.22 | 76.83 | 87.99 | **87.71** | **1344/1537 (87.44%)** |

Variant denominators: **390 / 354 / 410 / 383** → **1537**. Macro **Avg.** = unweighted mean of four suite **pc_success** (%). Counts from `per_group` in each `eval_info.json` (matches `summarize_plus_lang.py`).

**Long suite only (`molmoact2_full_nfe10_lang_10`, GPU3).** **87.99%** (**337/383**); `==== EXIT` **2026-10-16T19:00:29Z**; `overall.eval_s` ≈ **21 542 s** (wall-clock aggregate in json, not per-step efficiency).

**Run record.** Wall-clock for the four-suite serial on GPU3: **≈16.46 h** (`eval_molmoact2_full_nfe10_lang_spatial` `START` → `lang_10` `EXIT`, 2026-10-16 UTC). Do **not** sum `overall.eval_s` across suites (~58.8k s)—those aggregates are not additive wall time. π0.5 Language full reference (same 1537 variant ids): macro **89.20**, micro **1367/1537**.

**Video / logging.** π0.5 Plus jobs: `max_episodes_rendered=0` (no eval videos). Molmo Plus writes **per-success** rollout mp4 under each job’s `videos/` (see `eval_info.json` `video_paths`); inference flags match π0.5 (`compile_model=false`, `enable_inference_cuda_graph=false`).

**Reading.** Molmo macro **87.71** is **−1.49 pp** vs π0.5 Full **89.20**. **Spatial** Molmo **92.82** vs π0.5 **99.74** (**−6.92 pp**); Object **93.22** vs **91.24** (+1.98); Goal **76.83** vs **73.90** (+2.93); Long **87.99** vs **91.91** (−3.92). Drop-16 / materialized Plus Language is **not run** in this batch.

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

Layout Avg. **86.52 → 82.41 (−4.11)**. Spatial/Object/Goal stay within about 0.5–2 points; Long is **−12.82**. Sensor Noise Spatial+Object are in the core table; Goal/Long are pending.

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

---

# LIBERO-plus Sensor Noise（π0.5 Fisher Drop-6 vs full）

Official **Sensor Noise** category. Requires ImageMagick+wand (`MAGICK_HOME`); preflight in `run_eval_plus_lang.sh`. Same Plus recipe otherwise: 1 trial per variant, `seed=1000`, `n_action_steps=10`, `num_inference_steps=10`, `compile_model=false`. GPU0 = full; GPU2 = Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`.

**Status (2026-10-17).** **All four suites** finished (`eval_info.json` + `==== EXIT … exit=0` on every `plus_{full,fisher6}_noise_{spatial,object,goal,10}` log). Goal resumed earlier on GPU0/2; **Long** (`plus_*_noise_10`) completed **2026-10-16T20:38:27Z** (full, GPU0) and **2026-10-17T00:21:02Z** (Drop-6, GPU2).

Source: `results/eval_libero_plus/plus_{full,fisher6}_noise_{spatial,object,goal,10}/{eval_info.json,drop_meta.json}`. Script: `run_eval_plus_dim.sh` → `run_eval_plus_lang.sh`, category `Sensor Noise`.

Table: success rate (%) — **1601** noise variants (351 + 422 + 379 + 449).

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full | — | 98.58 | 96.92 | 92.08 | 87.75 | **93.83** |
| Drop-6 (6/18) | Fisher | 91.45 | 95.26 | 89.45 | 80.18 | **89.08** |
| Δ (D6−Full) | — | −7.13 | −1.66 | −2.63 | **−7.57** | **−4.75** |

Counts (successes / noise variants):

| Setting | Spatial | Object | Goal | Long | All |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 346/351 | 409/422 | 349/379 | **394/449** | **1498/1601 (93.57%)** |
| Fisher Drop-6 | 321/351 | 402/422 | 339/379 | **360/449** | **1422/1601 (88.82%)** |

**Long only (GPU0 / GPU2 jobs that ended 2026-10-16–17).** Full **87.75%** (394/449); Drop-6 **80.18%** (360/449); **−7.57 pp**. `overall.eval_s` (wall aggregate): full **≈166 237 s**; Drop-6 **≈179 215 s**.

## Same-variant disagreement (full vs Fisher Drop-6, all suites)

| Suite | n | both success | both fail | full✓ drop✗ | full✗ drop✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 351 | 319 | 3 | 27 | 2 |
| Object | 422 | 392 | 3 | 17 | 10 |
| Goal | 379 | 325 | 16 | 24 | 14 |
| Long | 449 | 343 | 38 | **51** | 17 |

Net extra failures under Drop-6 (paired variants): **76** across 1601 tasks (Long contributes the largest full-only-fail count).

## Jobs (Goal + Long completion)

| Job | GPU | Start (UTC) | End (UTC) |
| --- | ---: | --- | --- |
| plus_full_noise_goal | 0 | 2026-10-14T01:21:43Z | 2026-10-14T22:24:34Z |
| plus_full_noise_10 | 0 | 2026-10-14T22:24:48Z | **2026-10-16T20:38:27Z** |
| plus_fisher6_noise_goal | 2 | 2026-10-14T01:21:43Z | 2026-10-14T22:30:53Z |
| plus_fisher6_noise_10 | 2 | 2026-10-14T22:31:07Z | **2026-10-17T00:21:02Z** |

(Spatial/Object job times unchanged from 2026-10-12–13 batch; see git history of this file.)

## Reading

Four-suite Sensor Noise macro **93.83 → 89.08 (−4.75)**—in line with Language/Layout Drop-6 gaps. **Long** is the hardest suite under noise for **both** stacks and shows the largest Drop-6 penalty (**−7.57 pp**); Spatial is second (**−7.13**). Object is nearly flat (**−1.66**). Historical Spatial+Object-only macro was **97.75 / 93.36**; adding Goal/Long pulls the dimension average down because Long full is only **87.75%**. Sensor Noise is now included in the **seven-dimension** π0.5 core macro (**87.03 / 82.71**). Drop-6 NFE=1 noise queue on GPU1 remains separate.

---

# LIBERO-plus Language Instructions — Fisher Drop-6 **NFE=1**

Fisher Drop-6 only (`plus_fisher6_nfe1_lang_*`). Same Plus language task ids and `seed=1000` as § Language Instructions; **`num_inference_steps=1`** (NFE=10 rows **85.24** Avg. are unchanged). GPU1, all eight jobs `EXIT:0` (2026-10-13 UTC).

Source: `results/eval_libero_plus/plus_fisher6_nfe1_lang_{spatial,object,goal,10}/`. Launcher: `scripts/pi05_libero/run_eval_plus_fisher6_nfe1_pair.sh lang` → `run_eval_plus_dim.sh` with NFE=1.

| Setting | Spatial | Object | Goal | Long | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drop-6 NFE=1 | 96.92 | 88.98 | 69.76 | 83.03 | **84.67** |
| Drop-6 NFE=10 (anchor) | 95.64 | 89.27 | 71.71 | 84.33 | 85.24 |
| Full NFE=10 (reference) | 99.74 | 91.24 | 73.90 | 91.91 | 89.20 |

Counts: **1297/1537** successes (micro 84.39%). vs Drop-6 NFE=10 **1306/1537** (−9 variants, −0.59 pp micro).

Δ vs Drop-6 NFE=10 macro (**−0.57**): Spatial **+1.28**, Object **−0.29**, Goal **−1.95**, Long **−1.30**.

## NFE=10 vs NFE=1 disagreement (same Drop-6, same variants)

| Suite | n | both success | both fail | NFE10✓ NFE1✗ | NFE10✗ NFE1✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 390 | 371 | 10 | 2 | 7 |
| Object | 354 | 309 | 32 | 7 | 6 |
| Goal | 410 | 278 | 108 | 16 | 8 |
| Long | 383 | 310 | 52 | 13 | 8 |

Net variants that fail under NFE=1 but succeed under NFE=10: 38 (Spatial 2, Object 7, Goal 16, Long 13). Net the other way: 29.

## Jobs

| Job | GPU | Start (UTC) | End (UTC) |
| --- | ---: | --- | --- |
| plus_fisher6_nfe1_lang_spatial | 1 | 2026-10-13T16:16:49Z | 2026-10-13T17:37:00Z |
| plus_fisher6_nfe1_lang_object | 1 | 2026-10-13T17:37:00Z | 2026-10-13T18:47:40Z |
| plus_fisher6_nfe1_lang_goal | 1 | 2026-10-13T18:47:40Z | 2026-10-13T20:15:29Z |
| plus_fisher6_nfe1_lang_10 | 1 | 2026-10-13T20:15:29Z | 2026-10-13T22:22:02Z |

## Reading

Plus language under Drop-6 is already −3.96 vs full at NFE=10. Cutting inference to **one Euler step** costs only **−0.57** more Avg. vs Drop-6 NFE=10—much smaller than the OOD gap vs full (−4.53 NFE=1 vs full NFE=10). Long remains the softest suite (−1.30 pp vs NFE=10). Not a collapse.

---

# LIBERO-plus Robot Initial States — Fisher Drop-6 **NFE=1**

Fisher Drop-6 only (`plus_fisher6_nfe1_init_*`). Same Init task ids and `seed=1000` as § Robot Initial States; **`num_inference_steps=1`**. GPU3, all eight jobs `EXIT:0` (2026-10-13 UTC).

Source: `results/eval_libero_plus/plus_fisher6_nfe1_init_{spatial,object,goal,10}/`. Launcher: `run_eval_plus_fisher6_nfe1_pair.sh init`.

| Setting | Spatial | Object | Goal | Long | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drop-6 NFE=1 | 87.14 | 68.59 | 70.66 | 71.50 | **74.47** |
| Drop-6 NFE=10 (anchor) | 88.00 | 68.84 | 70.42 | 69.47 | 74.18 |
| Full NFE=10 (reference) | 86.57 | 72.61 | 77.26 | 74.30 | 77.69 |

Counts: **1148/1550** (micro 74.06%). vs Drop-6 NFE=10 **1143/1550** (+5 variants, +0.32 pp micro).

Δ vs Drop-6 NFE=10 macro (**+0.29**): Spatial **−0.86**, Object **−0.25**, Goal **+0.24**, Long **+2.03**.

## NFE=10 vs NFE=1 disagreement (same Drop-6)

| Suite | n | both success | both fail | NFE10✓ NFE1✗ | NFE10✗ NFE1✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 350 | 296 | 33 | 12 | 9 |
| Object | 398 | 248 | 99 | 26 | 25 |
| Goal | 409 | 263 | 95 | 25 | 26 |
| Long | 393 | 253 | 92 | 20 | 28 |

## Jobs

| Job | GPU | Start (UTC) | End (UTC) |
| --- | ---: | --- | --- |
| plus_fisher6_nfe1_init_spatial | 3 | 2026-10-13T16:16:49Z | 2026-10-13T17:39:14Z |
| plus_fisher6_nfe1_init_object | 3 | 2026-10-13T17:39:14Z | 2026-10-13T19:09:20Z |
| plus_fisher6_nfe1_init_goal | 3 | 2026-10-13T19:09:20Z | 2026-10-13T20:35:25Z |
| plus_fisher6_nfe1_init_10 | 3 | 2026-10-13T20:35:25Z | 2026-10-13T22:50:11Z |

## Reading

Init OOD at Drop-6 NFE=10 is **74.18** Avg. NFE=1 is **74.47** (+0.29)—no extra collapse from a single denoising step on this slice. Long gains **+2.03** pp at 1-trial resolution (39 variants flip). Spatial/Object are flat to slightly down. In-distribution π0.5 Drop-6 NFE=1 on standard LIBERO is **96.50%** (400 ep) / **95.70–96.25%** (2000 ep); Plus Init is a different task distribution—compare only qualitatively.

---

# LIBERO-plus Camera Viewpoints — Fisher Drop-6 **NFE=1**

Fisher Drop-6 only (`plus_fisher6_nfe1_camera_*`). Same Camera task ids and `seed=1000` as § Camera Viewpoints; **`num_inference_steps=1`**. GPU1, all eight jobs `EXIT:0` (2026-10-14 UTC).

Source: `results/eval_libero_plus/plus_fisher6_nfe1_camera_{spatial,object,goal,10}/`. Launcher: `run_eval_plus_fisher6_nfe1_camera_layout_pair.sh camera` → `run_eval_plus_dim.sh` with NFE=1.

| Setting | Spatial | Object | Goal | Long | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drop-6 NFE=1 | 72.07 | 62.37 | 63.73 | 46.78 | **61.24** |
| Drop-6 NFE=10 (anchor) | 70.48 | 63.13 | 64.95 | 44.15 | 60.68 |
| Full NFE=10 (reference) | 74.73 | 73.99 | 66.67 | 52.03 | 66.85 |

Counts: **974/1599** successes (micro 60.91%). vs Drop-6 NFE=10 **965/1599** (+9 variants, +0.56 pp macro).

Δ vs Drop-6 NFE=10 macro (**+0.56**): Spatial **+1.60**, Object **−0.76**, Goal **−1.23**, Long **+2.63**.

## NFE=10 vs NFE=1 disagreement (same Drop-6, same variants)

| Suite | n | both success | both fail | NFE10✓ NFE1✗ | NFE10✗ NFE1✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 376 | 240 | 80 | 25 | 31 |
| Object | 396 | 222 | 121 | 28 | 25 |
| Goal | 408 | 240 | 123 | 25 | 20 |
| Long | 419 | 164 | 202 | 21 | 32 |

Net variants that fail under NFE=1 but succeed under NFE=10: 99 (Spatial 25, Object 28, Goal 25, Long 21). Net the other way: 108.

## Jobs

| Job | GPU | Start (UTC) | End (UTC) |
| --- | ---: | --- | --- |
| plus_fisher6_nfe1_camera_spatial | 1 | 2026-10-14T01:21:29Z | 2026-10-14T02:55:34Z |
| plus_fisher6_nfe1_camera_object | 1 | 2026-10-14T02:55:34Z | 2026-10-14T04:27:28Z |
| plus_fisher6_nfe1_camera_goal | 1 | 2026-10-14T04:27:29Z | 2026-10-14T06:20:26Z |
| plus_fisher6_nfe1_camera_10 | 1 | 2026-10-14T06:20:26Z | 2026-10-14T09:43:22Z |

`eval_s`: Spatial 5458 / Object 5327 / Goal 6578 / Long 11947. MuJoCo-dominated; do not use as efficiency.

## Reading

Camera OOD at Drop-6 NFE=10 is already the weakest finished dim (**60.68** Avg.). One Euler step is **61.24** (+0.56)—no extra collapse; Long and Spatial tick up, Object/Goal tick down slightly. The gap vs full NFE=10 (−5.61 at NFE=1 vs −6.18 at NFE=10) is still dominated by OOD camera shift, not NFE. Not a collapse.

---

# LIBERO-plus Objects Layout — Fisher Drop-6 **NFE=1**

Fisher Drop-6 only (`plus_fisher6_nfe1_layout_*`). Same Layout task ids and `seed=1000` as § Objects Layout; **`num_inference_steps=1`**. GPU3, all eight jobs `EXIT:0` (2026-10-14 UTC).

Source: `results/eval_libero_plus/plus_fisher6_nfe1_layout_{spatial,object,goal,10}/`. Launcher: `run_eval_plus_fisher6_nfe1_camera_layout_pair.sh layout`.

| Setting | Spatial | Object | Goal | Long | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drop-6 NFE=1 | 95.32 | 86.35 | 62.12 | 80.13 | **80.98** |
| Drop-6 NFE=10 (anchor) | 96.36 | 87.84 | 66.59 | 78.85 | 82.41 |
| Full NFE=10 (reference) | 96.88 | 88.83 | 68.71 | 91.67 | 86.52 |

Counts: **1229/1525** (micro 80.59%). vs Drop-6 NFE=10 **1254/1525** (−25 variants, −1.43 pp macro).

Δ vs Drop-6 NFE=10 macro (**−1.43**): Spatial **−1.04**, Object **−1.49**, Goal **−4.47**, Long **+1.28**.

## NFE=10 vs NFE=1 disagreement (same Drop-6)

| Suite | n | both success | both fail | NFE10✓ NFE1✗ | NFE10✗ NFE1✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 385 | 360 | 7 | 11 | 7 |
| Object | 403 | 336 | 37 | 18 | 12 |
| Goal | 425 | 254 | 132 | 29 | 10 |
| Long | 312 | 230 | 46 | 16 | 20 |

## Jobs

| Job | GPU | Start (UTC) | End (UTC) |
| --- | ---: | --- | --- |
| plus_fisher6_nfe1_layout_spatial | 3 | 2026-10-14T01:21:29Z | 2026-10-14T03:00:55Z |
| plus_fisher6_nfe1_layout_object | 3 | 2026-10-14T03:00:55Z | 2026-10-14T04:46:21Z |
| plus_fisher6_nfe1_layout_goal | 3 | 2026-10-14T04:46:21Z | 2026-10-14T07:16:27Z |
| plus_fisher6_nfe1_layout_10 | 3 | 2026-10-14T07:16:27Z | 2026-10-14T09:52:20Z |

`eval_s`: Spatial 5775 / Object 6132 / Goal 8779 / Long 9156. MuJoCo-dominated.

## Reading

Layout Drop-6 NFE=10 **82.41** → NFE=1 **80.98** (−1.43). Most of the cost is **Goal (−4.47)**; Long is **+1.28** at 1-trial resolution (same pattern as Init Long flipping up under NFE=1). Spatial/Object stay within ~1.5 pp. Still not a collapse vs the **−4.11** full-vs-Drop-6 OOD gap at NFE=10.

---

# LIBERO-plus Light Conditions — Fisher Drop-6 **NFE=1**

Fisher Drop-6 only (`plus_fisher6_nfe1_light_*`). Same Light task ids and `seed=1000` as § Light Conditions; **`num_inference_steps=1`**. GPU1, all eight jobs `EXIT:0` (2026-10-14 UTC).

Source: `results/eval_libero_plus/plus_fisher6_nfe1_light_{spatial,object,goal,10}/`. Launcher: `run_eval_plus_fisher6_nfe1_light_background_pair.sh light`.

| Setting | Spatial | Object | Goal | Long | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drop-6 NFE=1 | 100.00 | 100.00 | 98.21 | 85.40 | **95.90** |
| Drop-6 NFE=10 (anchor) | 99.66 | 99.33 | 97.85 | 83.21 | 95.01 |
| Full NFE=10 (reference) | 100.00 | 99.33 | 97.85 | 96.72 | 98.47 |

Counts: **1097/1142** (micro 96.06%). vs Drop-6 NFE=10 **1087/1142** (+10 variants, +0.89 pp macro).

Δ vs Drop-6 NFE=10 macro (**+0.89**): Spatial **+0.34**, Object **+0.67**, Goal **+0.36**, Long **+2.19**.

## NFE=10 vs NFE=1 disagreement (same Drop-6)

| Suite | n | both success | both fail | NFE10✓ NFE1✗ | NFE10✗ NFE1✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 292 | 291 | 0 | 0 | 1 |
| Object | 297 | 295 | 0 | 0 | 2 |
| Goal | 279 | 268 | 0 | 5 | 6 |
| Long | 274 | 214 | 26 | 14 | 20 |

## Jobs

| Job | GPU | Start (UTC) | End (UTC) |
| --- | ---: | --- | --- |
| plus_fisher6_nfe1_light_spatial | 1 | 2026-10-14T13:04:30Z | 2026-10-14T14:09:29Z |
| plus_fisher6_nfe1_light_object | 1 | 2026-10-14T14:09:29Z | 2026-10-14T15:13:32Z |
| plus_fisher6_nfe1_light_goal | 1 | 2026-10-14T15:13:32Z | 2026-10-14T16:11:12Z |
| plus_fisher6_nfe1_light_10 | 1 | 2026-10-14T16:11:12Z | 2026-10-14T17:59:05Z |

`eval_s`: Spatial 3698 / Object 3645 / Goal 3256 / Long 6272. MuJoCo-dominated.

## Reading

Light Drop-6 NFE=10 **95.01** → NFE=1 **95.90** (**+0.89**). At NFE=10 the hole was almost entirely **Long (−13.50 vs full)**; at NFE=1, Long is **+2.19** vs Drop-6 NFE=10 (still **−11.32** vs full Long). Spatial/Object at NFE=1 are perfect on this slice. One Euler step does not worsen the Light OOD slice; it does not close the full-model Long gap.

---

# LIBERO-plus Background Textures — Fisher Drop-6 **NFE=1**

Fisher Drop-6 only (`plus_fisher6_nfe1_background_*`). Same Background task ids and `seed=1000` as § Background Textures; **`num_inference_steps=1`**. GPU3, all eight jobs `EXIT:0` (2026-10-14 UTC).

Source: `results/eval_libero_plus/plus_fisher6_nfe1_background_{spatial,object,goal,10}/`. Launcher: `run_eval_plus_fisher6_nfe1_light_background_pair.sh background`.

| Setting | Spatial | Object | Goal | Long | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drop-6 NFE=1 | 98.84 | 97.18 | 75.44 | 92.39 | **90.96** |
| Drop-6 NFE=10 (anchor) | 99.61 | 98.79 | 79.36 | 91.70 | 92.36 |
| Full NFE=10 (reference) | 99.61 | 98.79 | 93.95 | 94.12 | 96.62 |

Counts: **975/1076** (micro 90.61%). vs Drop-6 NFE=10 **990/1076** (−15 variants, −1.40 pp macro).

Δ vs Drop-6 NFE=10 macro (**−1.40**): Spatial **−0.78**, Object **−1.61**, Goal **−3.91**, Long **+0.69**.

## NFE=10 vs NFE=1 disagreement (same Drop-6)

| Suite | n | both success | both fail | NFE10✓ NFE1✗ | NFE10✗ NFE1✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 258 | 254 | 0 | 3 | 1 |
| Object | 248 | 238 | 0 | 7 | 3 |
| Goal | 281 | 201 | 47 | 22 | 11 |
| Long | 289 | 249 | 6 | 16 | 18 |

## Jobs

| Job | GPU | Start (UTC) | End (UTC) |
| --- | ---: | --- | --- |
| plus_fisher6_nfe1_background_spatial | 3 | 2026-10-14T13:04:30Z | 2026-10-14T14:01:03Z |
| plus_fisher6_nfe1_background_object | 3 | 2026-10-14T14:01:03Z | 2026-10-14T14:53:55Z |
| plus_fisher6_nfe1_background_goal | 3 | 2026-10-14T14:53:55Z | 2026-10-14T15:57:55Z |
| plus_fisher6_nfe1_background_10 | 3 | 2026-10-14T15:57:56Z | 2026-10-14T17:44:01Z |

`eval_s`: Spatial 3183 / Object 2967 / Goal 3634 / Long 6158. MuJoCo-dominated.

## Reading

Background Drop-6 NFE=10 **92.36** → NFE=1 **90.96** (−1.40). The NFE=1 cost matches the NFE=10 pattern: concentrated on **Goal** (−3.91 pp vs Drop-6 NFE=10; NFE=10 vs full was **−14.59**). Long is flat to slightly up (+0.69). Not a collapse beyond the existing Drop-6 OOD hole on Goal.
