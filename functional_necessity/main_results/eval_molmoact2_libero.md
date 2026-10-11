# Closed-loop LIBERO results (MolmoAct2 Drop-8 / Drop-12 / Drop-14 / Drop-15 / Drop-16 / Drop-17, no recovery)

Importance metrics for VLM-layer dropping on `allenai/MolmoAct2-LIBERO-LeRobot` / LIBERO.
Format follows DTR Table 3 (Spatial / Object / Goal / Long / Avg.).

**Protocol.** Official LeRobot MolmoAct2 eval: `inference_action_mode=continuous`, `dtype=bfloat16`, `eval_seed=1000`, `seed=1000`, `n_episodes=50`, `num_steps_wait=50`, four suites × 10 tasks × 50 episodes (2000 episodes), `batch_size=1`, no video, no recovery fine-tune. `chunk_size=n_action_steps=10` from the released config. Local-only difference vs the Hub recipe: `enable_inference_cuda_graph=false` (24GB). Skip is keep-KV identity residual on the 36 VLM layers (0–35): a dropped layer still writes K/V for the paired action expert and does not update the residual stream. High score = keep. Drop-8 is the eight lowest scores; Drop-12 is the twelve lowest (Taylor / Fisher share one set). Drop-14 / Drop-15 / Drop-16 / Drop-17 **exclude layer 0** from the redundancy order, then drop the k lowest remaining scores. Last-16 is a positional control `[20–35]`, not a ranked set. Identical drop sets share one 2000-episode run († / †† / § / ‖ / ¶ / ♯). Greedy-PPL was not scored. Hessian OOM; CKA ranking is not usable (36-layer `1−CKA` sits near `1e-6`).

Anchor is this machine's seed-1000 full model: Spatial 97.4 / Object 99.6 / Goal 96.8 / Long 96.2, avg 97.50 (1950/2000). Not the Hub 98.25.

Source: `results/molmoact2_libero/{baseline_eval,drop8_*,drop12_*,drop14_*,drop15_*,drop16_*,drop17_*}/eval_info.json`. CosSim has no `eval_info.json` (stopped during Long); suite rates from the 50-episode tqdm `running_success_rate` at each task boundary. Ranks: `results/molmoact2_libero/scores.md`. Calibration: 32 samples, continuous flow-matching with one frozen (noise, time) per sample. Drop-15 / Drop-16 / Drop-17 likewise exclude layer 0 from the redundancy order, then drop the k lowest remaining scores. Fisher and IGIA share one set at Drop-15 / Drop-16 / Drop-17; Taylor uses a different set at those k.

Table: success rate (%). Long = LIBERO-10.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 97.4 | 99.6 | 96.8 | 96.2 | 97.50 |
| Drop-8 (8/36) | Taylor † | 97.4 | 100.0 | 97.2 | 96.4 | 97.75 |
|  | Fisher † | 97.4 | 100.0 | 97.2 | 96.4 | 97.75 |
|  | IGIA † | 97.4 | 100.0 | 97.2 | 96.4 | 97.75 |
|  | GateProbe † | 97.4 | 100.0 | 97.2 | 96.4 | 97.75 |
|  | PPL | 59.4 | 75.4 | 32.8 | 33.0 | 50.15 |
|  | CosSim ‡ | 1.4 | 0.0 | 0.0 | 0.0 | 0.35 |
|  | Magnitude | 37.0 | 77.8 | 12.4 | 1.4 | 32.15 |
|  | CKA | — | — | — | — | not eval |
|  | Hessian | — | — | — | — | OOM |
| Drop-12 (12/36) | Taylor †† | 97.6 | 99.8 | 98.2 | 95.4 | 97.75 |
|  | Fisher †† | 97.6 | 99.8 | 98.2 | 95.4 | 97.75 |
| Drop-14 (14/36, no L0) | Taylor § | 97.6 | 100.0 | 98.0 | 96.4 | 98.00 |
|  | IGIA § | 97.6 | 100.0 | 98.0 | 96.4 | 98.00 |
|  | Fisher | 96.6 | 99.0 | 98.0 | 96.2 | 97.45 |
| Drop-15 (15/36, no L0) | Fisher ‖ | 98.4 | 99.2 | 98.6 | 96.2 | 98.10 |
|  | IGIA ‖ | 98.4 | 99.2 | 98.6 | 96.2 | 98.10 |
|  | Taylor | 97.0 | 99.8 | 97.6 | 96.4 | 97.70 |
| Drop-16 (16/36, no L0) | Fisher ¶ | 91.2 | 79.8 | 51.0 | 69.0 | 72.75 |
|  | IGIA ¶ | 91.2 | 79.8 | 51.0 | 69.0 | 72.75 |
|  | Taylor | 98.2 | 99.4 | 98.6 | 96.2 | 98.10 |
|  | GateProbe | 91.6 | 96.6 | 77.2 | 82.6 | 87.00 |
|  | PPL ♭ | 1.2 | 0.0 | 0.0 | 0.0 | 0.30 |
|  | CosSim ♮ | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Last-16 (control) | 74.4 | 72.2 | 46.4 | 40.0 | 58.25 |
| Drop-17 (17/36, no L0) | Taylor | 91.6 | 95.8 | 93.4 | 77.6 | 89.60 |
|  | GateProbe | 91.8 | 92.4 | 72.8 | 73.6 | 82.65 |
|  | Fisher ♯ | 91.0 | 77.8 | 49.8 | 63.8 | 70.60 |
|  | IGIA ♯ | 91.0 | 77.8 | 49.8 | 63.8 | 70.60 |
|  | PPL ♭♭ | 0.8 | 0.0 | 0.0 | 0.0 | 0.20 |

† Taylor / Fisher / IGIA / GateProbe are the same set `[28, 29, 30, 31, 32, 33, 34, 35]`. One 2000-episode run covers all four.

†† Drop-12 Taylor / Fisher are the same set `[1, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]`. One 2000-episode run covers both.

‡ CosSim `[1, 2, 3, 4, 5, 6, 7, 8]` stopped after Spatial / Object / Goal (10/10) and Long 3/10, all three Long tasks 0%. Remaining Long marked 0. Spatial is 1.4% (one task 14%, the other nine 0%). No `eval_info.json`.

§ Drop-14 IGIA / Taylor (no L0) are the same set `[1, 2, 3, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]`. One 2000-episode run covers both (`drop14_igia_nol0`). Native Taylor Drop-14 already excludes L0 (same set). Native IGIA Drop-14 is `[0, 1, 2, 25–35]` and was not closed-loop evaluated.

Drop-14 Fisher (no L0) is `[1, 2, 4, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` (`drop14_fisher_nol0`). Native Fisher Drop-14 is `[0, 1, 2, 25–35]` and was not closed-loop evaluated.

‖ Drop-15 Fisher / IGIA (no L0) are the same set `[1, 2, 3, 4, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` (`drop15_fisher_nol0`). One 2000-episode run covers both.

¶ Drop-16 Fisher / IGIA (no L0) are the same set `[1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` (`drop16_fisher_nol0`). One 2000-episode run covers both.

Drop-15 Taylor (no L0) is `[1, 2, 3, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` (`drop15_taylor_nol0`). Drop-16 Taylor (no L0) is `[1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` (`drop16_taylor_nol0`). Last-16 (control) is `[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` (`drop16_last`). Drop-17 Taylor (no L0) is `[1, 2, 3, 4, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` (`drop17_taylor_nol0`). Drop-16 GateProbe (no L0) is `[2, 3, 4, 6, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` (`drop16_gateprobe_nol0`). Drop-17 GateProbe (no L0) is `[2, 3, 4, 6, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` (`drop17_gateprobe_nol0`).

♯ Drop-17 Fisher / IGIA (no L0) are the same set `[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` (`drop17_fisher_nol0`). One 2000-episode run covers both.

♭ PPL Drop-16 (no L0) `[1, 3, 4, 5, 6, 7, 23, 26, 28–35]` was stopped during Spatial task 9 (9/10 Spatial tasks done, task 9 at 12/50, running 0%). Observed 6/462 successes (Spatial task 2 = 12%, the other eight finished Spatial tasks 0%). Remaining episodes, including unfinished Spatial and all of Object / Goal / Long, marked 0. No `eval_info.json`. Source: `results/molmoact2_libero/drop16_ppl_nol0/stopped.json`.

♭♭ PPL Drop-17 (no L0) `[1, 2, 3, 4, 5, 6, 7, 23, 26, 28–35]` was stopped during Spatial task 9 (9/10 Spatial tasks done, task 9 at 17/50, running 0%). Observed 4/467 successes (Spatial task 2 = 8%, the other eight finished Spatial tasks 0%). Remaining episodes marked 0. No `eval_info.json`. Source: `results/molmoact2_libero/drop17_ppl_nol0/stopped.json`.

♮ CosSim Drop-16 no-L0 `[1–13, 20, 21, 25]` was SIGTERM'd on GPU3 during Goal task 2 (Spatial 10/10, Object 10/10, Goal 2/10 finished, all 0/50; Goal task 2 at 6/50, running 0%). Observed 0/1106 successes. Remaining Goal / Long marked 0. No `eval_info.json`. Source: `results/molmoact2_libero/drop16_cossim_nol0/stopped.json`.

Δ Avg. vs this-machine anchor (97.50), completed rows only:

- Drop-8: Taylor / Fisher / IGIA / GateProbe **+0.25**; PPL **−47.35**; Magnitude **−65.35**; CosSim **−97.15**.
- Drop-12: Taylor / Fisher **+0.25**.
- Drop-14 (no L0): Taylor / IGIA **+0.50**; Fisher **−0.05**.
- Drop-15 (no L0): Fisher / IGIA **+0.60**; Taylor **+0.20**.
- Drop-16 (no L0): Fisher / IGIA **−24.75**; Taylor **+0.60**; GateProbe **−10.50**; PPL stopped **−97.20**; CosSim stopped **−97.50**; Last-16 (control) **−39.25**.
- Drop-17 (no L0): Taylor **−7.90**; GateProbe **−14.85**; Fisher / IGIA **−26.90**; PPL stopped **−97.30**.

Closed-loop wall time (`overall.eval_s`): baseline 16h 56m 22s, shared gradient Drop-8 16h 55m 46s (~30.5 s/episode), PPL Drop-8 28h 3m 18s, Magnitude Drop-8 31h 17m 55s, Taylor/Fisher Drop-12 16h 31m 47s, IGIA/Taylor Drop-14 no-L0 16h 52m 13s (1960/2000, ~30.4 s/episode), Fisher Drop-14 no-L0 16h 59m 29s (1949/2000, ~30.6 s/episode), Fisher/IGIA Drop-15 no-L0 17h 1m 32s (1962/2000, ~30.6 s/episode), Taylor Drop-15 no-L0 16h 48m 38s (1954/2000, ~30.3 s/episode), Fisher/IGIA Drop-16 no-L0 23h 7m 45s (1455/2000, ~41.6 s/episode), Taylor Drop-16 no-L0 16h 56m 28s (1962/2000, ~30.5 s/episode), Taylor Drop-17 no-L0 19h 41m 27s (1792/2000, ~35.4 s/episode), Fisher/IGIA Drop-17 no-L0 24h 10m 4s (1412/2000, ~43.5 s/episode), GateProbe Drop-16 no-L0 19h 47m 16s (1740/2000, ~35.6 s/episode), GateProbe Drop-17 no-L0 22h 17m 56s (1653/2000, ~40.1 s/episode), Last-16 control 26h 28m 53s (1165/2000, ~47.7 s/episode). Dropping the late VLM layers does not shorten the LIBERO loop when success holds; Fisher Drop-16 / Drop-17, Taylor Drop-17, and Last-16 are slower because more episodes hit the step cap. PPL / Magnitude Drop-8 are slower for the same reason.

## Drop indices (VLM 0–35)

From `scores.md`, high-score-keep. Hessian has no scores. CKA Drop-8 is listed for the record and was not closed-loop evaluated.

### Drop-4 (not yet closed-loop)

| Metric | Drop |
| --- | --- |
| Magnitude | [1, 2, 3, 21] |
| CosSim | [1, 2, 3, 4] |
| CKA | [29, 30, 31, 32] |
| PPL | [4, 6, 7, 23] |
| Taylor | [32, 33, 34, 35] |
| Fisher | [30, 32, 33, 35] |
| IGIA | [32, 33, 34, 35] |
| GateProbe | [30, 31, 33, 35] |
| Hessian | out of memory |

### Drop-8

| Metric | Drop |
| --- | --- |
| Taylor † | [28, 29, 30, 31, 32, 33, 34, 35] |
| Fisher † | [28, 29, 30, 31, 32, 33, 34, 35] |
| IGIA † | [28, 29, 30, 31, 32, 33, 34, 35] |
| GateProbe † | [28, 29, 30, 31, 32, 33, 34, 35] |
| PPL | [3, 4, 6, 7, 23, 26, 28, 34] |
| CosSim | [1, 2, 3, 4, 5, 6, 7, 8] |
| Magnitude | [1, 2, 3, 18, 19, 20, 21, 22] |
| CKA | [2, 3, 28, 29, 30, 31, 32, 33] |
| Hessian | out of memory |

### Drop-12

| Metric | Drop |
| --- | --- |
| Taylor †† | [1, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| Fisher †† | [1, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |

### Drop-14 (closed-loop excludes L0)

Layer 0 is removed from the redundancy order, then the 14 lowest remaining scores are dropped.

| Metric | Drop |
| --- | --- |
| Taylor § | [1, 2, 3, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| IGIA § | [1, 2, 3, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| Fisher | [1, 2, 4, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |

Native Drop-14 (includes L0, not closed-loop): IGIA / Fisher `[0, 1, 2, 25–35]`; Taylor is already the no-L0 set above.

### Drop-15 (closed-loop excludes L0)

| Metric | Drop |
| --- | --- |
| Fisher ‖ | [1, 2, 3, 4, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| IGIA ‖ | [1, 2, 3, 4, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| Taylor | [1, 2, 3, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |

### Drop-16 (closed-loop excludes L0)

| Metric | Drop |
| --- | --- |
| Fisher ¶ | [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| IGIA ¶ | [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| Taylor | [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| GateProbe | [2, 3, 4, 6, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| PPL ♭ | [1, 3, 4, 5, 6, 7, 23, 26, 28, 29, 30, 31, 32, 33, 34, 35] |
| CosSim ♮ | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 25] |
| Last-16 (control) | [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |

### Drop-17 (closed-loop excludes L0)

| Metric | Drop |
| --- | --- |
| Taylor | [1, 2, 3, 4, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| GateProbe | [2, 3, 4, 6, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| Fisher ♯ | [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| IGIA ♯ | [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] |
| PPL ♭♭ | [1, 2, 3, 4, 5, 6, 7, 23, 26, 28, 29, 30, 31, 32, 33, 34, 35] |

Taylor Drop-15 / Drop-16 / Drop-17 keep L24 (and Drop-17 also L23) instead of Fisher’s extra early layers at the same k. Last-16 is the last 16 VLM layers `[20–35]`; it is not taken from a redundancy order.

## Checks

Baseline, shared gradient Drop-8, PPL Drop-8, Magnitude Drop-8, Taylor/Fisher Drop-12, IGIA/Taylor Drop-14 no-L0, Fisher Drop-14 no-L0, Fisher/IGIA Drop-15 no-L0, Taylor Drop-15 no-L0, Fisher/IGIA Drop-16 no-L0, Taylor Drop-16 no-L0, Last-16 control, Taylor Drop-17 no-L0, Fisher/IGIA Drop-17 no-L0, GateProbe Drop-16 no-L0, and GateProbe Drop-17 no-L0 all wrote `eval_info.json` with `n_episodes=2000` (40 tasks × 50), `pc_success` matching `per_group` bits, `EXIT:0`. CosSim was stopped during Long (pid SIGTERM); Spatial/Object/Goal are complete, Long set to 0. CKA was not queued (ranking noise). Hessian has no layer ranking. Last-16 (`drop16_last`) is 58.25 matching 1165/2000 bits.

PPL Drop-16 / Drop-17 were SIGTERM'd during Spatial (GPU2 / GPU3). Remaining suites and unfinished Spatial episodes are scored 0, same rule as CosSim. GateProbe Drop-16 and Drop-17 later finished (87.00 and 82.65).

Local full-model anchor is 97.50%, vs Hub 98.25%. Use 97.50 as the referee, not the Hub number.

## CARE (Stage-2): action-discontinuity reweighting (2026-10-05)

Full write-up: `main_results/care_libero.md`.

No closed-loop. The loss still uses the stored one-step action. Weights are the mean adjacent L2 of the next 10 **loss-normalized** actions from the same episode (preprocessor masked-normalizer + clamp; first step matches the stored action with max abs diff 0). 7-d vs arm-6-d Spearman = 0.70, so both were scored. Taylor α is outside the loss.

Weighted Taylor Drop-16 no-L0 is still `[1, 2, 3, 4, 24–35]` for both 7-d and arm-6-d. Jaccard vs Stage-1 = **1.0**. Stop. 98.10 is already at the ceiling; a new set would only be asked whether it still holds, and this set did not change. Source: `results/stage2_action_weight/`.

Weighted Fisher Drop-16 no-L0 (α outside the loss; 7-d and arm-6-d identical) moves from `[1, 2, 3, 4, 5, 25–35]` to `[1, 2, 3, 4, 24–35]` (L5 ↔ L24, Stage-1 Fisher scores 0.00319 vs 0.00327). Jaccard vs Stage-1 Fisher = **0.882**. The new set is exactly Taylor Drop-16, already closed-loop at **98.10**. Do not re-run 2000 episodes.

CARE IGIA Drop-16 no-L0 is still `[1, 2, 3, 4, 5, 25–35]` for both 7-d and arm-6-d. Jaccard vs Stage-1 = **1.0**. Stop. Do not re-evaluate 72.75 and do not write that CARE maps IGIA onto Taylor. IGIA L5 = 104.28 vs L24 = 131.32 (not a near-tie). Drop-17 no-L0 is still `[1–5, 24–35]`. Source: `results/stage2_action_weight/molmoact2_igia_action_discontinuity.json`.

CARE Drop-17 no-L0. Fisher / Taylor did not change (Jaccard 1.0); do not re-eval 70.60 / 89.60. The Drop-16 L5 ↔ L24 swap is absorbed once both layers are in the Drop-17 set. GateProbe did change, and both variants finished (2000 ep, seed=1000, EXIT:0). Do not overwrite the Stage-1 GateProbe 82.65 row.

| Setting | Drop | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GateProbe Stage-1 | `[2–4, 6, 22, 24–35]` | 91.8 | 92.4 | 72.8 | 73.6 | 82.65 |
| CARE 7-d | `[1–4, 6, 24–35]` | 45.0 | 16.4 | 15.6 | 2.2 | 19.80 |
| CARE arm 6-d | `[1–4, 22, 24–35]` | 98.2 | 99.0 | 96.8 | 96.6 | 97.65 |

7-d swaps L22 ↔ L1 and drops L6 (396/2000, 33h 54m). Arm 6-d swaps L6 ↔ L1 and keeps L6 (1953/2000, 18h 14m). The two dimensions disagree, so the cut stays refused. Full write-up: `care_libero.md` §7.2. Sources: `results/molmoact2_libero/drop17_gateprobe_action_discontinuity{,_arm6}/eval_info.json`.
