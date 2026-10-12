# Closed-loop LIBERO results (DB-CogACT Drop-8, no recovery)

Importance metrics for whole-layer dropping on `Dexmal/libero-db-cogact` / LIBERO.
Format follows DTR Table 3 (Spatial / Object / Goal / Long / Avg.).

**Protocol.** Dexbotic-Benchmark (not LeRobot): in-process closed loop aligned with `evaluation/configs/libero/{suite}.yaml`, `seed=7`, `n_episodes=50`, `num_steps_wait=10`, agentview 180° flip + pad-resize 256, prompt template `What action should the robot take to {task}?`, discrete gripper, per-suite `replan_step` Spatial 12 / Object 16 / Goal 16 / Long 15, `cfg_scale=1.5`, 10 DDIM steps, chunk 16. Four suites × 10 tasks × 50 episodes (2000 episodes). No docker/HTTP evaluator, no video, no recovery fine-tune. Skip is whole-layer identity on Qwen2.5-7B blocks 0–27 (the DiT-B action head reads the last unmasked hidden state, not per-layer K/V). High score = keep.

Anchor is this machine's seed-7 full model: Spatial 87.2 / Object 97.2 / Goal 95.8 / Long 91.6, avg 92.95 (1859/2000). Not the Hub 94.9 (93.8 / 97.8 / 96.2 / 91.8). The Spatial gap vs Hub is mostly Spatial t5 (`pick up the black bowl on the ramekin`, 15/50 = 30%); Object / Goal / Long are within 0.6 points of Hub.

Source: `results/cogact_libero/{baseline_eval,drop8_*}/eval_info.json`. Ranks: `results/cogact_libero/scores.md`. Calibration: 32 frames from `lerobot/libero`, seed 42, diffusion noise MSE, one frozen (noise, timestep) per sample.

Table: success rate (%). Long = LIBERO-10.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 87.2 | 97.2 | 95.8 | 91.6 | 92.95 |
| Drop-8 (8/28) | Fisher (no L0) ‡ | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | GateProbe § | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |

‡ Fisher Drop-8 closed-loop excludes layer 0 from the redundancy order, then drops the eight lowest remaining scores: `[1, 2, 3, 4, 6, 7, 9, 10]`. Native Fisher Drop-8 in `scores.md` is `[0, 1, 2, 3, 4, 6, 7, 9]` (includes L0) and was not closed-loop evaluated. Stopped after 115/2000 episodes on GPU2: Spatial t0 0/50, t1 0/50, t2 0/15. Remaining marked 0.

§ GateProbe Drop-8 is `[1, 2, 3, 4, 9, 19, 21, 22]` (L0 not in the set). Stopped after 237/2000 episodes on GPU3: Spatial t0–t3 0/50, t4 0/37. Remaining marked 0.

Δ Avg. vs this-machine anchor (92.95), completed rows only:

- Drop-8: Fisher (no L0) **−92.95**; GateProbe **−92.95**.

Closed-loop wall time (`overall.eval_s`): baseline 5h 12m 26s (~9.4 s/episode). Fisher Drop-8 and GateProbe Drop-8 have no `eval_s` (stopped early; remaining episodes marked 0).

## Drop indices (Qwen2.5 0–27)

From `scores.md`, high-score-keep. Hessian has no scores.

### Drop-4 (not yet closed-loop)

| Metric | Drop |
| --- | --- |
| Magnitude | [0, 1, 2, 9] |
| CosSim | [4, 7, 8, 9] |
| CKA | [6, 7, 8, 17] |
| PPL | [1, 2, 4, 5] |
| Taylor | [1, 2, 4, 5] |
| Fisher | [0, 1, 2, 4] |
| IGIA | [0, 1, 2, 4] |
| GateProbe | [3, 4, 9, 19] |
| Hessian | out of memory |

### Drop-8

| Metric | Drop |
| --- | --- |
| Magnitude | [0, 1, 2, 3, 4, 5, 6, 9] |
| CosSim | [2, 4, 6, 7, 8, 9, 10, 19] |
| CKA | [4, 6, 7, 8, 9, 10, 17, 22] |
| PPL | [1, 2, 4, 5, 6, 9, 10, 20] |
| Taylor | [0, 1, 2, 3, 4, 5, 6, 9] |
| Fisher | [0, 1, 2, 3, 4, 6, 7, 9] |
| Fisher (no L0) ‡ | [1, 2, 3, 4, 6, 7, 9, 10] |
| IGIA | [0, 1, 2, 3, 4, 6, 7, 10] |
| GateProbe | [1, 2, 3, 4, 9, 19, 21, 22] |
| Hessian | out of memory |

### Drop-12 (not yet closed-loop)

| Metric | Drop |
| --- | --- |
| Magnitude | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14] |
| CosSim | [2, 4, 6, 7, 8, 9, 10, 11, 14, 18, 19, 20] |
| CKA | [4, 6, 7, 8, 9, 10, 15, 16, 17, 19, 21, 22] |
| PPL | [1, 2, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22] |
| Taylor | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 27] |
| Fisher | [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 20, 25] |
| IGIA | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] |
| GateProbe | [0, 1, 2, 3, 4, 6, 9, 19, 21, 22, 23, 24] |
| Hessian | out of memory |

## Checks

Baseline wrote `eval_info.json` with `n_episodes=2000` (40 tasks × 50), `pc_success` matching `per_group` and `avg4`, `EXIT:0`. Fisher Drop-8 (no L0) has `eval_info.json` with `stopped_early=true`, `n_episodes_run=115`, `n_successes_run=0`; remaining 1885 episodes marked 0. GateProbe Drop-8 has `eval_info.json` with `stopped_early=true`, `n_episodes_run=237`, `n_successes_run=0`; remaining 1763 episodes marked 0.

Local full-model anchor is 92.95%, vs Hub 94.9%. Use 92.95 as the referee, not the Hub number.
