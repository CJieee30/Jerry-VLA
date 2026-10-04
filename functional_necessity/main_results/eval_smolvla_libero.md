# Closed-loop LIBERO results (SmolVLA Drop-1 / Drop-2 / Drop-3 / Drop-4, no recovery)

Importance metrics for VLM-layer dropping on `lerobot/smolvla_libero` / LIBERO.
Format follows DTR Table 3 (Spatial / Object / Goal / Long / Avg.).

**Protocol.** LeRobot `lerobot-eval`, `n_action_steps=10`, `num_steps=10`, `chunk_size=50`, `seed=1000`, `control_mode=relative`, `n_episodes=10`, four suites × 10 tasks × 10 episodes (400 episodes), `batch_size=1`, no recovery fine-tune. Checkpoint adaptation: camera `rename_map` `image→camera1`, `image2→camera2` (same map as training on `lerobot/libero`), `empty_cameras=0`, and `observation.state` shape patched 6→8 to match the saved 8-dim normalizer stats. The checkpoint default `n_action_steps=50` is not used. Skip is keep-KV identity residual on the 16 VLM layers (0–15): a dropped layer still writes K/V for the action expert's cross-attention and does not update the VLM residual stream. The action expert is never dropped. High score = keep. Drop-1 / Drop-2 / Drop-3 / Drop-4 are the native k lowest scores (layer 0 is **not** excluded). Drop-1 `[15]` covers Taylor / GateProbe / PPL (††); Drop-1 `[14]` covers IGIA / Fisher (‡). Fisher and IGIA share one Drop-3 set (†). One closed-loop run per GPU; no two LIBERO evals shared a card.

Anchor is this machine's seed-1000 full model at `n_action_steps=10`: Spatial 78.0 / Object 89.0 / Goal 89.0 / Long 76.0, avg 83.00 (332/400). Not the paper 87.3. `HuggingFaceVLA/smolvla_libero` (32 layers, expert ×0.5) is a different checkpoint (64.75 at nas=10) and is not in this table.

Source: `results/smolvla_libero/{lerobot_official_nas10,drop1_*,drop2_*,drop3_*,drop4_*}/eval_info.json`. Ranks: `results/smolvla_libero/scores.md`. Calibration: 32 frames from `lerobot/libero`, seed 42, continuous flow-matching with one frozen (noise, time) per sample. Formulas are the π0.5 / MolmoAct2 ones; only the layer hooks are SmolVLA-specific (see Scoring below). Greedy-PPL was not scored.

Table: success rate (%). Long = LIBERO-10. 100 episodes per suite, so one episode is 1.0 point per suite and 0.25 point on Avg.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 78.0 | 89.0 | 89.0 | 76.0 | 83.00 |
| Drop-1 (1/16) | Taylor †† | 79.0 | 84.0 | 83.0 | 71.0 | 79.25 |
|  | GateProbe †† | 79.0 | 84.0 | 83.0 | 71.0 | 79.25 |
|  | PPL †† | 79.0 | 84.0 | 83.0 | 71.0 | 79.25 |
|  | IGIA ‡ | 82.0 | 90.0 | 82.0 | 63.0 | 79.25 |
|  | Fisher ‡ | 82.0 | 90.0 | 82.0 | 63.0 | 79.25 |
| Drop-2 (2/16) | Taylor | 80.0 | 90.0 | 84.0 | 68.0 | 80.50 |
|  | IGIA | 78.0 | 91.0 | 83.0 | 65.0 | 79.25 |
|  | Fisher | 79.0 | 88.0 | 75.0 | 67.0 | 77.25 |
|  | GateProbe | 70.0 | 88.0 | 79.0 | 54.0 | 72.75 |
| Drop-3 (3/16) | Taylor | 78.0 | 87.0 | 81.0 | 65.0 | 77.75 |
|  | IGIA † | 75.0 | 89.0 | 77.0 | 58.0 | 74.75 |
|  | Fisher † | 75.0 | 89.0 | 77.0 | 58.0 | 74.75 |
|  | GateProbe | 75.0 | 81.0 | 75.0 | 65.0 | 74.00 |
| Drop-4 (4/16) | Taylor | 72.0 | 88.0 | 84.0 | 70.0 | 78.50 |
|  | PPL | 76.0 | 80.0 | 83.0 | 60.0 | 74.75 |
|  | IGIA | 74.0 | 89.0 | 72.0 | 60.0 | 73.75 |
|  | GateProbe | 72.0 | 86.0 | 72.0 | 58.0 | 72.00 |
|  | Fisher | 69.0 | 86.0 | 72.0 | 50.0 | 69.25 |
|  | Hessian | 70.0 | 75.0 | 65.0 | 52.0 | 65.50 |
|  | CosSim | 39.0 | 49.0 | 7.0 | 27.0 | 30.50 |
|  | CKA | 30.0 | 40.0 | 6.0 | 15.0 | 22.75 |
|  | Magnitude | 6.0 | 8.0 | 0.0 | 5.0 | 4.75 |

† Drop-3 Fisher / IGIA are the same set `[10, 12, 14]`. One 400-episode run covers both (`drop3_igia`).

†† Drop-1 Taylor / GateProbe / PPL are the same set `[15]`. One 400-episode run covers all three (`drop1_taylor`).

‡ Drop-1 IGIA / Fisher are the same set `[14]`. One 400-episode run covers both (`drop1_igia`).

Δ Avg. vs this-machine anchor (83.00):

- Drop-1: Taylor / GateProbe / PPL **−3.75**; IGIA / Fisher **−3.75**.
- Drop-2: Taylor **−2.50**; IGIA **−3.75**; Fisher **−5.75**; GateProbe **−10.25**.
- Drop-3: Taylor **−5.25**; IGIA / Fisher **−8.25**; GateProbe **−9.00**.
- Drop-4: Taylor **−4.50**; PPL **−8.25**; IGIA **−9.25**; GateProbe **−11.00**; Fisher **−13.75**; Hessian **−17.50**; CosSim **−52.50**; CKA **−60.25**; Magnitude **−78.25**.

No SmolVLA Drop-1 through Drop-4 set holds the baseline. Taylor Drop-2 (80.50) is the closest. Taylor Drop-3 keeps L12 relative to Drop-4 and is 0.75 below Taylor Drop-4 (three episodes on 400). The six function-based metrics drop late layers (10–15) and lose 4.5–17.5 points; Long and Goal take most of the loss. The three representation/weight metrics drop early layers and collapse: Magnitude removes L0, CosSim / CKA remove L1/L2/L4–L6. Contrast MolmoAct2 (36 layers), where gradient-family Drop-8 to Drop-15 stay at or above its anchor.

Closed-loop wall time (`overall.eval_s`): baseline 1h 56m 30s (~17.5 s/episode), Taylor/GateProbe/PPL Drop-1 1h 55m 17s, IGIA/Fisher Drop-1 1h 56m 24s, Taylor Drop-2 2h 3m 34s, IGIA Drop-2 1h 54m 36s, Fisher Drop-2 1h 55m 3s, GateProbe Drop-2 2h 6m 0s, Taylor Drop-3 2h 6m 16s, IGIA/Fisher Drop-3 2h 11m 20s, GateProbe Drop-3 2h 7m 36s, PPL Drop-4 1h 55m 10s, Taylor Drop-4 1h 57m 29s, GateProbe Drop-4 2h 1m 43s, Fisher Drop-4 2h 2m 14s, IGIA Drop-4 2h 6m 7s, Hessian 2h 8m 48s, CosSim 2h 51m 8s, CKA 2h 57m 12s, Magnitude 3h 9m 17s (~28.4 s/episode). Collapsed runs are slower because failed episodes run to the step cap.

## Drop indices (VLM 0–15)

From `results/smolvla_libero/scores.md`, high-score-keep.

### Drop-1

| Metric | Drop |
| --- | --- |
| Taylor †† | [15] |
| GateProbe †† | [15] |
| PPL †† | [15] |
| IGIA ‡ | [14] |
| Fisher ‡ | [14] |

### Drop-2

| Metric | Drop |
| --- | --- |
| Taylor | [14, 15] |
| IGIA | [12, 14] |
| Fisher | [10, 14] |
| GateProbe | [7, 15] |

### Drop-3

| Metric | Drop |
| --- | --- |
| Taylor | [13, 14, 15] |
| IGIA † | [10, 12, 14] |
| Fisher † | [10, 12, 14] |
| GateProbe | [7, 14, 15] |

### Drop-4

| Metric | Drop |
| --- | --- |
| Magnitude | [0, 3, 7, 8] |
| CosSim | [2, 4, 5, 6] |
| CKA | [1, 2, 4, 10] |
| PPL | [11, 13, 14, 15] |
| Taylor | [12, 13, 14, 15] |
| Fisher | [10, 11, 12, 14] |
| IGIA | [10, 12, 14, 15] |
| GateProbe | [7, 13, 14, 15] |
| Hessian | [8, 10, 12, 14] |

All nine sets are distinct; each has its own 400-episode run.

### Drop-8 (scored, not closed-loop)

| Metric | Drop |
| --- | --- |
| Magnitude | [0, 2, 3, 4, 6, 7, 8, 9] |
| CosSim | [1, 2, 4, 5, 6, 7, 8, 10] |
| CKA | [1, 2, 4, 5, 7, 10, 12, 15] |
| PPL | [7, 9, 10, 11, 12, 13, 14, 15] |
| Taylor | [7, 8, 10, 11, 12, 13, 14, 15] |
| Fisher | [4, 6, 8, 10, 11, 12, 13, 14] |
| IGIA | [7, 8, 10, 11, 12, 13, 14, 15] |
| GateProbe | [7, 8, 10, 11, 12, 13, 14, 15] |
| Hessian | [6, 7, 8, 10, 11, 12, 13, 14] |

## Scoring

SmolVLA's joint graph does not call HuggingFace `DecoderLayer.forward`; `SmolVLMWithExpertModel.forward` runs QKV in `forward_attn_layer` / `forward_cross_attn_layer`, then o_proj + MLP per stream. Keep-KV skip is a patch of that loop: the dropped VLM layer still computes K/V (prefix prefill writes them to the cache; denoise cross-attends to them) and its VLM residual passes through unchanged. CosSim / CKA / GateProbe read the residual at `input_layernorm` of each layer and at the final norm. All nine methods finished; Hessian (Hutchinson, math SDPA) fit on a shared 24GB card.

L15 scores exactly 0 under PPL and GateProbe: the action expert reads L15's K/V, which keep-KV preserves, and nothing downstream reads L15's residual update, so skipping it leaves the loss unchanged. `1−CKA` is near 0 for most layers (a few slightly negative, ~1e-5); only L3 and L13 stand out.

## Checks

Baseline, Drop-1 (Taylor/GateProbe/PPL and IGIA/Fisher), Drop-2 (Taylor, IGIA, Fisher, GateProbe), Drop-3 (Taylor, IGIA/Fisher, GateProbe), and all nine Drop-4 runs wrote `eval_info.json` with `n_episodes=400` (40 tasks × 10), `pc_success` matching the per-task success bits, `EXIT:0`. Logs show `n_action_steps=10`, `num_steps=10`, `seed=1000`, `control_mode=relative`, cameras `camera1` / `camera2`, action shape 7; no tracebacks. `drop.json` in each run matches the Drop-1 / Drop-2 / Drop-3 / Drop-4 tables. The GPU2 Drop-1 queue, GPU3 Drop-2 queue, GPU0 Drop-3 queue, and the Drop-4 dispatcher started each run only after the previous run on that GPU had exited.

Local full-model anchor is 83.00%, vs paper 87.3%. Use 83.00 as the referee.
