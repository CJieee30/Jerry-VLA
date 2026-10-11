# Closed-loop LIBERO results (Drop-6 / Drop-7 / Drop-8 / Drop-9, no recovery)

Importance metrics for LLM block dropping on π0.5-LIBERO-finetuned (`lerobot/pi05_libero_finetuned_v044`) / LIBERO.
Format follows DTR Table 3 (Spatial / Object / Goal / Long / Avg.).

**Protocol (not DTR’s 20-trial + recovery FT).** Official v044 eval: `n_action_steps=10`, `compile_model=false`, `seed=1000`, four suites × 10 tasks × 10 episodes (400 episodes), `batch_size=1`, no video. Skip is DTR Appendix B at runtime (keep prefix K/V, drop Q / attention / O / MLP). No recovery fine-tune. All nine Drop-6 sets and all nine independent Drop-9 sets are unique; no † merge. Greedy PPL Drop-9 is a tenth unique set `[2, 4, 7, 10, 11, 14, 15, 16, 17]` (L11 instead of independent PPL’s L9). Drop-7 / Drop-8 started as Fisher only (`redundancy_order[:7]` / `[:8]`). Taylor Drop-6 / Drop-7 / Drop-8 **without L0** take the next layers after removing 0 from `redundancy_order`; Drop-7 / Drop-8 of that list equal Fisher `[1, 2, 13–17]` / `[1, 2, 3, 13–17]`, so those two k reuse the Fisher runs. IGIA Drop-7 without L0 is a new set (adds L5 to native IGIA Drop-6). GateProbe Drop-7 / Drop-8 without L0 equal native GateProbe Drop-7 / Drop-8 because L0 is already last in `redundancy_order`: `[2, 4, 13–17]` and `[2, 4, 10, 13–17]`. **Last-6** is the tail control `[12–17]` at the same k as Fisher Drop-6. **Skip L0 only** drops `[0]`. Baseline is the same compile=false run for all k.

Source: `results/eval_drop/{baseline,k1_*,k3_*,k6_*,k7_*,k8_*,k9_*}/{eval_info.json,drop_meta.json}`. Ranks: `results/ranks/ranks.json`. Greedy PPL search: `results/ppl_subset_search/v044_drop9_greedy.md`.

Drop-3 closed-loop is paused (GPUs yielded). Five unique drop-sets; Fisher/IGIA/Hessian/Taylor and PPL/GateProbe share one 400-ep each (†). Incomplete `k3_fisher` / `k3_cossim` dirs have no `eval_info.json` and will be re-run.

Table: Importance metrics for LLM block dropping on π0.5-LIBERO, no recovery. 10 episodes per task, seed 1000. Long = LIBERO-10.

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 97.0 | 100.0 | 95.0 | 97.0 | 97.25 |
| Skip L0 only (1/18) | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| Drop-6 (6/18) | Fisher | 97.0 | 100.0 | 96.0 | 98.0 | 97.75 |
|  | Taylor (no L0) | 95.0 | 99.0 | 99.0 | 96.0 | 97.25 |
|  | IGIA | 97.0 | 99.0 | 94.0 | 98.0 | 97.00 |
|  | PPL | 99.0 | 99.0 | 92.0 | 95.0 | 96.25 |
|  | GateProbe | 97.0 | 93.0 | 94.0 | 96.0 | 95.00 |
|  | Hessian | 89.0 | 99.0 | 65.0 | 93.0 | 86.50 |
|  | Last-6 (control) | 78.0 | 91.0 | 69.0 | 56.0 | 73.50 |
|  | CKA | 8.0 | 0.0 | 0.0 | 0.0 | 2.00 |
|  | CosSim | 3.0 | 0.0 | 0.0 | 0.0 | 0.75 |
|  | Taylor | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Magnitude | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| Drop-7 (7/18) | Fisher ★ | 96.0 | 100.0 | 94.0 | 94.0 | 96.00 |
|  | Taylor (no L0) ★ | 96.0 | 100.0 | 94.0 | 94.0 | 96.00 |
|  | GateProbe (no L0) | 97.0 | 94.0 | 87.0 | 95.0 | 93.25 |
|  | IGIA (no L0) | 94.0 | 95.0 | 92.0 | 91.0 | 93.00 |
| Drop-8 (8/18) | GateProbe (no L0) | 94.0 | 87.0 | 77.0 | 85.0 | 85.75 |
|  | Fisher ★★ | 91.0 | 89.0 | 73.0 | 89.0 | 85.50 |
|  | Taylor (no L0) ★★ | 91.0 | 89.0 | 73.0 | 89.0 | 85.50 |
| Drop-9 (9/18) | Fisher | 89.0 | 87.0 | 53.0 | 78.0 | 76.75 |
|  | GateProbe | 84.0 | 88.0 | 56.0 | 64.0 | 73.00 |
|  | IGIA | 80.0 | 83.0 | 35.0 | 50.0 | 62.00 |
|  | Taylor (no L0) | 76.0 | 71.0 | 11.0 | 31.0 | 47.25 |
|  | Last-9 (control) | 22.0 | 34.0 | 16.0 | 6.0 | 19.50 |
|  | Greedy PPL | 50.0 | 72.0 | 22.0 | 44.0 | 47.00 |
|  | PPL | 46.0 | 77.0 | 13.0 | 36.0 | 43.00 |
|  | Hessian | 6.0 | 0.0 | 3.0 | 0.0 | 2.25 |
|  | Taylor | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Magnitude | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | CosSim | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | CKA | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |

Δ Avg. vs this-machine baseline (97.25):

- Skip L0 only: **−97.25**.
- Drop-6: Fisher +0.50, Taylor (no L0) **0.00**, IGIA −0.25, PPL −1.00, GateProbe −2.25, Hessian −10.75, Last-6 (control) **−23.75**, CKA −95.25, CosSim −96.50, Taylor −97.25, Magnitude −97.25.
- Drop-7: Fisher / Taylor (no L0) −1.25; GateProbe (no L0) **−4.00**; IGIA (no L0) **−4.25**.
- Drop-8: GateProbe (no L0) **−11.50**; Fisher / Taylor (no L0) −11.75.
- Drop-9: Fisher −20.50, GateProbe −24.25, IGIA −35.25, Greedy PPL −50.25, PPL −54.25, Taylor (no L0) **−50.00**, Last-9 (control) **−77.75**, Hessian −95.00, Taylor / Magnitude / CosSim / CKA −97.25.

## Drop / keep indices (PaliGemma 18 layers, 0–17)

### Skip L0 only

| Metric | Drop | Keep |
| --- | --- | --- |
| L0 | [0] | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] |

### Drop-6

| Metric | Drop | Keep |
| --- | --- | --- |
| Fisher | [1, 13, 14, 15, 16, 17] | [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| IGIA | [2, 13, 14, 15, 16, 17] | [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| PPL | [2, 9, 14, 15, 16, 17] | [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13] |
| GateProbe | [2, 4, 14, 15, 16, 17] | [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13] |
| Hessian | [2, 3, 5, 15, 16, 17] | [0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| CKA | [3, 4, 5, 6, 8, 10] | [0, 1, 2, 7, 9, 11, 12, 13, 14, 15, 16, 17] |
| CosSim | [3, 4, 5, 6, 7, 8] | [0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17] |
| Taylor | [0, 2, 14, 15, 16, 17] | [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] |
| Taylor (no L0) | [1, 2, 14, 15, 16, 17] | [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] |
| Magnitude | [0, 1, 5, 8, 9, 10] | [2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 16, 17] |
| Last-6 (control) | [12, 13, 14, 15, 16, 17] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] |

### Drop-7

Fisher `redundancy_order[:7]` adds L2. Taylor without L0 is the same set ★. IGIA without L0 adds L5 to native IGIA Drop-6. GateProbe without L0 adds L13 to native GateProbe Drop-6.

| Metric | Drop | Keep |
| --- | --- | --- |
| Fisher ★ | [1, 2, 13, 14, 15, 16, 17] | [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| Taylor (no L0) ★ | [1, 2, 13, 14, 15, 16, 17] | [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| GateProbe (no L0) | [2, 4, 13, 14, 15, 16, 17] | [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12] |
| IGIA (no L0) | [2, 5, 13, 14, 15, 16, 17] | [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12] |

### Drop-8

Fisher `redundancy_order[:8]` adds L3. Taylor without L0 is the same set ★★ (not re-run). GateProbe without L0 adds L10 to GateProbe Drop-7.

| Metric | Drop | Keep |
| --- | --- | --- |
| GateProbe (no L0) | [2, 4, 10, 13, 14, 15, 16, 17] | [0, 1, 3, 5, 6, 7, 8, 9, 11, 12] |
| Fisher ★★ | [1, 2, 3, 13, 14, 15, 16, 17] | [0, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| Taylor (no L0) ★★ | [1, 2, 3, 13, 14, 15, 16, 17] | [0, 4, 5, 6, 7, 8, 9, 10, 11, 12] |

### Drop-9

| Metric | Drop | Keep |
| --- | --- | --- |
| Fisher | [1, 2, 3, 10, 13, 14, 15, 16, 17] | [0, 4, 5, 6, 7, 8, 9, 11, 12] |
| GateProbe | [2, 4, 9, 10, 13, 14, 15, 16, 17] | [0, 1, 3, 5, 6, 7, 8, 11, 12] |
| IGIA | [1, 2, 4, 5, 13, 14, 15, 16, 17] | [0, 3, 6, 7, 8, 9, 10, 11, 12] |
| Greedy PPL | [2, 4, 7, 10, 11, 14, 15, 16, 17] | [0, 1, 3, 5, 6, 8, 9, 12, 13] |
| PPL | [2, 4, 7, 9, 10, 14, 15, 16, 17] | [0, 1, 3, 5, 6, 8, 11, 12, 13] |
| Hessian | [1, 2, 3, 4, 5, 7, 15, 16, 17] | [0, 6, 8, 9, 10, 11, 12, 13, 14] |
| Taylor | [0, 1, 2, 3, 13, 14, 15, 16, 17] | [4, 5, 6, 7, 8, 9, 10, 11, 12] |
| Taylor (no L0) | [1, 2, 3, 4, 13, 14, 15, 16, 17] | [0, 5, 6, 7, 8, 9, 10, 11, 12] |
| Last-9 (control) | [9, 10, 11, 12, 13, 14, 15, 16, 17] | [0, 1, 2, 3, 4, 5, 6, 7, 8] |
| Magnitude | [0, 1, 4, 5, 7, 8, 9, 10, 11] | [2, 3, 6, 12, 13, 14, 15, 16, 17] |
| CosSim | [3, 4, 5, 6, 7, 8, 9, 10, 11] | [0, 1, 2, 12, 13, 14, 15, 16, 17] |
| CKA | [2, 3, 4, 5, 6, 7, 8, 9, 10] | [0, 1, 11, 12, 13, 14, 15, 16, 17] |

## Checks

All Drop-6 (9 methods + baseline) and Drop-9 (9 independent methods + greedy PPL) jobs wrote `eval_info.json`. Four Drop-9 tmux queues ended `failed=0`; greedy PPL ended `exit=0`. Fisher Drop-7 (`k7_fisher`) wrote `eval_info.json`, 40 tasks × 10 episodes, `pc_success` 96.00 matching 384/400 bits, `exit=0`. Fisher Drop-8 (`k8_fisher`) wrote `eval_info.json`, `pc_success` 85.50 matching 342/400 bits, `exit=0`. Taylor Drop-6 without L0 (`k6_taylor_nol0`) wrote `eval_info.json`, `pc_success` 97.25 matching 389/400 bits, `exit=0`, drop `[1, 2, 14, 15, 16, 17]`. IGIA Drop-7 without L0 (`k7_igia_nol0`) wrote `eval_info.json`, `pc_success` 93.00 matching 372/400 bits, `exit=0`, drop `[2, 5, 13, 14, 15, 16, 17]`. GateProbe Drop-7 without L0 (`k7_gateprobe_nol0`) wrote `eval_info.json`, `pc_success` 93.25 matching 373/400 bits, `exit=0`, drop `[2, 4, 13, 14, 15, 16, 17]`. GateProbe Drop-8 without L0 (`k8_gateprobe_nol0`) wrote `eval_info.json`, `pc_success` 85.75 matching 343/400 bits, `exit=0`, drop `[2, 4, 10, 13, 14, 15, 16, 17]`. Last-6 (`k6_last`) wrote `eval_info.json`, `pc_success` 73.50 matching 294/400 bits, `exit=0`, drop `[12, 13, 14, 15, 16, 17]`. Skip L0 only (`k1_l0`) wrote `eval_info.json`, `pc_success` 0.00 matching 0/400 bits, `exit=0`, drop `[0]`. Taylor Drop-7 / Drop-8 without L0 equal Fisher and were not re-run. Each job has four suites, `compile_model=false`, `n_action_steps=10`, `seed=1000`, `recovery=false`, no videos. Independent drop lists match `ranks.json` at that k except Drop-7 / Drop-8 (Fisher `redundancy_order[:7]` / `[:8]`, not `ranks.json` keys); greedy PPL `[2, 4, 7, 10, 11, 14, 15, 16, 17]` comes from the subset search, not `ranks.json`. Greedy PPL closed-loop Avg 47.00 vs independent PPL 43.00 (+4.00); both remain well below Fisher 76.75. Taylor Drop-9 without L0 (`k9_taylor_nol0`) wrote `eval_info.json`, 40 tasks × 10 episodes, `pc_success` 47.25 matching 189/400 bits, `EXIT:0`, drop `[1, 2, 3, 4, 13, 14, 15, 16, 17]`. Last-9 (`k9_last`) wrote `eval_info.json`, `pc_success` 19.50 matching 78/400 bits, `EXIT:0`, drop `[9, 10, 11, 12, 13, 14, 15, 16, 17]`. Do not overwrite `k9_taylor` (native, contains L0, 0.00). CARE v2 high-α GateProbe Drop-6 (`k6_gateprobe_v2_hialpha`) wrote `eval_info.json`, `pc_success` 97.25 matching 389/400 bits, `EXIT:0`, drop `[2, 3, 14, 15, 16, 17]`. Do not overwrite Stage-1 GateProbe 95.00.

Local full-model anchor is 97.25% (389/400), vs Hub v044 97.5% under `compile_model=true`. Difference is one episode; use 97.25 as the referee for both k, not the Hub number.

Collapsed runs take ~3.5–3.9 h/job (full horizon); methods that still succeed take ~2.0–2.8 h at Drop-9. That pattern matches success rate, not a hung eval.

Do not read closed-loop `eval_s` as efficiency. MuJoCo dominates wall time. Prefix skip savings are measured on `predict_action_chunk` only (GPU1 RTX 3090, `compile_model=false`, `n_action_steps=10`, 10 denoise steps, one calib observation, CUDA events warmup=5 / repeats=20). Source: `results/eval_drop/efficiency_fisher_k6.json`.

Table: Fisher Drop-6 `[1, 13, 14, 15, 16, 17]` vs this-machine baseline. Skip is Appendix B (keep K/V). FLOPs are PyTorch `FlopCounterMode` on one `predict_action_chunk`. **Stored params do not change** (weights stay in memory). Active params subtract unused language Q / O / MLP / post-attn LN on the six dropped layers.

| Setting | LIBERO Avg. | Params stored (M) | Params active (M) | Inference (ms) | FLOPs (G) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 97.25 | 4143.40 | 4143.40 | 541.22 | 5024.4 |
| Fisher Drop-6 | 97.75 | 4143.40 | 3489.08 | 521.19 | 3711.6 |
| Δ | **+0.50** | 0.00 (0%) | **−654.32 (−15.79%)** | **−20.03 (−3.70%)** | **−1312.8 (−26.13%)** |

Language-only active params: 2508.53 → 1854.21 M (−26.08%). Expert denoise and SigLIP are unchanged, which is why chunk latency only drops ~4% while measured FLOPs drop ~26%. An older GateProbe Drop-6 profile (`efficiency_k6.json`) matches these FLOPs (uniform layer width); do not re-use its millisecond column (GPU was contended).

## CARE (Stage-2): action-discontinuity reweighting (2026-10-05)

Full write-up: `main_results/care_libero.md`.

No closed-loop. Weights from the first 10 valid steps of the stored 50-step normalized chunk. Fisher still reduces at the original 64×8 batches (4× mb=2 forwards, weighted-mean gradient, then square; 64 batches equal-weighted). α=1 on batch 0 matched the Stage-1 orig-batch Fisher contribution exactly (`rel_abs_diff=0`).

Weighted Fisher Drop-6 no-L0 is still `[1, 13, 14, 15, 16, 17]`. Jaccard vs Stage-1 = **1.0**. Stop. Do not re-evaluate 97.75.

The full redundancy order is not identical: Stage-1 `[17, 16, 15, 13, 14, 1, 2, 3, 10, 12, 4, …]`; weighted `[17, 16, 15, 13, 14, 1, 2, 3, 4, 10, 12, …]`. The operating-point set does not change. Source: `results/stage2_action_weight/`.

Weighted GateProbe (same orig-batch reduction; α=1 signed inner on batch 0 matches Stage-1 with `rel_abs_diff=0`). Drop-6 no-L0 moves from `[2, 4, 14, 15, 16, 17]` to `[2, 13, 14, 15, 16, 17]` (L4 ↔ L13). Jaccard vs Stage-1 = **0.714**. L0 remains last. Closed-loop (400 ep, seed=1000, EXIT:0):

| Setting | Drop | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GateProbe Stage-1 (historical) | `[2, 4, 14–17]` | 97.0 | 93.0 | 94.0 | 96.0 | 95.00 |
| GateProbe Stage-1 reval (GPU0) | `[2, 4, 14–17]` | 97.0 | 93.0 | 94.0 | 93.0 | 94.25 |
| GateProbe weighted (GPU1) | `[2, 13–17]` | 97.0 | 99.0 | 94.0 | 98.0 | **97.00** |

Reval vs historical: −0.75 (noise). Weighted vs contemporaneous reval: **+2.75**. Weighted vs baseline 97.25: **−0.25**. The new set holds the ceiling; swapping L13 in for L4 is not a fluctuation. Do not overwrite the Stage-1 GateProbe 95.00 row. Sources: `results/eval_drop/{k6_gateprobe,k6_gateprobe_reval,k6_gateprobe_action_discontinuity}/eval_info.json`.

### CARE v2 high-α quartile GateProbe Drop-6 (GPU1, 2026-10-10)

Not α-weighted mean (that set is still `[2, 13–17]`). Mean of the 120 frames in the top quartile of whitened-7d α on the shared 480-frame set. Drop `[2, 3, 14, 15, 16, 17]` (L13 ↔ L3 vs v2 uniform). 400 ep, seed=1000, `compile_model=false`, `n_action_steps=10`, keep-KV, `EXIT:0`. Do not overwrite Stage-1 GateProbe 95.00 or the CARE weighted 97.00 row.

| Setting | Drop | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GateProbe Stage-1 (historical) | `[2, 4, 14–17]` | 97.0 | 93.0 | 94.0 | 96.0 | 95.00 |
| GateProbe Stage-1 reval | `[2, 4, 14–17]` | 97.0 | 93.0 | 94.0 | 93.0 | 94.25 |
| v2 uniform / CARE-weighted | `[2, 13–17]` | 97.0 | 99.0 | 94.0 | 98.0 | 97.00 |
| v2 high-α quartile (GPU1) | `[2, 3, 14–17]` | 97.0 | 99.0 | 93.0 | 100.0 | **97.25** |

High-α vs v2 uniform: **+0.25** (1 episode, noise). High-α vs baseline 97.25: **0.00** (389/400). High-α vs Fisher-6 97.75: **−0.50**. Jaccard vs `[2, 13–17]` = 0.71. Wall `eval_s` 6192 s (~1 h 43 m). Source: `results/eval_drop/k6_gateprobe_v2_hialpha/eval_info.json`.

CARE IGIA Drop-6 is still `[2, 13, 14, 15, 16, 17]`. Jaccard vs Stage-1 = **1.0**. Stop. Do not re-evaluate 97.00. Drop-7 no-L0 is still `[2, 5, 13–17]` (Jaccard 1.0 vs 93.00). α=1 on batch 0 matches Stage-1 orig-batch IGIA (`rel_abs_diff=0`). Source: `results/stage2_action_weight/pi05_igia_action_discontinuity.json`.

CARE Drop-7 / Drop-8 no-L0 GateProbe (same weights; 400 ep, seed=1000, EXIT:0). No Stage-1 reval this round. Do not overwrite `k7_gateprobe_nol0` / `k8_gateprobe_nol0`.

| Setting | Drop | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GateProbe Drop-7 Stage-1 | `[2, 4, 13–17]` | 97.0 | 94.0 | 87.0 | 95.0 | 93.25 |
| CARE Drop-7 (GPU0) | `[2, 9, 13–17]` | 97.0 | 97.0 | 88.0 | 94.0 | **94.00** |
| GateProbe Drop-8 Stage-1 | `[2, 4, 10, 13–17]` | 94.0 | 87.0 | 77.0 | 85.0 | 85.75 |
| CARE Drop-8 (GPU2) | `[2, 4, 9, 13–17]` | 97.0 | 88.0 | 79.0 | 88.0 | **88.00** |

Drop-7 vs historical: **+0.75** (noise). Still **−2.00** vs Fisher Drop-7 96.00. Drop-8 vs historical: **+2.25**; also **+2.50** vs Fisher Drop-8 85.50. Fisher / Taylor Drop-8 remain Jaccard 1.0 (do not re-eval 85.50). Sources: `results/eval_drop/{k7_gateprobe_nol0,k7_gateprobe_action_discontinuity,k8_gateprobe_nol0,k8_gateprobe_action_discontinuity}/eval_info.json`. Full write-up: `care_libero.md` §7.1.
