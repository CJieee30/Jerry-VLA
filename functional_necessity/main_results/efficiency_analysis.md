# Efficiency: `predict_action_chunk` (not LIBERO wall time)

Protocol matches the π0.5 measurements: RTX 3090, `compile_model=false`, CUDA events warmup=5 / repeats=20, one calibration observation, PyTorch `FlopCounterMode`. Stored weights are not deleted. **Active params** subtract only the compute the skip operator actually skips.

Skip operators differ by stack and must not be mixed:

- π0.5: DTR Appendix B (keep prefix K/V; skip language Q / attention / O / MLP).
- MolmoAct2 / SmolVLA in §1–§2 and §4: keep-KV identity residual that still ran Q and attention (Molmo skipped MLP + `ff_norm`; SmolVLA skipped VLM `o_proj` + MLP + post-attn LN). §3 language tables and §5–§6: K/V-only Appendix B (only write K/V; skip Q, attention, output projection, and MLP). The action expert is never dropped.

Table 1 time / FLOP shares are over the three isolated modules (Vision + Language + Action), not over `predict_action_chunk` wall time. Do not read closed-loop `eval_s` as efficiency; MuJoCo dominates LIBERO wall time.

Source:

- π0.5: `results/eval_drop/{efficiency_pi05_baseline_modules,efficiency_fisher_k6,efficiency_pi05_drop6_language_only}.json`
- MolmoAct2 Taylor Drop-16 `[1–4,24–35]`, §1–§2 / §4: `results/molmoact2_libero/efficiency_taylor_drop16.json`
- SmolVLA Taylor Drop-4 `[12–15]`, §1–§2 / §4: `results/smolvla_libero/efficiency_taylor_drop4.json`
- §3 Molmo / SmolVLA language tables and §5: `results/stage_nfe/sota_efficiency_{molmoact2,smolvla}_kvonly.json`
- §4 paired remeasure (do not mix with the rows above): `results/stage_nfe/sota_efficiency_{pi05,molmoact2,smolvla}.json`

Closed-loop Avg. is from the corresponding `eval_*.md` (this-machine anchors). §1–§2 and the π0.5 row of §3 are the original NFE=10 reference. §3 Molmo / SmolVLA language tables are K/V-only from the §5 session; do not mix those milliseconds with Table 1. §4 is a separate paired run for the operating point.

## Eager runtime K/V-only skip — multi-input diagnostic (2026-10-17)

**Label:** exploratory only — **not** deployment latency. Raw JSON unchanged: `results/stage_nfe/formal_efficiency_{pi05,molmoact2}_20261017.json`.

**Protocol** (`measure_formal_efficiency.py`): both stacks use **runtime** keep-KV skip (not materialize); Molmo `enable_inference_cuda_graph=false`; 32 calib obs × 3 rounds; sync wall `predict_action_chunk`. **Active params** in the table are *analytical* skipped-compute counts — **not** stored weights and **not** materialized VRAM.

| Stack | Config | Median (ms) | P95 (ms) | vs Full NFE10 (same session) | Active params (M)† | Peak alloc (GiB)† |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| π0.5 | Full NFE10 | 564.5 | 688.4 | 1.00 | 4143.4 | 8.89 |
| π0.5 | Fisher Drop-6 NFE10 | 537.6 | 665.1 | 1.05 | 3489.1 | 8.88 |
| π0.5 | Full NFE1 | 211.7 | 219.9 | 2.67 | 4143.4 | 8.89 |
| π0.5 | Drop-6 NFE1 | 185.9 | 195.3 | 3.04 | 3489.1 | 8.88 |
| MolmoAct2 | Full NFE10 | 1179.6 | 1222.2 | 1.00 | 5442.2 | 12.66 |
| MolmoAct2 | Taylor Drop-16 NFE10 | 1178.7 | 1223.6 | 1.00 | 3911.2 | 12.31 |
| MolmoAct2 | Full NFE1 | 261.8 | 281.7 | 4.51 | 5442.2 | 12.66 |
| MolmoAct2 | Drop-16 NFE1 | 244.1 | 260.9 | 4.83 | 3911.2 | 12.30 |

†Do **not** use for materialized deployment claims.

**Stability (diagnostic):** π0.5 Full NFE10 **per-round medians** ≈ **550 / 557 / 669 ms** (~**21%** spread round 0→2); a single end recheck drift (~1.4%) **does not** prove run stability. Molmo recheck drift ~2.3% — same caveat.

**NFE=1 pruning delta (same diagnostic JSON, pooled medians):** vs matched Full NFE1, median latency drops **12.2%** (π0.5 Drop-6) and **6.8%** (Molmo Drop-16). **Do not** mix these ms with the deployment table below for speedup ratios.

## Formal deployment wall clock (2026-10-17)

**Protocol** (`measure_formal_deployment_efficiency.py`, `formal_deployment_20261017/`): π0.5 **Full = eager**; **Pruned = materialized Fisher Drop-6** (`materialize.py`). Molmo target stack is **materialized Taylor Drop-16 + native action-expert CUDA Graph** on Full/Pruned; **headline Molmo milliseconds below are pass-1** (see ‡). No `torch.compile`. Each config: **separate process**, fresh load; 32 obs with fingerprints; warmup 10; 3 rounds; `nvidia-smi` keyed by **physical GPU index/UUID** (not confused with `cuda:0` under `CUDA_VISIBLE_DEVICES`). Stored **params_stored_m** = actual `numel` after materialize.

**Results (physical GPU 2, UUID in per-config JSON):**

| Stack | Config | Median (ms) | P95 | vs anchor NFE10 | Stored params (M) | Peak alloc (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| π0.5 | Full NFE10 eager | 547.7 | 559.1 | 1.00 | 4143.4 | 8.89 |
| π0.5 | Drop-6 NFE10 materialized | 524.2 | 543.4 | 1.04 | 3489.1 | **7.67** |
| π0.5 | Full NFE1 eager | 186.2 | 192.9 | 2.94 | 4143.4 | 8.89 |
| π0.5 | Drop-6 NFE1 materialized | 161.5 | 167.3 | 3.39 | 3489.1 | **7.67** |
| Molmo | Full NFE10 ‡ | 1178.1 | 1199.4 | 1.00 | 5442.2 | 12.31 |
| Molmo | Drop-16 NFE10 materialized ‡ | 1171.6 | 1198.1 | 1.01 | 3911.2 | **9.45** |
| Molmo | Full NFE1 ‡ | 264.2 | 281.1 | 4.46 | 5442.2 | 12.30 |
| Molmo | Drop-16 NFE1 materialized ‡ | 243.3 | 250.6 | 4.84 | 3911.2 | **9.45** |

‡**Molmo pass-1 (`summary_molmoact2_pass1_eager_fallback.json`, 2026-10-17):** `action_cuda_graph_manager` was nulled in the disable path → timed path is **eager expert** (config flag true but `graph_fallback` in that run’s configs; per-config JSON since overwritten by graph reruns). **Pass-3** re-enabled the manager (`manager_present=true`, `max_abs_vs_eager=0` on 3 fixed obs) but each timed `predict_action_chunk` calls `policy.reset()`, so **~22%** of samples still hit **>500 ms** graph recapture (P95 ≈2.6 s) while the pooled median collapses to **~242 ms** — **not** interchangeable with pass-1 wall clock. **Use pass-1 for deployment headline latency** until graph timing matches eval (`reset` + capture policy) or Plus enables graph.

**Stability (deployment):** π0.5 Full NFE10 round medians **547.3 / 548.1 / 548.0 ms** (~0.15% spread); anchor recheck drift **2.0%** (≤3% practical target). Molmo pass-1 Full NFE10 round medians **1180.4 / 1177.9 / 1175.7 ms** (~0.4% spread); anchor recheck drift **7.7%** (>5% — keep raw JSON, no slow-round deletion). **Pruned vs Full NFE1 (pass-1 medians):** π0.5 **−13.3%**; Molmo **−7.9%**.

Archives: `summary_molmoact2_pass2_graph_capture_in_timed.json`, `summary_molmoact2_pass1_eager_fallback.json`; latest graph attempt configs `configs/molmo_*_graph.json`. **Do not** mix deployment ms with § diagnostic eager JSON for speedups.

Raw: `summary_pi05.json`, `summary_molmoact2.json` (Molmo = pass-1). **Closed-loop SR not remeasured.**

---



## 1. Baseline module split

These three tables are the **unpruned** full models (no layer skip). Vision / Language / Action are isolated phases of one `predict_action_chunk` on that baseline. Dropped-model numbers are only in §2 and §3.



### π0.5


| π0.5 baseline  | Vision  | Language | Action |
| -------------- | ------- | -------- | ------ |
| #Param (M)     | 414.8   | 2508.5   | 693.4  |
| Vision tokens  | 256 × 3 | 768      | —      |
| Denoising      | —       | —        | 10     |
| Inference (ms) | 56.0    | 78.6     | 387.1  |
| 时间占比           | 10.7%   | 15.1%    | 74.2%  |
| FLOPs (G)      | 660.53  | 3974.95  | 388.92 |
| FLOP 占比        | 13.1%   | 79.1%    | 7.7%   |


3 camera views × 256 tokens. Language FLOPs are the PaliGemma prefix (seq 968). Action is 10 Euler denoise steps on the Gemma expert. Table-1 Language 78.6 ms and Table-3 Language 78.10 ms are two CUDA-event runs of the same isolation.

### MolmoAct2


| MolmoAct2 baseline | Vision  | Language | Action |
| ------------------ | ------- | -------- | ------ |
| #Param (M)         | 439.1   | 4029.7   | 577.6  |
| Vision tokens      | 196 × 2 | 392      | —      |
| Denoising          | —       | —        | 10     |
| Inference (ms)     | 38.4    | 81.3     | 1019.7 |
| 时间占比               | 3.4%    | 7.1%     | 89.5%  |
| FLOPs (G)          | 1291.18 | 3820.07  | 82.15  |
| FLOP 占比            | 24.9%   | 73.6%    | 1.6%   |


2 LIBERO cameras. `flow_matching_num_steps=10`, `chunk_size=n_action_steps=10`, `enable_inference_cuda_graph=false`. Language is the 36-layer VLM prefix (seq 505) after vision features are already in the embeddings. Action is 10 flow-matching steps of the 36-layer expert (cross-attends to every VLM layer's K/V). Isolated-module FLOPs sum to 5193.40 G; full `predict_action_chunk` is 5250.59 G (packing / depth-gate / embeddings outside the three buckets).

### SmolVLA


| SmolVLA baseline | Vision | Language | Action |
| ---------------- | ------ | -------- | ------ |
| #Param (M)       | 98.2   | 204.6    | 99.8   |
| Vision tokens    | 64 × 2 | 128      | —      |
| Denoising        | —      | —        | 10     |
| Inference (ms)   | 22.2   | 45.4     | 415.8  |
| 时间占比             | 4.6%   | 9.4%     | 86.0%  |
| FLOPs (G)        | 430.64 | 57.60    | 110.19 |
| FLOP 占比          | 72.0%  | 9.6%     | 18.4%  |


`lerobot/smolvla_libero`, nas=10, `num_steps=10`, `chunk_size=50`, 2 LIBERO cameras after `image→camera1` / `image2→camera2`. Language is the 16-layer VLM prefix (seq 177). Action is 10 Euler steps of the expert. Isolated-module FLOPs sum to 598.43 G, matching full `predict_action_chunk`.

---



## 2. Shared drop comparison (active params / chunk)

Closed-loop skip on `predict_action_chunk`. Param is **active** (stored params do not change). Success Rate is LIBERO Avg. vs this-machine anchor.


|                     | π0.5             | MolmoAct2        | SmolVLA         |
| ------------------- | ---------------- | ---------------- | --------------- |
| 冗余层数                | 6/18             | 16/36            | 4/16            |
| Param (M)           | 3489.08（−15.79%） | 4246.78（−21.97%） | 416.86（−7.37%）  |
| Inference Time (ms) | 521.19（−3.70%）   | 1158.76（−1.03%）  | 485.10（−1.64%）  |
| FLOPs (G)           | 3711.6（−26.13%）  | 4043.3（−22.99%）  | 586.7（−1.96%）   |
| Success Rate        | 97.75（+0.50%）    | 98.10（+0.60%）    | 78.50（−4.50%）   |


π0.5: Fisher Drop-6 `[1,13–17]` vs anchor 97.25. MolmoAct2: Taylor Drop-16 no L0 `[1–4,24–35]` vs anchor 97.50. SmolVLA: Taylor Drop-4 `[12–15]` vs anchor 83.00. Chunk latency barely moves because Action dominates wall time (Table 1); FLOPs drop with the language prefix. SmolVLA’s language is only 9.6% of chunk FLOPs, so Drop-4 saves ~2% of the full chunk.

---



## 3. Language module only

Prefix language forward only (vision embeddings cached; expert denoise not run). π0.5 is DTR Appendix B (`efficiency_pi05_drop6_language_only.json`). MolmoAct2 and SmolVLA are K/V-only (`sota_efficiency_{molmoact2,smolvla}_kvonly.json`): dropped layers write K/V and do not run Q, attention, the output projection, or the MLP. Those two tables replace the earlier keep-KV rows that still ran Q and attention (Molmo −29.7% / −16.0% / −31.6%; SmolVLA −16.2% / −8.2% / −20.4%). Do not mix Molmo / SmolVLA milliseconds here with Table 1 or §4.

### π0.5 language module


| π0.5 language module | Params active (M) | Inference (ms) | FLOPs (G)        |
| -------------------- | ----------------- | -------------- | ---------------- |
| Baseline 18/18       | 2508.53           | 78.10          | 3974.95          |
| Drop-6 12/18         | 1854.21           | 53.43          | 2662.15          |
| Δ                    | −654.32（−26.1%）   | −24.67（−31.6%） | −1312.80（−33.0%） |




### MolmoAct2 language module


| MolmoAct2 language module | Params active (M) | Inference (ms) | FLOPs (G)        |
| ------------------------- | ----------------- | -------------- | ---------------- |
| Baseline 36/36            | 4029.68           | 88.68          | 3820.07          |
| Drop-16 20/36             | 2498.71           | 71.01          | 2206.99          |
| Δ                         | −1530.96（−38.0%）  | −17.68（−19.9%） | −1613.08（−42.2%） |


Language FLOPs Δ equals the full-chunk FLOPs Δ (−1613.08 G): K/V-only skip removes Q, attention, the output projection, and the MLP. Active params also exclude Q and `att_proj` (the old skip only subtracted MLP + `ff_norm`, −29.7%). Prefix time −19.9% does not show up in Table 2 because that table is the pre-K/V-only session and Action is 89.5% of module time. Same numbers, 1-decimal rounding: §5 language 88.7 → 71.0 ms (−20.0%) and 3820.1 → 2207.0 G (−42.2%).

### SmolVLA language module


| SmolVLA language module | Params active (M) | Inference (ms) | FLOPs (G)      |
| ----------------------- | ----------------- | -------------- | -------------- |
| Baseline 16/16          | 204.63            | 47.93          | 57.60          |
| Drop-4 12/16            | 167.76            | 39.82          | 44.07          |
| Δ                       | −36.87（−18.0%）    | −8.11（−16.9%）  | −13.53（−23.5%） |


Language is only 9.6% of SmolVLA chunk FLOPs (Table 1). Taylor Drop-4 K/V-only saves 23.5% of the VLM prefix (the old skip was −20.4% FLOPs and −8.2% time because dropped layers still ran Q/K/V). Full-chunk FLOPs in the same session fall 598.4 → 584.9 G (−2.3%). That is expected for a 16-layer VLM sitting next to a larger vision encoder and a 10-step expert.

---

## 4. Operating point: layer drop × reduced NFE

Paired remeasure on one RTX 3090, one calibration frame, same inference noise, `compile_model=false`, CUDA events warmup=5 / repeats=20, `FlopCounterMode`. Each model is one process. `vs Full` is against that process's full model at NFE=10. Params are active params (stored weights do not change; NFE does not change params). Success rate is the existing closed-loop, not re-run. No prefix trim. π0.5 and MolmoAct2 use NFE=1. SmolVLA uses NFE=2.

Source: `results/stage_nfe/sota_efficiency_{pi05,molmoact2,smolvla}.json`. Script: `scripts/stage_nfe/measure_sota_efficiency.py`. SmolVLA's operating NFE is 2, not 1.

### π0.5 — Fisher Drop-6 × NFE=1

Drop `[1, 13–17]`, DTR Appendix B. All four success rates are 400 episodes.

| Setting | Layers | NFE | Params (M) | Time (ms) | vs Full | FLOPs (G) | vs Full | SR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 18/18 | 10 | 4143.4 | 524.3 | — | 5024.4 | — | 97.25 |
| Fisher Drop-6 | 12/18 | 10 | 3489.1 (−15.8%) | 502.0 | −4.3% | 3711.6 | −26.1% | 97.75 |
| Full | 18/18 | 1 | 4143.4 | 181.1 | −65.5% | 4674.4 | −7.0% | 97.00 |
| **Fisher Drop-6** | **12/18** | **1** | **3489.1 (−15.8%)** | **155.9** | **−70.3%** | **3361.6** | **−33.1%** | **96.50** |

Drop-6 at NFE=10 removes 26.1% of FLOPs and 4.3% of chunk time. NFE=1 on the full model removes 65.5% of time and 7.0% of FLOPs. The operating point is both: 155.9 ms (−70.3%) and 3361.6 G (−33.1%), SR 96.50 versus 97.25.

SigLIP identity skip is not a row in this table. A paired remeasure (`results/stage_nfe/vision_drop4_efficiency.json`, `eval_pi05_vision_module_libero.md` §6) finds vision Drop-4 removes 97.2 G and 7–9 ms: −1.3% of the full NFE=10 chunk, −5.6% of Drop-6×NFE=1. That is not a fifth operating-point stage.

### MolmoAct2 — Taylor Drop-16 × NFE=1

Drop `[1–4, 24–35]`, keep-KV. NFE=10 success rates are 2000 episodes. † is 400 episodes, so those two rates are not a same-precision delta against 97.50 / 98.10.

| Setting | Layers | NFE | Params (M) | Time (ms) | vs Full | FLOPs (G) | vs Full | SR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 36/36 | 10 | 5442.2 | 1166.6 | — | 5250.6 | — | 97.50 |
| Taylor Drop-16 | 20/36 | 10 | 4246.8 (−22.0%) | 1153.0 | −1.2% | 4043.3 | −23.0% | 98.10 |
| Full | 36/36 | 1 | 5442.2 | 253.4 | −78.3% | 5176.7 | −1.4% | 98.25† |
| **Taylor Drop-16** | **20/36** | **1** | **4246.8 (−22.0%)** | **240.4** | **−79.4%** | **3969.3** | **−24.4%** | **99.25†** |

Drop-16 at NFE=10 removes 23.0% of FLOPs and 1.2% of chunk time. NFE=1 on the full model removes 78.3% of time and 1.4% of FLOPs. The operating point is 240.4 ms (−79.4%) and 3969.3 G (−24.4%).

### Module split at the operating point

Same run as the tables above. Shares use Vision + Language + Action as the denominator, not `predict_action_chunk` wall time. Vision is measured once per model: neither the skip nor NFE changes it.

#### π0.5

| Module | Full, NFE=10 | | | Drop-6, NFE=1 | | |
|---|---:|---:|---:|---:|---:|---:|
| | Time (ms) | Share | FLOPs (G) | Time (ms) | Share | FLOPs (G) |
| Vision | 56.2 | 10.8% | 660.53 | 56.2 | 37.8% | 660.53 |
| Language | 79.6 | 15.3% | 3974.95 | 54.3 | 36.5% | 2662.15 |
| Action | 383.7 | 73.9% | 388.92 | 38.1 | 25.7% | 38.89 |
| Sum | 519.5 | 100% | 5024.40 | 148.6 | 100% | 3361.57 |

Chunk time sits 4.9 ms (full) and 7.3 ms (operating point) above the three-module sum. FLOPs match the chunk exactly.

Action falls from 73.9% of module time to 25.7%. No module is then above 40%. Vision and language, which still run once per chunk, are 37.8% and 36.5%. The old single bottleneck is gone; what remains is the prefix.

#### MolmoAct2

| Module | Full, NFE=10 | | | Drop-16, NFE=1 | | |
|---|---:|---:|---:|---:|---:|---:|
| | Time (ms) | Share | FLOPs (G) | Time (ms) | Share | FLOPs (G) |
| Vision | 37.2 | 3.3% | 1291.18 | 37.2 | 18.0% | 1291.18 |
| Language | 81.7 | 7.2% | 3820.07 | 67.5 | 32.7% | 2612.74 |
| Action | 1022.0 | 89.6% | 82.15 | 101.8 | 49.3% | 8.21 |
| Sum | 1140.9 | 100% | 5193.40 | 206.5 | 100% | 3912.13 |

Chunk time sits 25.8 ms (full) and 33.8 ms (operating point) above the sum. Chunk FLOPs sit 57.2 G above the sum on both rows (packing, depth gate, embeddings outside the three buckets), the same gap as §1.

Action falls from 89.6% of module time to 49.3%. It is still the largest single module, and it is no longer most of the chunk. Language is 32.7% and vision is 18.0%. One expert step is the largest remaining piece; there is no second module at the old 90% share.

### SmolVLA — Taylor Drop-4 × NFE=2

Drop `[12–15]`, keep-KV. `num_steps` is the Euler count. All four success rates are 400 episodes. The full-model NFE=2 row is the same paired run; it is not a second operating point.

| Setting | Layers | NFE | Params (M) | Time (ms) | vs Full | FLOPs (G) | vs Full | SR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 16/16 | 10 | 450.0 | 556.3 | — | 598.4 | — | 83.00 |
| Taylor Drop-4 | 12/16 | 10 | 416.9 (−7.4%) | 573.0 | +3.0% | 586.7 | −2.0% | 78.50 |
| Full | 16/16 | 2 | 450.0 | 173.8 | −68.8% | 510.3 | −14.7% | 82.00 |
| **Taylor Drop-4** | **12/16** | **2** | **416.9 (−7.4%)** | **181.8** | **−67.3%** | **498.5** | **−16.7%** | **77.00** |

Drop-4 at NFE=10 removes 2.0% of FLOPs and does not reduce chunk time (+3.0% in this run). NFE=2 on the full model removes 68.8% of time and 14.7% of FLOPs. The operating point is 181.8 ms (−67.3%) and 498.5 G (−16.7%), SR 77.00 versus 83.00. Chunk time follows the Euler count. The layer skip does not make Drop-4 × NFE=2 faster than the full model at NFE=2 (173.8 ms).

#### SmolVLA module split

Same run. Shares use Vision + Language + Action as the denominator. Vision is measured once: neither the skip nor NFE changes it. Action at the operating point is the Drop-4 prefix cache with 2 Euler steps; the expert itself is not dropped.

| Module | Full, NFE=10 | | | Drop-4, NFE=2 | | |
|---|---:|---:|---:|---:|---:|---:|
| | Time (ms) | Share | FLOPs (G) | Time (ms) | Share | FLOPs (G) |
| Vision | 25.4 | 4.5% | 430.64 | 25.4 | 15.2% | 430.64 |
| Language | 49.9 | 8.9% | 57.60 | 48.8 | 29.2% | 45.86 |
| Action | 485.2 | 86.6% | 110.19 | 93.1 | 55.6% | 22.04 |
| Sum | 560.5 | 100% | 598.43 | 167.3 | 100% | 498.54 |

Chunk FLOPs match the three-module sum (598.4 G and 498.5 G). Chunk time sits 4.2 ms below the sum on the full model and 14.5 ms above it at the operating point.

Action falls from 86.6% of module time to 55.6%. It is still the largest module. Language time barely moves (49.9→48.8 ms) while its FLOPs fall 57.60→45.86 G, because keep-KV still runs Q/K/V. Vision stays 430.64 G and becomes 15.2% of module time once the expert is two steps.

---

## 5. K/V-only skip (2026-10-09)

MolmoAct2 and SmolVLA dropped layers now follow π0.5 Appendix B: input norm (pre-norm Molmo only), K/V, RoPE, cache write. Q, attention, the output projection, and the MLP do not run. The action expert is not dropped. §1, §2, and §4 stay as measured; those Molmo/SmolVLA rows still ran Q and attention on dropped layers and then discarded the output. §3 language tables use this session. Do not subtract milliseconds across §4 and §5. Full-model FLOPs match (§4 5250.6 / 5176.7 G and 598.4 / 510.3 G). The full-model wall clock in this session is slower, so every delta below is inside `sota_efficiency_{molmoact2,smolvla}_kvonly.json`.

Checked on one calibration frame before the timing run. SmolVLA actions and the frozen-noise flow loss match the previous skip exactly (`max_abs=0`, loss 0.027666). Dropped `q_proj` is not called. Molmo `att_proj` is not called. Its K/V come from the K and V rows of the fused projection, so they differ from a slice of the fused GEMM by a few bf16 ulps; qwen3 K-norm makes the largest key element differ by 0.031. On that frame, NFE=1 actions differ by at most **7.8e-3**, and the flow loss moves 0.638279 → 0.631221. No closed-loop rerun. Existing success rates stay on the §4 rows.

GPU1, `compile_model=false`, CUDA events warmup=5 / repeats=20, one calibration frame. Active params now also exclude Q and the attention output projection.

### MolmoAct2 Taylor Drop-16

| Setting | NFE | Params (M) | Time (ms) | vs full NFE=10 | FLOPs (G) | vs full NFE=10 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 10 | 5442.2 | 1394.5 | — | 5250.6 | — |
| Drop-16 | 10 | 3911.2 (−28.1%) | 1331.4 | −4.5% | 3637.5 | −30.7% |
| Full | 1 | 5442.2 | 293.4 | −79.0% | 5176.7 | −1.4% |
| **Drop-16** | **1** | **3911.2 (−28.1%)** | **276.6** | **−80.2%** | **3563.6** | **−32.1%** |

Same NFE=1: 276.6 vs 293.4 ms (**−5.7%**), 3563.6 vs 5176.7 G (**−31.2%**). §4 was −5.1% and −23.3%. The extra FLOPs are 405.7 G, all in the language prefix.

| Module | Full | | Drop-16 | |
|---|---:|---:|---:|---:|
| | Time (ms) | FLOPs (G) | Time (ms) | FLOPs (G) |
| Language | 88.7 | 3820.1 | 71.0 (−20.0%) | 2207.0 (−42.2%) |

§4 language FLOPs were 3820.1 → 2612.7 G (−31.6%) because dropped layers still ran attention. Language time in that run was −17%. This run is −20.0% time and −42.2% FLOPs. The chunk percentage stays near −6% because one expert step is still about 123 ms.

### SmolVLA Taylor Drop-4

| Setting | NFE | Params (M) | Time (ms) | vs full NFE=10 | FLOPs (G) | vs full NFE=10 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 10 | 450.0 | 546.5 | — | 598.4 | — |
| Drop-4 | 10 | 413.2 (−8.2%) | 538.2 | −1.5% | 584.9 | −2.3% |
| Full | 2 | 450.0 | 173.2 | −68.3% | 510.3 | −14.7% |
| **Drop-4** | **2** | **413.2 (−8.2%)** | **164.9** | **−69.8%** | **496.7** | **−17.0%** |

Same NFE=2: 164.9 vs 173.2 ms (**−4.8%**), 496.7 vs 510.3 G (**−2.7%**). §4 had Drop-4 slower than the full model at NFE=2 (181.8 vs 173.8 ms). Language in this run is 47.9 → 39.8 ms (−16.9%) and 57.6 → 44.1 G (−23.4%). §4 language time was 49.9 → 48.8 ms.

---

## 6. Materialized pruning and deployment (2026-10-09, GPU1)

Dropped layers are real modules. Q, the attention output projection, and the MLP weights are freed. The forward matches the §5 K/V-only skip: action max abs is **0** on one calibration frame for all three operating points (π0.5 at NFE=1 and NFE=10, Molmo at NFE=1, SmolVLA at NFE=2). TF32 is off. `compile_model` stays false. JSON: `results/stage_nfe/deployment_{pi05,molmoact2,smolvla}.json`. §4 and §5 are unchanged.

**MolmoAct2 LIBERO closed-loop (materialized Drop-16, GPU3, 2026-10-15).** After materialize, Taylor Drop-16 `[1–4, 24–35]` at **NFE=10** matches Full on **98.50%** Avg. (**394/400** each; Spatial 99.0 / Object 100.0 / Goal 99.0 / Long 96.0 vs Full 97.0 / 99.0 / 100.0 / 98.0). Full same protocol **98.50%**. Preflight `action_max_abs_vs_kvonly=0`. Protocol, paths, and caveats vs keep-KV 2000-ep anchors: `eval_nfe_libero.md` §7.3. This does not replace `eval_molmoact2_libero.md` keep-KV rows.

`torch.compile(mode="reduce-overhead")` is a latency column, not an action-equivalent protocol. It moves actions (π0.5 pruned 0.040, SmolVLA 0.022–0.026, Molmo 0.0078–0.016). Molmo's own expert CUDA graph does not: action max abs is 0. π0.5 / SmolVLA materialized closed-loop on LIBERO is still not in this section.

Allocated memory is after load, with the caching allocator emptied. Peak is one inference. Freed params assume bf16 (2 bytes).

| Model | Eager full → pruned | Allocated | Peak inference | Params freed |
|---|---|---:|---:|---:|
| π0.5 NFE=1 | 204.3 → 174.9 ms (−29 ms) | 8923 → 7685 MiB | 9081 → 7829 MiB | 654M, 1248 MiB |
| π0.5 NFE=10 | 629.8 → 631.3 ms | same | same | same |
| Molmo NFE=1 | 309.6 → 290.2 ms (−19 ms) | 11552 → 8635 MiB | 12590 → 9637 MiB | 1531M, 2920 MiB |
| Molmo NFE=10 | 1392.8 → 1401.2 ms | same | same | same |
| SmolVLA NFE=2 | 169.8 → 171.0 ms | 885 → 824 MiB | 920 → 851 MiB | 37M, 70 MiB |
| SmolVLA NFE=10 | 548.1 → 542.1 ms | same | same | same |

Molmo expert CUDA graph, same weights, action max abs 0. The graph wraps the expert flow loop; the VLM stays eager.

| | NFE=1 | NFE=10 | Peak allocated |
|---|---:|---:|---:|
| Full | 183.6 ms | 273.9 ms | 13061 MiB |
| Drop-16 | 171.5 ms | 254.0 ms | 10118 MiB |

The pruning delta stays 12 ms at NFE=1 and 20 ms at NFE=10. The graph removes most of the expert's eager overhead (NFE=10 falls from 1393 ms to 274 ms) and does not remove the prefix cost.

`reduce-overhead` NFE=1 latency, with the action drift above: π0.5 123.8 → 104.1 ms; Molmo 215.3 → 118.2 ms; SmolVLA NFE=2 22.3 → 22.1 ms. Molmo's compiler skipped CUDA graphs (`mutated inputs`) and hit the recompile limit, so the 97 ms Molmo gap is not a clean pruning measurement. SmolVLA's 22 ms is the compiled chunk; Drop-4 does not move it. The π0.5 full-model compile action check in that JSON was taken against Drop-6 actions and is not an equivalence result. The pruned π0.5 check (0.040) is against the materialized eager model.