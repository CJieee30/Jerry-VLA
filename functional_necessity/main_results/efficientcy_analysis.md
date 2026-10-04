# Efficiency: `predict_action_chunk` (not LIBERO wall time)

Protocol matches the π0.5 measurements: RTX 3090, `compile_model=false`, CUDA events warmup=5 / repeats=20, one calibration observation, PyTorch `FlopCounterMode`. Stored weights are not deleted. **Active params** subtract only the compute the skip operator actually skips.

Skip operators differ by stack and must not be mixed:

- π0.5: DTR Appendix B (keep prefix K/V; skip language Q / attention / O / MLP).
- MolmoAct2 / SmolVLA: keep-KV identity residual (the dropped VLM layer still writes K/V; Molmo skips MLP + ff_norm; SmolVLA skips VLM `o_proj` + MLP + post-attn LN). The action expert is never dropped.

Table 1 time / FLOP shares are over the three isolated modules (Vision + Language + Action), not over `predict_action_chunk` wall time. Do not read closed-loop `eval_s` as efficiency; MuJoCo dominates LIBERO wall time.

Source:

- π0.5: `results/eval_drop/{efficiency_pi05_baseline_modules,efficiency_fisher_k6,efficiency_pi05_drop6_language_only}.json`
- MolmoAct2 Taylor Drop-16 `[1–4,24–35]`: `results/molmoact2_libero/efficiency_taylor_drop16.json`
- SmolVLA Taylor Drop-4 `[12–15]`: `results/smolvla_libero/efficiency_taylor_drop4.json`

Closed-loop Avg. is from the corresponding `eval_*.md` (this-machine anchors).

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

Same skip, prefix language forward only (vision embeddings cached; expert denoise not run).

### π0.5 language module


| π0.5 language module | Params active (M) | Inference (ms) | FLOPs (G)        |
| -------------------- | ----------------- | -------------- | ---------------- |
| Baseline 18/18       | 2508.53           | 78.10          | 3974.95          |
| Drop-6 12/18         | 1854.21           | 53.43          | 2662.15          |
| Δ                    | −654.32（−26.1%）   | −24.67（−31.6%） | −1312.80（−33.0%） |




### MolmoAct2 language module


| MolmoAct2 language module | Params active (M) | Inference (ms) | FLOPs (G)        |
| ------------------------- | ----------------- | -------------- | ---------------- |
| Baseline 36/36            | 4029.68           | 81.32          | 3820.07          |
| Drop-16 20/36             | 2834.26           | 68.30          | 2612.74          |
| Δ                         | −1195.42（−29.7%）  | −13.03（−16.0%） | −1207.33（−31.6%） |


Language FLOPs Δ equals the full-chunk FLOPs Δ (−1207.33 G): keep-KV skip only removes VLM MLP compute. Prefix time −16.0% does not show up in Table 2 because Action is 89.5% of module time.

### SmolVLA language module


| SmolVLA language module | Params active (M) | Inference (ms) | FLOPs (G)      |
| ----------------------- | ----------------- | -------------- | -------------- |
| Baseline 16/16          | 204.63            | 45.37          | 57.60          |
| Drop-4 12/16            | 171.45            | 41.64          | 45.86          |
| Δ                       | −33.18（−16.2%）    | −3.73（−8.2%）   | −11.74（−20.4%） |


Language is only 9.6% of SmolVLA chunk FLOPs (Table 1), so Taylor Drop-4 saves 20.4% of the VLM prefix and about 2% of the full chunk. That is expected for a 16-layer VLM sitting next to a larger vision encoder and a 10-step expert.