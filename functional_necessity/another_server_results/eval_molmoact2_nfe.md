# MolmoAct2 action expert: fewer flow-matching steps (NFE) on LIBERO

`allenai/MolmoAct2-LIBERO-LeRobot`, local overlay on HF config/tokenizer, `inference_action_mode=continuous`, `enable_inference_cuda_graph=false`, no VLM drop. Euler from t=0→1, `x ← x + v/N`, default `flow_matching_num_steps=10`. Override is `policy.num_inference_steps`. NFE=1 queries t=0, which training never samples exactly (`time_offset=0.001`). Prefix (VLM) runs once; the action expert loops once per NFE.

**Phase A (offline).** 32 calib frames (4 suites × 8), same generator seed per frame, NFE ∈ {10, 4, 2, 1}. r(N) = median ||a_N − a_10|| / ||a_10|| on the 7 real action dims. Pass A if r(10) vs self is ~0 and `predict_action_chunk` latency falls with NFE. Do not judge success by r(1) magnitude. Source: `results/nfe_source/molmoact2_libero/molmoact2_nfe_probe.json`.

Phase A **PASS**. r(10 vs self) median=0.000e+00 (sanity True); r(10 repeat) median=0.000e+00 (gate True); chunk latency monotone (gate True); expert-loop latency monotone (gate True).


| NFE | median r vs NFE=10 | mean r | max r  | chunk ms | prefix ms | expert-loop ms | per-expert-step ms |
| --- | ------------------ | ------ | ------ | -------- | --------- | -------------- | ------------------ |
| 10  | 0.0000             | 0.0000 | 0.0000 | 1078.4   | 346.7     | 703.2          | 68.24              |
| 4   | 0.0111             | 0.0114 | 0.0194 | 658.5    | 346.4     | 279.7          | 68.11              |
| 2   | 0.0173             | 0.0192 | 0.0364 | 523.0    | 347.1     | 140.1          | 67.74              |
| 1   | 0.0258             | 0.0280 | 0.0667 | 439.0    | 337.3     | 71.2           | 69.45              |


NFE=10 vs a second NFE=10 with the same seed: median r=0.000e+00, max=0.000e+00.

r(1) is recoverability of the 10-step action, not closed-loop necessity. A large r(1) does not fail Phase A.

**Phase B (closed-loop).** Four suites × 10 tasks × 10 episodes (400 episodes), `seed=1000`, `per_episode_seed=true`, `eval_seed=1000`, `n_action_steps=10`, `num_steps_wait=50`, camera map `image` / `wrist_image`. Full-model NFE=10 then NFE=1 first. Hold if NFE=1 is within ~2 points of same-protocol NFE=10. NFE=2/4 only if NFE=1 drops. No 2000-episode run. No structured noise. After NFE=1 held, Taylor Drop-16 × NFE=1 uses keep-KV identity skip on `[1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` and is scored against the 2000-ep Taylor Drop-16 NFE=10 rate 98.10.

Table: LIBERO success (%), no recovery, split protocol.


| LLM drop                          | NFE | Spatial | Object | Goal | Long | Avg.  | Δ vs same-protocol NFE=10 |
| --------------------------------- | --- | ------- | ------ | ---- | ---- | ----- | ------------------------- |
| None (full model), split protocol | 10  | 98.0    | 99.0   | 98.0 | 97.0 | 98.00 | —                         |
| None (full model), split protocol | 1   | 99.0    | 100.0  | 99.0 | 95.0 | 98.25 | +0.25                     |


NFE=1 vs same-protocol NFE=10: 98.25 vs 98.00 (Δ +0.25). Holds within the ~2-point bar used on π0.5-LIBERO.

## Taylor Drop-16 × NFE=1

Keep-KV identity skip, no-L0 Taylor prefix Drop-16 `[1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]`. Reference is the other-machine 2000-ep Taylor Drop-16 at default NFE=10: Spatial 98.2 / Object 99.4 / Goal 98.6 / Long 96.2 / **Avg. 98.10**. This cell is 400 episodes (split protocol), same as the full-model NFE rows above.


| Setting                                         | NFE | Spatial | Object | Goal | Long | Avg.  | Δ vs Taylor-16 98.10 |
| ----------------------------------------------- | --- | ------- | ------ | ---- | ---- | ----- | -------------------- |
| Taylor Drop-16, 2000-ep (NFE=10, other machine) | 10  | 98.2    | 99.4   | 98.6 | 96.2 | 98.10 | —                    |
| Taylor Drop-16, split protocol                  | 1   | 100.0   | 98.0   | 97.0 | 98.0 | 98.25 | +0.15                |


Taylor Drop-16 × NFE=1 = 98.25 vs 98.10 (Δ +0.15). Holds within ~2 points of the 2000-ep NFE=10 reference.
Per suite: Spatial 100.0, Object 98.0, Goal 97.0, Long 98.0.

## Efficiency (`predict_action_chunk`)

Isolated RTX 3090, CUDA graph off, batch=1, warmup 3 / repeats 8, same calib frame. Source: `results/nfe_source/molmoact2_libero/efficiency_threeway.json`. Closed-loop `eval_s` is not this table.


| Setting                | NFE | Drop  | FLOPs (G) | vs baseline | Time (ms) | vs baseline | Prefix (ms) | Expert (ms) |
| ---------------------- | --- | ----- | --------- | ----------- | --------- | ----------- | ----------- | ----------- |
| Baseline (full)        | 10  | 0/36  | 5378.5    | 1.00×       | 830.6     | 1.00×       | 136.4       | 673.8       |
| molmoact2_nfe          | 1   | 0/36  | 5304.4    | 1.01×       | 222.1     | 3.74×       | 135.5       | 66.7        |
| Taylor Drop-16 × NFE=1 | 1   | 16/36 | 4058.9    | 1.33×       | 199.9     | 4.16×       | 111.6       | 67.1        |


NFE=10→1 is almost all latency (expert 674→67 ms) with almost no FLOP cut, because the VLM prefix is ~98% of FLOPs. Drop-16 then skips MLP on the keep-KV layers and is the FLOP lever (5304→4059 G); expert time is unchanged.