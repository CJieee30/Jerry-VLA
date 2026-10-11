# CARE: Calibration with Action-discontinuity REweighting

Copy the two tables below into the paper. The Chinese method and adoption rules follow those tables. Lab notes start after that section. IGIA is omitted. Do not re-evaluate Jaccard=1.0 sets.

CARE reweights each calibration frame by action-discontinuity (mean adjacent L2, H=10, mean-normalized to 1). It is not a new primary selector and is not DySL-VLA. Uniform calibration already finds the recommended drop sets; CARE leaves those sets unchanged and, when a k-cut sits on a near-tie, recovers a stronger already-evaluated set.

Local full-model anchors: π0.5 **97.25**, MolmoAct2 **97.50**, SmolVLA **83.00**. Skip is keep-KV. Seed=1000. 0.25–0.75 Avg. points are noise.

---

## Paper Table 1. LIBERO Avg. success rate (%) — Fisher / Taylor / GateProbe, uniform vs CARE

k is the Stage-1 operating point on each stack (π0.5 Drop-6, MolmoAct2 Drop-16 no L0). **Bold** = CARE changed the drop set and SR moved. † same drop set as uniform, SR not re-run. ‡ CARE drop set is identical to Taylor Drop-16, SR taken from that 2000-ep run (1962/2000). π0.5 GateProbe uniform is the **paired reval** (94.25), not the historical 95.00.

| Model | k | Method | Uniform drop | CARE drop | Jacc. | Uniform | CARE | Δ |
|---|---|---|---|---|---:|---:|---:|---:|
| π0.5 | 6/18 | Fisher | `[1, 13–17]` | `[1, 13–17]` | 1.00 | 97.75 | 97.75 † | 0.00 |
|  |  | Taylor (no L0) | `[1, 2, 14–17]` | `[1, 2, 14–17]` | 1.00 | 97.25 | 97.25 † | 0.00 |
|  |  | GateProbe | `[2, 4, 14–17]` | `[2, 13–17]` | 0.71 | 94.25 | **97.00** | **+2.75** |
| MolmoAct2 | 16/36 | Taylor (no L0) | `[1–4, 24–35]` | `[1–4, 24–35]` | 1.00 | 98.10 | 98.10 † | 0.00 |
|  |  | Fisher (no L0) | `[1–4, 5, 25–35]` | `[1–4, 24–35]` | 0.88 | 72.75 | **98.10** ‡ | **+25.35** |
|  |  | GateProbe (no L0) | `[2–4, 6, 24–35]` | `[1–4, 24–35]` | 0.88 | 87.00 | **98.10** ‡ | **+11.10** |

Reading: Taylor never moves, and neither does π0.5 Fisher, so CARE does not disturb the recommended selectors. The three CARE-changed rows all move **toward** those selectors (Molmo Fisher / GateProbe become Taylor 98.10; π0.5 GateProbe reaches the ceiling, −0.25 vs 97.25).

---

## Paper Table 2. Suite breakdown for every CARE-changed set

Uniform π0.5 rows are contemporaneous revals of the Stage-1 drop set. Molmo CARE rows reuse Taylor Drop-16 (same layers). Long = LIBERO-10.

| Model | Method | Setting | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---:|---:|---:|---:|---:|
| π0.5 Drop-6 | GateProbe | Uniform `[2, 4, 14–17]` | 97.0 | 93.0 | 94.0 | 93.0 | 94.25 |
|  |  | CARE `[2, 13–17]` | 97.0 | **99.0** | 94.0 | **98.0** | **97.00** |
| MolmoAct2 Drop-16 | Fisher | Uniform `[1–4, 5, 25–35]` | 91.2 | 79.8 | 51.0 | 69.0 | 72.75 |
|  |  | CARE = Taylor `[1–4, 24–35]` | **98.2** | **99.4** | **98.6** | **96.2** | **98.10** |
|  | GateProbe | Uniform `[2–4, 6, 24–35]` | 91.6 | 96.6 | 77.2 | 82.6 | 87.00 |
|  |  | CARE = Taylor `[1–4, 24–35]` | **98.2** | **99.4** | **98.6** | **96.2** | **98.10** |

Molmo Goal 51.0 → 98.6 is the Fisher cliff (L5 dropped vs L24 dropped). π0.5 Object 93 → 99 and Long 93 → 98 are the GateProbe L4 → L13 swap.

Suggested one-sentence caption: *CARE agrees with uniform calibration on Taylor and on π0.5 Fisher (Jaccard=1.0), and lifts near-tie cuts of Fisher / GateProbe onto those same high-SR sets.*

---

## 方法与采纳规则

CARE（动作不连续加权校准）不另训模型，也不把轨迹切成片段。校准集里的每一帧各自得到一个权重：看损失已经用过的归一化动作，在随后 H=10 步（评测的动作步长，不是按闭环调出来的 k）里，相邻两步差得越大，这一帧越不平滑，权重越高。占比很大的平滑移动因此不会主导层冗余排序。权重只用示范动作，不用闭环成功率去拟合，也不用它来选 H 或选 7 维还是臂 6 维。

设第 i 帧窗口内的归一化动作为 a_{i,1}, \ldots, a_{i,H}。先算相邻步的平均 L2，再除以全体均值，使均匀场精确为 1：

$$
d_i = \frac{1}{H-1}\sum_{t=1}^{H-1}\lVert a_{i,t+1}-a_{i,t}\rVert_2,
\qquad
\alpha_i = \frac{d_i}{\frac{1}{N}\sum_{j=1}^{N} d_j}.
$$

d_i 大表示这段动作变化急；d_i 小表示手臂在做长时间的平滑移动。\alpha_i 的均值是 1，所以全体相等时 CARE 与均匀加权是同一套分数。打分时 \alpha_i 只乘在该帧的分数贡献上，放在 flow-matching 损失外面，存好的噪声和时间步不动。以 Fisher 为例，层分数由均匀的 \frac{1}{N}\sum_i (g_i^2\theta^2) 改成 \frac{1}{N}\sum_i \alpha_i (g_i^2\theta^2)；Taylor、GateProbe 同样是在损失外乘 \alpha_i，再保持与 Stage-1 相同的归约。\pi0.5 的 Fisher / GateProbe 仍先在原始 batch 内做加权平均，再平方或取绝对值，不把全部帧拉平后平方。

三套栈的张量不一样。π0.5 与 SmolVLA 的权重和损失都是已存 50 步块的前 10 个有效步。MolmoAct2 的损失仍是原来那一步；权重是同一条轨迹上随后 10 帧，归一化与损失相同。7 维与臂 6 维（去掉夹爪）各算一份。Spearman ≥ 0.9 时只打 7 维（π0.5 0.94，SmolVLA 0.98）。低于 0.9 时两种都打（MolmoAct2 0.70）。

采纳 CARE 的新删除集，必须同时满足下面三条，否则保留均匀校准的原集，不开闭环：

1. 若该栈必须打两种维度，7 维与臂 6 维的 Drop-k 相同。只打 7 维的栈没有这一条。
2. 切口不是数值平局：被删的最后一层与被留的下一层，分数差不低于这条冗余序上相邻分数差的中位数的 10%。
3. 与均匀校准的 Jaccard 小于 1。Jaccard = 1 表示集合没变，不重评。

第 2 条不看成功率。MolmoAct2 GateProbe Drop-17 就是它要拦住的情况：两种维度的层分数 Spearman 仍是 0.999，但 7 维上 L6 与 L22 只差 6.7×10⁻⁶，约为中位相邻差的 1.3%，而且 7 维与臂 6 维选层不同。这种切口保留均匀集。Drop-16 上 Fisher 与 GateProbe 的两种维度选层相同，不受这条影响。

不要因为某一侧闭环更好，就把默认改成臂 6 维，或回头改 α 和 H。那是用成功率回搜权重。

---

# Lab notes (not for the paper)

Stage-2 of the flow-matching VLA pruning study. Chinese name: **动作不连续加权校准（CARE）**. Date: 2026-10-05. Do not rerun completed scoring or closed-loop. Do not overwrite Stage-1 rows in the per-stack `eval_*.md` tables.

CARE asks whether the Stage-1 drop sets are an artifact of treating every calibration timestamp as equal. It is **not** DySL-VLA (that method trains an adapter + controller + KD). It is **not** a new primary selector. Weights are never fit on closed-loop success rate.

This file is the authority for CARE numbers. Stack-specific Stage-1 tables stay in `eval_pi05_libero_v044.md`, `eval_molmoact2_libero.md`, `eval_smolvla_libero.md`.

---

## 1. Method

On the **loss-normalized** demonstration action, take the mean adjacent L2 over a window of H=10 (the eval action horizon, not a closed-loop-tuned k). Divide by the global mean so a uniform field is exactly α=1. High α = less continuous = more weight in scoring. α multiplies that sample's score contribution **outside** the flow-matching loss; the stored (noise, time) pair is unchanged.

The three stacks do **not** share one tensor:

| Stack | Weight tensor | Loss tensor |
|---|---|---|
| π0.5 | First 10 valid steps of the stored 50-step chunk | The same 50-step chunk |
| SmolVLA | First 10 valid steps of the stored 50-step chunk | The same 50-step chunk |
| MolmoAct2 | Next 10 frames of the same episode, same masked-normalizer + clamp as the loss | The original one-step action `(1,1,32)` |

Molmo's first subsequent step matches the stored loss action (max abs diff = 0). π0.5 filled 5/512 frames whose window had fewer than 2 steps with the median. Spearman(7-d, arm-6-d) ≥ 0.9 → GPU scoring uses 7-d only; Molmo is 0.7005 so both 7-d and arm-6-d were scored.

| Stack | n | ESS (7-d) | Spearman 7-d vs arm-6-d | GPU weights |
|---|---:|---:|---:|---|
| π0.5 | 512 | 386.80 | 0.9375 | 7-d |
| SmolVLA | 32 | 22.70 | 0.9824 | 7-d |
| MolmoAct2 | 32 | 23.30 | 0.7005 | 7-d and 6-d |

Scoring aggregation (must match Stage-1 when α=1):

- **π0.5 Fisher / GateProbe:** still reduce at the original 64×8 batches (4× mb=2 forwards). Fisher: weighted-mean language gradient, then square; 64 batches equal-weighted. Flattening 512 frames before squaring is forbidden. GateProbe: weighted-mean signed inner, then abs. α=1 on batch 0 matches Stage-1 exactly (`rel_abs_diff=0` for both).
- **Molmo / SmolVLA Fisher, Taylor, and GateProbe:** α outside the loss; divide by N=32. Molmo Fisher is `(1/N) Σ_i α_i (g_i² θ²)`. GateProbe is `(1/N) Σ_i α_i |I_i|`.

High score = keep. Molmo / π0.5 no-L0: drop the k lowest remaining layers after removing 0 from `redundancy_order`. 0.25–0.75 Avg. points are noise (1 episode = 0.25 on 400-ep stacks). Local anchors are the referee: π0.5 **97.25**, MolmoAct2 **97.50**, SmolVLA **83.00**.

Sources: `results/stage2_action_weight/` (weights, scores, `ranking.md`). Closed-loop: `results/eval_drop/k6_gateprobe_{reval,action_discontinuity}/`, `results/smolvla_libero/drop4_fisher_{reval,action_discontinuity}/`.

---

## 2. What CARE is allowed to claim

CARE is a **robustness check** of the Stage-1 recipe, plus a **tie-break** when a k-cut sits on adjacent near-tied layers.

Write:

- Uniform calibration already yields the recommended drop sets. CARE **agrees** at those k (Jaccard 1.0), so the recipe is not an artifact of equal frame weights.
- At k-cuts that sit on a near-tie, the same weights can swap one adjacent layer. The three swaps we measured moved toward a better or already-evaluated high-SR set.
- On the three operating points, CARE does not change the drop set and therefore does not harm those selectors.

Do not write:

- CARE is “especially necessary” to find Fisher-6 / Taylor-16 / Taylor-4. Stage-1 already found them.
- CARE universally “improves the accuracy of redundancy metrics.” Operating-point accuracy is **reproduced**, not raised. Accuracy evidence is three near-tie cuts, one definition (H=10), no SR-tuned weights.
- Those three operating-point sets are globally optimal. They are the **recommended sets among methods already scored at that k**. CARE did not re-score all nine metrics under weights.
- CARE never hurts any reliable method at any k. Later ranks of π0.5 Fisher and Molmo Taylor already swap; a different k was not closed-loop evaluated. A near-tie can in principle flip the wrong way; we did not see that in these three cuts.
- CARE is DySL, or training-free skip control.

---

## 3. Operating points (drop set unchanged)

Pinned to Stage-1: π0.5 Fisher Drop-6 **97.75**; Molmo Taylor Drop-16 (no L0) **98.10**; SmolVLA Taylor Drop-4 **78.50**. π0.5 / Molmo sit at the ceiling; a changed set would only be asked whether it still holds, not whether SR goes higher. These sets did not change, so there is no new closed-loop.

| Stack | Stage-1 drop | CARE drop | Jaccard | Closed-loop |
|---|---|---|---:|---|
| π0.5 Fisher k=6 no-L0 | `[1, 13–17]` | `[1, 13–17]` | 1.0 | stop; do not re-eval 97.75 |
| Molmo Taylor k=16 no-L0 (7-d) | `[1–4, 24–35]` | `[1–4, 24–35]` | 1.0 | stop; do not re-eval 98.10 |
| Molmo Taylor k=16 no-L0 (arm 6-d) | same | same | 1.0 | same |
| SmolVLA Taylor k=4 | `[12–15]` | `[12–15]` | 1.0 | stop; set unchanged |
| SmolVLA GateProbe k=4 | `[7, 13–15]` | `[7, 13–15]` | 1.0 | stop; do not re-eval 72.00 |

SmolVLA Taylor's full `redundancy_order` is identical to Stage-1. SmolVLA GateProbe Drop-4 is the same set (72.00); later ranks swap L12 ↔ L9 at Drop-8, which was not closed-loop. π0.5 Fisher operating-point set is unchanged; ranks after the sixth slot swap (Stage-1 `[17, 16, 15, 13, 14, 1, 2, 3, 10, 12, 4, …]` vs CARE `[17, 16, 15, 13, 14, 1, 2, 3, 4, 10, 12, …]`). That is not a new Drop-6. Source: `results/stage2_action_weight/smolvla_gateprobe_action_discontinuity.{pt,json}`.

---

## 4. Near-tie cuts (one adjacent swap)

The three Stage-1 **suboptimal** cuts below all sat on adjacent layers. CARE swapped that pair. Weights were not fit on these SR numbers. SmolVLA GateProbe Drop-4 was also re-scored and did **not** swap (Jaccard 1.0); L15 remains 0 under keep-KV.

| Stack | Stage-1 drop | CARE drop | Swap | Jaccard |
|---|---|---|---|---:|
| Molmo Fisher k=16 no-L0 | `[1–4, 5, 25–35]` | `[1–4, 24–35]` | L5 ↔ L24 | 0.882 |
| π0.5 GateProbe k=6 no-L0 | `[2, 4, 14–17]` | `[2, 13–17]` | L4 ↔ L13 | 0.714 |
| SmolVLA Fisher k=4 | `[10, 11, 12, 14]` | `[10, 12, 13, 14]` | L11 ↔ L13 | 0.600 |

Molmo 7-d and arm-6-d CARE Fisher `redundancy_order` are identical. Stage-1 Fisher scores: L5 = 0.00319, L24 = 0.00327 (L24 slightly more important, so Drop-16 **keeps L24 and drops L5**, SR **72.75**). CARE flips the pair: Drop-16 **drops L24 and keeps L5**, which is exactly Taylor Drop-16 (**98.10**). Do not rerun 2000 episodes.

π0.5 GateProbe L0 remains last, so no-L0 equals native. α=1 signed inner on batch 0 matches Stage-1 (`rel_abs_diff=0`).

---

## 5. Closed-loop (paired 400-ep, seed=1000)

Stage-1 sets were re-evaluated in **new directories** so the historical 95.00 / 69.25 rows stay. All four jobs: `n_episodes=400`, `EXIT:0`. CARE jobs on free GPU1/3; Stage-1 revals on GPU0/2 next to non-LIBERO processes (not killed). 0.25–0.75 is still noise.

### 5.1 π0.5 GateProbe Drop-6

Skip is Appendix B (keep prefix K/V). Referee: this-machine baseline **97.25**; Fisher Drop-6 **97.75**.

| Setting | Drop | Spatial | Object | Goal | Long | Avg. | Δ vs 97.25 |
|---|---|---:|---:|---:|---:|---:|---:|
| GateProbe Stage-1 (historical) | `[2, 4, 14–17]` | 97.0 | 93.0 | 94.0 | 96.0 | 95.00 | −2.25 |
| GateProbe Stage-1 reval (GPU0) | `[2, 4, 14–17]` | 97.0 | 93.0 | 94.0 | 93.0 | 94.25 | −3.00 |
| CARE GateProbe (GPU1) | `[2, 13–17]` | 97.0 | 99.0 | 94.0 | 98.0 | **97.00** | **−0.25** |

Reval vs historical: **−0.75** (noise). CARE vs contemporaneous reval: **+2.75** (11 episodes). CARE vs Fisher 97.75: **−0.75** (noise band). The L4→L13 swap holds the ceiling. Gain is Object 93→99 and Long 93/96→98. Do not overwrite the Stage-1 GateProbe 95.00 row. Do not replace Fisher-6 as the π0.5 operating point.

Sources: `results/eval_drop/{k6_gateprobe,k6_gateprobe_reval,k6_gateprobe_action_discontinuity}/eval_info.json`.

### 5.1b CARE v2 high-α quartile (π0.5 GateProbe Drop-6)

Coverage ablation, not α-weighted CARE. Same 480-frame pool as E1; scores are the equal mean of the 120 frames in the top quartile of whitened-7d α. Drop `[2, 3, 14–17]` (L13 ↔ L3 vs v2 uniform `[2, 13–17]`). GPU1, 400 ep, seed=1000, `compile_model=false`, keep-KV, `EXIT:0` 2026-10-10T09:33:02Z.

| Setting | Drop | Spatial | Object | Goal | Long | Avg. | Δ vs 97.25 |
|---|---|---:|---:|---:|---:|---:|---:|
| GateProbe Stage-1 (historical) | `[2, 4, 14–17]` | 97.0 | 93.0 | 94.0 | 96.0 | 95.00 | −2.25 |
| GateProbe Stage-1 reval | `[2, 4, 14–17]` | 97.0 | 93.0 | 94.0 | 93.0 | 94.25 | −3.00 |
| v2 uniform / CARE-weighted | `[2, 13–17]` | 97.0 | 99.0 | 94.0 | 98.0 | 97.00 | −0.25 |
| v2 high-α quartile (GPU1) | `[2, 3, 14–17]` | 97.0 | 99.0 | 93.0 | **100.0** | **97.25** | **0.00** |

High-α vs v2 uniform: **+0.25** (1 episode). Ties the 97.25 anchor (389/400). Still **−0.50** vs Fisher-6 97.75. Jaccard vs `[2, 13–17]` = 0.71. Do not overwrite Stage-1 95.00 or CARE 97.00. Do not promote this set over Fisher-6.

Source: `results/eval_drop/k6_gateprobe_v2_hialpha/{eval_info,selection,drop_meta}.json`.

### 5.2 SmolVLA Fisher Drop-4

Skip is keep-KV. Referee: this-machine baseline **83.00**; Taylor Drop-4 **78.50**.

| Setting | Drop | Spatial | Object | Goal | Long | Avg. | Δ vs 83.00 |
|---|---|---:|---:|---:|---:|---:|---:|
| Fisher Stage-1 (historical) | `[10, 11, 12, 14]` | 69.0 | 86.0 | 72.0 | 50.0 | 69.25 | −13.75 |
| Fisher Stage-1 reval (GPU2) | `[10, 11, 12, 14]` | 67.0 | 85.0 | 72.0 | 55.0 | 69.75 | −13.25 |
| CARE Fisher (GPU3) | `[10, 12, 13, 14]` | 73.0 | 80.0 | 70.0 | 62.0 | **71.25** | **−11.75** |

Reval vs historical: **+0.50** (noise). CARE vs contemporaneous reval: **+1.50** (6 episodes, just above noise). Suites mix (Long/Spatial up, Object down). CARE Fisher is still **−7.25** vs Taylor 78.50. The L11↔L13 swap points the right way; Fisher stays a suboptimal selector on this stack. Do not overwrite the Stage-1 Fisher 69.25 row.

Sources: `results/smolvla_libero/{drop4_fisher,drop4_fisher_reval,drop4_fisher_action_discontinuity}/eval_info.json`.

### 5.3 MolmoAct2 Fisher Drop-16 (no L0)

No new 2000-ep run. CARE Fisher drop = Taylor Drop-16 `[1–4, 24–35]`, already **98.10** (1962/2000). Stage-1 Fisher Drop-16 `[1–4, 5, 25–35]` remains the historical **72.75** row. Same-set rule: do not rerun.

---

## 6. Reading

Uniform calibration is enough to **choose** the three operating points. CARE shows those choices are **stable** under action-discontinuity weights, and that when a scorer's k-cut is a near-tie, the same weights can break the tie toward the better set. That is a robustness check and a tie-break, not a new main method and not evidence that uniform calibration failed on Fisher-6 / Taylor-16 / Taylor-4.

A different importance definition or window needs new scoring. Do not tune these α on closed-loop SR.

---

## 7. Cliff cuts (π0.5 Drop-8, Molmo Drop-17)

Same CARE weights. Stage-1 SR at these k: π0.5 Fisher / Taylor no-L0 Drop-7 **96.00**, GateProbe **93.25**; Drop-8 Fisher / Taylor **85.50**, GateProbe **85.75**; Molmo Taylor Drop-17 **89.60**, GateProbe **82.65**, Fisher **70.60**. Jaccard 1.0 → stop. π0.5 GateProbe CARE Drop-7 / Drop-8 finished (2026-10-06). Molmo GateProbe CARE Drop-17 finished (2026-10-07): 7-d **19.80**, arm 6-d **97.65**. Do not restart any of the four.

π0.5 Taylor is new scoring (Stage-1 Taylor is 256× mb=2, not orig-batch). α=1 on batch 0 matches Stage-1 Taylor (`rel_abs_diff=0`, n_microbatches=256). Molmo GateProbe is new scoring: α outside the loss, `(1/N) Σ_i α_i |I_i|`, N=32; signed inners computed once, 7-d and arm-6-d applied after.

| Stack | Stage-1 drop | CARE drop | Swap | Jaccard | Closed-loop |
|---|---|---|---|---:|---|
| π0.5 Fisher k=8 no-L0 | `[1–3, 13–17]` | `[1–3, 13–17]` | — | 1.0 | stop; do not re-eval 85.50 |
| π0.5 Taylor k=8 no-L0 | `[1–3, 13–17]` | `[1–3, 13–17]` | — | 1.0 | stop; same set as Fisher |
| π0.5 GateProbe k=7 no-L0 | `[2, 4, 13–17]` | `[2, 9, 13–17]` | L4 ↔ L9 | 0.750 | **94.00** vs hist. 93.25 (+0.75, noise) |
| π0.5 GateProbe k=8 no-L0 | `[2, 4, 10, 13–17]` | `[2, 4, 9, 13–17]` | L10 ↔ L9 | 0.778 | **88.00** vs hist. 85.75 (**+2.25**) |
| Molmo Fisher k=17 no-L0 (7-d / 6-d) | `[1–5, 24–35]` | `[1–5, 24–35]` | — | 1.0 | stop; do not re-eval 70.60 |
| Molmo Taylor k=17 no-L0 (7-d / 6-d) | `[1–4, 23–35]` | `[1–4, 23–35]` | — | 1.0 | stop; do not re-eval 89.60 |
| Molmo GateProbe k=17 no-L0 (7-d) | `[2–4, 6, 22, 24–35]` | `[1–4, 6, 24–35]` | L22 ↔ L1 | 0.889 | **19.80** vs hist. 82.65 (**−62.85**) |
| Molmo GateProbe k=17 no-L0 (arm 6-d) | `[2–4, 6, 22, 24–35]` | `[1–4, 22, 24–35]` | L6 ↔ L1 | 0.889 | **97.65** vs hist. 82.65 (**+15.00**) |

Fisher Drop-16 on Molmo swapped L5 ↔ L24. At Drop-17 both layers are already in the drop set, so CARE Fisher Drop-17 equals Stage-1. Taylor Drop-16 / Drop-17 and π0.5 Fisher / Taylor Drop-8 are similarly stable. The two GateProbe cuts sit on near-ties (π0.5 L9 = 0.00387 vs L10 = 0.00413; Molmo 7-d L1 = 0.01191 vs L22 = 0.01342, L6 = 0.01341). CARE 7-d GateProbe Drop-17 is not Taylor `[1–4, 23–35]` and not Fisher `[1–5, 24–35]`.

CARE Taylor Drop-6 no-L0 is still `[1, 2, 14–17]` (Jaccard 1.0 vs Stage-1 Taylor 97.25). Later ranks swap L12 ↔ L7; that is not a new Drop-8.

Sources: `results/stage2_action_weight/pi05_{fisher,taylor,gateprobe}_action_discontinuity.{pt,json}`, `molmoact2_{fisher,taylor,gateprobe}_action_discontinuity{,_arm6}.{pt,json}`. π0.5 Taylor sanity: `pi05_taylor_sanity.json`.

### 7.1 π0.5 GateProbe Drop-7 / Drop-8 (400 ep, seed=1000, EXIT:0)

No contemporaneous Stage-1 reval this round; uniform numbers are the historical `k7_gateprobe_nol0` / `k8_gateprobe_nol0` rows. Referee: this-machine baseline **97.25**; Fisher Drop-7 **96.00**; Fisher Drop-8 **85.50**.

| Setting | Drop | Spatial | Object | Goal | Long | Avg. | Δ vs hist. |
|---|---|---:|---:|---:|---:|---:|---:|
| GateProbe Drop-7 Stage-1 | `[2, 4, 13–17]` | 97.0 | 94.0 | 87.0 | 95.0 | 93.25 | — |
| CARE Drop-7 (GPU0) | `[2, 9, 13–17]` | 97.0 | 97.0 | 88.0 | 94.0 | **94.00** | +0.75 |
| GateProbe Drop-8 Stage-1 | `[2, 4, 10, 13–17]` | 94.0 | 87.0 | 77.0 | 85.0 | 85.75 | — |
| CARE Drop-8 (GPU2) | `[2, 4, 9, 13–17]` | **97.0** | 88.0 | 79.0 | **88.0** | **88.00** | **+2.25** |

Drop-7 +0.75 is 3 episodes (Object 94→97, Goal 87→88, Long 95→94). That is the noise band. CARE Drop-7 is still **−2.00** vs Fisher 96.00.

Drop-8 +2.25 is 9 episodes and every suite is up. CARE 88.00 is also **+2.50** vs Fisher Drop-8 85.50. The L10→L9 swap is the cliff evidence on this stack. Do not overwrite the Stage-1 GateProbe 93.25 / 85.75 rows. Do not promote GateProbe over Fisher-6 as the operating point.

Sources: `results/eval_drop/{k7_gateprobe_nol0,k7_gateprobe_action_discontinuity,k8_gateprobe_nol0,k8_gateprobe_action_discontinuity}/eval_info.json`.

### 7.2 MolmoAct2 GateProbe Drop-17 (2000 ep, seed=1000, EXIT:0)

Both runs finished with `n_episodes=2000` (500 per suite) and `EXIT:0`. 7-d on GPU1, 33h 54m (396/2000). Arm 6-d on GPU3, 18h 14m (1953/2000). No contemporaneous Stage-1 reval; the uniform number is historical `drop17_gateprobe_nol0` **82.65**. Referee: this-machine baseline **97.50**; Taylor Drop-17 **89.60**.

| Setting | Drop | Spatial | Object | Goal | Long | Avg. | Δ vs hist. |
|---|---|---:|---:|---:|---:|---:|---:|
| GateProbe Drop-17 Stage-1 | `[2–4, 6, 22, 24–35]` | 91.8 | 92.4 | 72.8 | 73.6 | 82.65 | — |
| CARE 7-d (GPU1) | `[1–4, 6, 24–35]` | 45.0 | 16.4 | 15.6 | 2.2 | **19.80** | **−62.85** |
| CARE arm 6-d (GPU3) | `[1–4, 22, 24–35]` | **98.2** | **99.0** | **96.8** | **96.6** | **97.65** | **+15.00** |

7-d keeps L22 and drops L6. Every suite falls; Long is 2.2. Arm 6-d keeps L6 and drops L22, and sits **+0.15** vs the 97.50 anchor (noise) and **+8.05** vs Taylor Drop-17 89.60. The two dimensions do not agree, and the 7-d cut was already a numerical tie (L6 vs L22, about 1.3% of the median adjacent gap). The adoption rule therefore still refuses this cut. Do not overwrite the Stage-1 GateProbe 82.65 row, and do not promote either CARE set over Taylor Drop-16.

Sources: `results/molmoact2_libero/drop17_gateprobe_action_discontinuity{,_arm6}/eval_info.json`.

---

## 8. IGIA (operating points, 2026-10-05)

Same CARE weights. No closed-loop: all three operating-point drop sets are Jaccard **1.0**. π0.5 uses orig-batch weighted-mean gradient then square (no θ², no 1/N, 4× mb=2); α=1 on batch 0 matches Stage-1 (`rel_abs_diff=0`). Molmo / SmolVLA: α outside the loss, `Σ_i α_i ||g_i||²` (no 1/N), divide-not-used so α=1 matches Stage-1 IGIA. Molmo 7-d and arm-6-d scored separately.

| Stack | Stage-1 drop | CARE drop | Jaccard | Closed-loop |
|---|---|---|---:|---|
| π0.5 IGIA k=6 | `[2, 13–17]` | `[2, 13–17]` | 1.0 | stop; do not re-eval 97.00 |
| π0.5 IGIA k=7 no-L0 | `[2, 5, 13–17]` | `[2, 5, 13–17]` | 1.0 | stop; do not re-eval 93.00 |
| Molmo IGIA k=16 no-L0 (7-d / 6-d) | `[1–4, 5, 25–35]` | `[1–4, 5, 25–35]` | 1.0 | stop; do **not** map to Taylor 98.10 |
| Molmo IGIA k=17 no-L0 (7-d) | `[1–5, 24–35]` | `[1–5, 24–35]` | 1.0 | stop; do not re-eval 70.60 |
| SmolVLA IGIA k=4 | `[10, 12, 14, 15]` | `[10, 12, 14, 15]` | 1.0 | stop; do not re-eval 73.75 |

CARE Fisher Drop-16 on Molmo flipped L5 ↔ L24 because those Fisher scores were a near-tie (0.00319 vs 0.00327). CARE IGIA does **not** flip: L5 = 104.28 vs L24 = 131.32, so Drop-16 still drops L5 and keeps L24 (historical **72.75**). Do not write that CARE rescues IGIA at this cliff. Drop-17 then includes both layers, same as Stage-1.

π0.5 IGIA L0 remains last. Later Drop-8 would be `[2, 4, 5, 13–17]` (Stage-1 IGIA Drop-8 was never closed-loop).

Sources: `results/stage2_action_weight/{pi05,molmoact2,smolvla}_igia_action_discontinuity{,_arm6}.{pt,json}`. π0.5 sanity: `pi05_igia_sanity.json`.

## Checks

Weight dumps: `results/stage2_action_weight/weights_{pi05,smolvla,molmoact2}.pt`. π0.5 α=1 sanity: Fisher, GateProbe, Taylor, and IGIA `rel_abs_diff=0` on batch 0. Molmo / SmolVLA CARE Fisher, Taylor, and GateProbe divide by N=32; CARE IGIA does not (Stage-1 IGIA has no 1/N). Historical `k6_gateprobe`, `k7_gateprobe_nol0`, `k8_gateprobe_nol0`, and `drop4_fisher` directories were not overwritten. π0.5 CARE GateProbe Drop-6 / Drop-7 / Drop-8 wrote `eval_info.json` with `n_episodes=400` and `EXIT:0` (97.00 / 94.00 / 88.00). CARE v2 high-α GateProbe Drop-6 (`k6_gateprobe_v2_hialpha`) wrote `eval_info.json` with `n_episodes=400` and `EXIT:0` (97.25, 389/400), drop `[2, 3, 14, 15, 16, 17]`. Molmo CARE GateProbe Drop-17 wrote `eval_info.json` with `n_episodes=2000` and `EXIT:0` (7-d 19.80, arm 6-d 97.65). CARE IGIA operating-point sets are Jaccard 1.0; do not re-eval 97.00 / 72.75 / 73.75. SmolVLA CARE GateProbe Drop-4 is Jaccard 1.0; do not re-eval 72.00.
