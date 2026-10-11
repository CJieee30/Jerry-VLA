# CARE: Calibration with Action-discontinuity REweighting

**地位（2026-10-10 起）：负结果，不是方法贡献。** 不要把下面两张表当论文第二条方法。表内数字保留，作为已经测过的记录。失败原因、闸门、CARE v2 E1、D/S 时间戳、高 α 分位、IGIA 未翻盘，全部写在本文 lab notes（第 2、9–12 节）。IGIA 不进论文表。Jaccard=1.0 的集合不要重评。不要用闭环成功率调 α 或 H。不要再做随机权重、反向权重或留出模型。

CARE 用动作不连续（相邻 L2 均值，H=10，再除以全体均值使均匀场为 1）给校准帧加权。它不是新的主选层器，也不是 DySL-VLA（DySL 要 adapter + controller + KD）。均匀校准已经找出三个经营点；CARE 在那些 k 上不改层集。看起来「变好」的切口，是近邻平局翻到已经评过的更强集合。Molmo GateProbe Drop-17 的 7 维是 **19.80** 反例。同一 480 帧上加权相对均匀的 Jaccard 恒为 1.0；突变/平滑时间戳也不能证明「多采突变帧就测得更准」。

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

Reading (measurement only, not a method claim): Taylor never moves, and neither does π0.5 Fisher. The three CARE-changed rows all move toward already-evaluated selectors (Molmo Fisher / GateProbe become Taylor 98.10; π0.5 GateProbe 97.00 is −0.25 vs the 97.25 anchor and −0.75 vs Fisher-6). Table 1 omits Molmo GateProbe Drop-17 7-d = 19.80.

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

Do not use a method-style caption. If these two tables appear at all, they belong in an appendix of negative results: operating-point sets are unchanged; the three CARE-changed rows land on already-evaluated sets; Molmo GateProbe Drop-17 7-d (not in Table 1) is 19.80.

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

不要因为某一侧闭环更好，就把默认改成臂 6 维，或回头改 α 和 H。那是用成功率回搜权重。闸门拦住了 19.80 的切口，但这不能把 CARE 做成可自动采用的方法：经营点本来就 Jaccard = 1，过闸门的近邻翻转又都落到已有集合。规则本身是实验记录，不是还要执行的选层流程。

---

# Lab notes (not for the paper)

## 0. 结论（2026-10-10，先读这一节）

CARE **没有做成方法**。论文主贡献是无恢复深度冗余，不是校准加权。附录可以写：均匀校准已经给出经营点；按动作不连续加权既不改变那些层集，也不能稳定地提高次优切口；再把校准帧换成突变帧或平滑帧，方向也随模型和指标改变。

三条经营点（π0.5 Fisher-6 `[1,13–17]` **97.75**，Molmo Taylor-16 `[1–4,24–35]` **98.10**，SmolVLA Taylor-4 `[12–15]` **78.50**）在 CARE 下 Jaccard 均为 1.0。后面的 E1 / D/S / 高 α 都没有推翻这一点。

Stage-2 of the flow-matching VLA pruning study. Chinese name: **动作不连续加权校准（CARE）**. v1 打分与闭环：2026-10-05 至 2026-10-07。v2 E1 / 高 α / D/S：2026-10-09 至 2026-10-10。地位改定：2026-10-10。Do not rerun completed scoring or closed-loop. Do not overwrite Stage-1 rows in the per-stack `eval_*.md` tables.

CARE asks whether the Stage-1 drop sets are an artifact of treating every calibration timestamp as equal. It is **not** DySL-VLA (that method trains an adapter + controller + KD). It is **not** a new primary selector. Weights are never fit on closed-loop success rate.

This file is the authority for CARE numbers, including the later negative checks. Stack-specific Stage-1 tables stay in `eval_pi05_libero_v044.md`, `eval_molmoact2_libero.md`, `eval_smolvla_libero.md`. Raw v2 tables also live in `results/stage2_care_v2/e1.md` and `results/stage2_care_v2/composition/composition.md`; if they disagree with this file, this file wins after a json check.

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

## 2. 为何失败（可以写什么 / 禁止写什么）

CARE 曾经被写成「稳健性检查 + 近邻决胜」。2026-10-10 之后这条也不再当贡献。它是负结果：假设不成立，也撑不起附录以外的方法段落。完整因果链见第 11 节，实验清单见第 12 节。

### 2.1 可以写（附录）

- 均匀校准已经给出三个经营点。CARE 在这些 k 上 Jaccard = 1.0，说明经营点**不是**「每帧等权」这一条假设的伪影。
- 三个次优切口（Molmo Fisher-16、π0.5 GateProbe-6、SmolVLA Fisher-4）各对换一层近邻。新集要么等于已经评过的更强集合，要么仍远低于该栈的经营点。
- 平局闸门是必要的：Molmo GateProbe Drop-17 的 7 维闭环 **19.80**（相对历史 82.65 为 −62.85）。7 维与臂 6 维选层不一致，闸门拒绝采用。这是方法不安全的证据，不是「闸门救了 CARE」。
- CARE v2 把校准集固定成同一 480 帧之后，α 加权相对均匀加权的 Drop-k **全部** Jaccard = 1.0。v1 里相对 Stage-1 的 Jaccard < 1，主要来自校准集 / 估计器更换，不是 α。
- D/S 时间戳：Taylor 经营点不变。π0.5 Fisher-6 上平滑帧映射到已评的 97.75，突变帧映射到已评的 97.00，和「突变帧更准」相反。Molmo Fisher-16 上方向又反过来。没有可迁移的时间戳规则。
- 高 α 分位 GateProbe Drop-6 为 97.25，相对 CARE 加权 97.00 是 +0.25（1 集，噪声），仍低于 Fisher-6 的 97.75。

### 2.2 禁止写

- 「CARE 大幅提高稳定性」，或把 CARE 写成第二条方法贡献。
- 「CARE 对找出 Fisher-6 / Taylor-16 / Taylor-4 特别必要」。Stage-1 均匀校准已经找到它们。
- 「CARE 普遍提高冗余指标的准确性」。经营点准确性是**复现**，不是提高。三个近邻翻转不是普适精度证据。
- 「三个经营点全局最优」。它们只是该 k 上已经打过分的方法里的推荐集。CARE 没有对九法全部重新加权打分。
- 「CARE 从不伤害可靠方法」。Molmo GateProbe Drop-17 7 维就是伤害。π0.5 Fisher / Molmo Taylor 在更靠后的名次已经对换，那些 k 没有闭环。
- 「多采突变帧就能测得更准」。第 10 节 D/S 否定了这条。
- 「α 加权本身改变了 Drop 集」。第 9 节 E1 否定了这条（同一 480 帧上 J(CARE, uniform) = 1.0）。
- CARE 是 DySL，或 training-free 的跳层控制器。
- 用闭环 SR 回头选 7 维还是臂 6 维、选 H、选高 α 分位。那是用成功率回搜权重。

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

## 6. Reading（v1 闭环当时的读法；2026-10-10 之后以第 0、2、11 节为准）

v1 当时可以写：均匀校准已经够用来**选出**三个经营点；CARE 下这些选择保持不变；近邻平局上同一套权重会翻到更好或已评过的集合。那是稳健性检查，不是新主方法，也不能说均匀校准在 Fisher-6 / Taylor-16 / Taylor-4 上失败了。

2026-10-10 之后这条也不再当贡献。E1（第 9 节）表明固定校准帧后 α 不改 Drop-k。D/S（第 10 节）表明突变帧并不更准。Molmo GateProbe Drop-17 7 维 19.80 是反例。综合失败原因见第 11 节。不要用闭环 SR 调 α。

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

---

## 9. CARE v2 E1：同一 480 帧上均匀 vs 加权（2026-10-09，只打分）

v1 把 CARE 加权层集拿去和 Stage-1 比，混进了三件不同的事：校准集换了、估计器换了（π0.5 Fisher / GateProbe 的 Stage-1 是 orig-batch 归约）、以及 α。E1 把这三件拆开。

协议（冻结，不用闭环 SR 调）：40 任务 × 12 帧 = **480** 帧，`seed=42`，H=10，对原始 7 维示范动作做分维白化相邻 L2，再均值归一化得到 α。只打 7 维，没有臂 6 维分支。ESS = **384.1**。Molmo 读了 109 个 parquet。均匀臂：同一 480 帧、等权。CARE 臂：同一 480 帧、乘 α。Fisher / Taylor / GateProbe 都是逐帧打分再汇总。E1 **不开闭环**。

α 范围：min 0.069、max 3.227。d 分位（白化后）p0/25/50/75/100 = 0.142 / 1.288 / 1.933 / 2.651 / 6.606。

| Stack | Method | Stage-1 drop | v2 uniform | v2 CARE | J(uni,S1) | J(CARE,S1) | J(CARE,uni) | τ(uni,S1) | c_uni | c_CARE |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| pi05 | fisher | `[1, 13–17]` | `[2, 13–17]` | `[2, 13–17]` | 0.71 | 0.71 | **1.00** | 0.908 | 0.93 | 0.90 |
| pi05 | taylor | `[1, 2, 14–17]` | `[1, 2, 14–17]` | `[1, 2, 14–17]` | 1.00 | 1.00 | **1.00** | 0.922 | 1.00 | 1.00 |
| pi05 | gateprobe | `[2, 4, 14–17]` | `[2, 13–17]` | `[2, 13–17]` | 0.71 | 0.71 | **1.00** | 0.882 | 0.91 | 0.78 |
| smolvla | fisher | `[10, 11, 12, 14]` | `[10, 12, 13, 14]` | `[10, 12, 13, 14]` | 0.60 | 0.60 | **1.00** | 0.900 | 0.98 | 0.99 |
| smolvla | taylor | `[12–15]` | `[12–15]` | `[12–15]` | 1.00 | 1.00 | **1.00** | 0.950 | 1.00 | 1.00 |
| smolvla | gateprobe | `[7, 13–15]` | `[11, 13–15]` | `[11, 13–15]` | 0.60 | 0.60 | **1.00** | 0.767 | 0.99 | 0.98 |
| molmoact2 | fisher | `[1–4, 5, 25–35]` | `[1–4, 24–35]` | `[1–4, 24–35]` | 0.88 | 0.88 | **1.00** | 0.921 | 0.75 | 0.71 |
| molmoact2 | taylor | `[1–4, 24–35]` | `[1–4, 24–35]` | `[1–4, 24–35]` | 1.00 | 1.00 | **1.00** | 0.971 | 1.00 | 1.00 |
| molmoact2 | gateprobe | `[2–4, 6, 24–35]` | `[1–3, 21, 24–35]` | `[1–3, 21, 24–35]` | 0.78 | 0.78 | **1.00** | 0.829 | 0.55 | 0.41 |

**E1 的关键事实：九行的 J(CARE, uniform) 全部是 1.00。** 在校准帧固定之后，α 加权不改 Drop-k。v1 里相对 Stage-1 的 Jaccard < 1，是换了 480 帧共享集、以及 π0.5 上改成逐帧估计，不是 CARE 权重在起作用。

经营点三条在 v2 均匀下：

- π0.5 Taylor-6、Molmo Taylor-16、SmolVLA Taylor-4：与 Stage-1 相同（Jaccard 1.0）。Taylor 对校准集更换也不动。
- π0.5 Fisher-6：v2 均匀变成 `[2, 13–17]`，这是 v1 CARE GateProbe-6 的 97.00 集，不是 Stage-1 Fisher-6 的 97.75。E1 规定：经营点若被 v2 均匀改掉，必须先闭环才能替换 Stage-1。没有替换。Fisher-6 仍是 `[1, 13–17]` **97.75**。
- Molmo Fisher-16：v2 均匀已经等于 Taylor-16 `[1–4, 24–35]`（98.10）。这再次说明「翻到 Taylor」来自校准集，不是 α。

bootstrap 一致性 c：CARE 加权在 π0.5 Fisher（0.90 vs 0.93）、π0.5 GateProbe（0.78 vs 0.91）、Molmo Fisher（0.71 vs 0.75）、Molmo GateProbe（0.41 vs 0.55）上**低于**均匀。加权没有让排序更稳。

v1 CARE 与 v2 均匀在 Molmo GateProbe Drop-16 上**不是同一集**：v1 CARE 是 Taylor `[1–4, 24–35]`（表 1 的 98.10）；v2 是新集 `[1, 2, 3, 21, 24–35]`。所谓「CARE 集」随校准定义改变。不要把两套数字混成一个方法。

SmolVLA GateProbe v2 是 `[11, 13–15]`，相对 Stage-1 `[7, 13–15]` Jaccard 0.60。没有闭环。不要重评历史 72.00。

Sources: `results/stage2_care_v2/{e1.md,e1.json,shared.json}`。脚本：`scripts/stage2_care_v2/{protocol,analyze_e1,run_e0_e1}.sh`。

---

## 10. D/S 时间戳组成（2026-10-10，只打分，没有新闭环）

假设：校准时间戳若改成「动作突变最大的那一帧」，冗余排序会比均匀或平滑帧更准。E1 已经证明 α 加权本身不改 Drop-k，所以这一步不再乘 α，只换时间戳。

同一 480 条轨迹（与 E1 相同 episode）：

| 臂 | 定义 | 白化 d 中位数 |
|---|---|---:|
| U | v2 随机时间戳，等权 | 1.933 |
| D | 该 episode 内 whitened-d **最大**的合法起点 | 4.423 |
| S | 该 episode 内、且 d ≥ 全库 10 分位（p10 = 0.756）的 **最小** d | 0.780 |

D 与 S 没有重叠帧（`n_same_disc_smooth = 0`）。D 与 U 也没有重叠（`n_same_disc_uniform = 0`）。S 与 U 只有 5 帧相同。S 没有触发 p10 回退（`n_smooth_fallback = 0`）。等权，不用 α。**不开新闭环**：层集若等于已经评过的集合，成功率直接映射；新集保持「new」，不排队。

| Stack | Method | k | U | D | S | J(D,U) | J(S,U) | J(D,S) | U 映射 | D 映射 | S 映射 |
|---|---|---:|---|---|---|---:|---:|---:|---|---|---|
| pi05 | fisher | 6 | `[2, 13–17]` | `[2, 13–17]` | `[1, 13–17]` | 1.00 | 0.71 | 0.71 | CARE-v1 GP-6 97.00 | CARE-v1 GP-6 97.00 | **S1 Fisher-6 97.75** |
| pi05 | fisher | 8 | `[1, 2, 12, 13–17]` | `[1, 2, 11, 13–17]` | `[1–3, 13–17]` | 0.78 | 0.78 | 0.78 | new | new | S1 Fisher-8 85.50 |
| pi05 | taylor | 6 | `[1, 2, 14–17]` | `[1, 2, 14–17]` | `[1, 2, 14–17]` | 1.00 | 1.00 | 1.00 | S1 Taylor-6 97.25 | S1 Taylor-6 97.25 | S1 Taylor-6 97.25 |
| pi05 | taylor | 8 | `[1–3, 13–17]` | `[1–3, 13–17]` | `[1–3, 13–17]` | 1.00 | 1.00 | 1.00 | S1 Taylor-8 85.50 | S1 Taylor-8 85.50 | S1 Taylor-8 85.50 |
| pi05 | gateprobe | 6 | `[2, 13–17]` | `[2, 11, 14–17]` | `[2, 13–17]` | 0.71 | 1.00 | 0.71 | CARE-v1 GP-6 97.00 | new | CARE-v1 GP-6 97.00 |
| pi05 | gateprobe | 8 | `[2, 4, 10, 13–17]` | `[2, 10, 11, 13–17]` | `[2, 3, 10, 13–17]` | 0.78 | 0.78 | 0.78 | S1 GP-8 85.75 | new | new |
| smolvla | fisher | 4 | `[10, 12, 13, 14]` | `[10, 12, 13, 14]` | `[6, 10, 12, 14]` | 1.00 | 0.60 | 0.60 | new | new | new |
| smolvla | fisher | 8 | `[6–8, 10–14]` | `[7, 9–15]` | `[4, 6–8, 10–12, 14]` | 0.60 | 0.78 | 0.45 | new | new | new |
| smolvla | taylor | 4 | `[12–15]` | `[12–15]` | `[12–15]` | 1.00 | 1.00 | 1.00 | S1 Taylor-4 78.50 | S1 Taylor-4 78.50 | S1 Taylor-4 78.50 |
| smolvla | taylor | 8 | `[8–15]` | `[8–15]` | `[7, 8, 10–15]` | 1.00 | 0.78 | 0.78 | new | new | new |
| smolvla | gateprobe | 4 | `[11, 13–15]` | `[11, 13–15]` | `[11, 13–15]` | 1.00 | 1.00 | 1.00 | new | new | new |
| smolvla | gateprobe | 8 | `[7, 8, 10–15]` | `[7, 9–15]` | `[7, 8, 10–15]` | 0.78 | 1.00 | 0.78 | new | new | new |
| molmoact2 | fisher | 16 | `[1–4, 24–35]` | `[1–4, 24–35]` | `[1–4, 5, 25–35]` | 1.00 | 0.88 | 0.88 | **S1 Taylor-16 98.10** | **S1 Taylor-16 98.10** | S1 Fisher-16 72.75 |
| molmoact2 | fisher | 17 | `[1–5, 24–35]` | `[1–4, 23–35]` | `[1–5, 24–35]` | 0.89 | 1.00 | 0.89 | S1 Fisher-17 70.60 | S1 Taylor-17 89.60 | S1 Fisher-17 70.60 |
| molmoact2 | taylor | 16 | `[1–4, 24–35]` | `[1–4, 24–35]` | `[1–4, 24–35]` | 1.00 | 1.00 | 1.00 | S1 Taylor-16 98.10 | S1 Taylor-16 98.10 | S1 Taylor-16 98.10 |
| molmoact2 | taylor | 17 | `[1–4, 23–35]` | `[1–4, 23–35]` | `[1–4, 23–35]` | 1.00 | 1.00 | 1.00 | S1 Taylor-17 89.60 | S1 Taylor-17 89.60 | S1 Taylor-17 89.60 |
| molmoact2 | gateprobe | 16 | `[1–3, 21, 24–35]` | `[2, 3, 22, 23, 24–35]` | `[1–3, 5, 24–35]` | 0.78 | 0.88 | 0.78 | new | new | new |
| molmoact2 | gateprobe | 17 | `[1–3, 5, 21, 24–35]` | `[1–3, 22, 23, 24–35]` | `[1–3, 5, 6, 24–35]` | 0.79 | 0.89 | 0.79 | new | new | new |

读法（已经映射到闭环的行就够否定假设，不必为 `new` 开评）：

1. **Taylor 经营点与悬崖在 U/D/S 下都不动。** π0.5 Taylor-6/8、SmolVLA Taylor-4、Molmo Taylor-16/17 的 Jaccard(U,D,S) = 1.0。推荐选择器对时间戳类型不敏感。
2. **π0.5 Fisher-6：平滑帧更好。** S = Stage-1 Fisher-6 **97.75**；U = D = CARE-v1 GateProbe-6 **97.00**。方向和「突变帧更能找出冗余」**相反**。
3. **Molmo Fisher-16：突变/均匀更好。** U = D = Taylor-16 **98.10**；S = Fisher-16 **72.75**。方向又和 π0.5 Fisher-6 **相反**。
4. **Molmo Fisher-17：** D = Taylor-17 89.60，U = S = Fisher-17 70.60。这里 D 更好，仍不能外推成规则。
5. 其余 `new` 集没有闭环。不要为它们开评。映射过的集合已经表明：时间戳类型的「对错」随栈和指标改变，不是方法。

Sources: `results/stage2_care_v2/composition/{composition.md,composition.json,arms.json}`。脚本：`scripts/stage2_care_v2/{select_arms,analyze_ds,run_ds}.sh`。`EXIT:0` 2026-10-10。

---

## 11. 失败原因（综合）

CARE 要回答的问题是：Stage-1 的经营点会不会只是「每帧等权」造成的？若加权、或改采突变帧，能不能自动得到更好的层集？下面八条合在一起，答案是否定的。任何一条单独都不够，但也不缺一条。

### 11.1 经营点层集不变

π0.5 Fisher-6、Molmo Taylor-16、SmolVLA Taylor-4 在 v1 CARE 下 Jaccard = 1.0。CARE 没有发现它们，均匀校准已经发现它们。把 CARE 写成「找出可删层的方法」是错的。

### 11.2 「变好」的切口只是近邻平局，且落到已有集合

| 切口 | 对换 | 新集 | 闭环 | 是否新发现 |
|---|---|---|---|---|
| Molmo Fisher-16 | L5 ↔ L24（0.00319 vs 0.00327） | = Taylor-16 | 映射 **98.10**，不重跑 | 否，Stage-1 Taylor 已有 |
| π0.5 GateProbe-6 | L4 ↔ L13 | `[2, 13–17]` | 原集重评 94.25 → 新集 **97.00** | 否，等于 IGIA-6 / 接近天花板，仍低于 Fisher-6 97.75 |
| SmolVLA Fisher-4 | L11 ↔ L13 | `[10, 12, 13, 14]` | 69.75 → **71.25**（+1.50） | 否，仍低于 Taylor-4 **78.50** 达 7.25 |

三次近邻翻转加一次被闸门拦住的失败，不是选层器。

### 11.3 反例：Molmo GateProbe Drop-17 的 7 维崩盘

同一套 CARE 权重，7 维 Drop-17 = `[1–4, 6, 24–35]`，2000 集 **19.80**（Spatial 45.0 / Object 16.4 / Goal 15.6 / Long 2.2），相对历史 82.65 为 **−62.85**。臂 6 维是另一集 `[1–4, 22, 24–35]`，**97.65**。两种维度 Jaccard 不一致。L6 与 L22 的 7 维分差约 6.7×10⁻⁶，只有中位相邻差的 1.3%，触发平局闸门。闸门拦住了错误集，但也等于承认：**没有闸门就不能用 CARE**；有闸门时，这个切口根本不应被采纳。不要把 97.65 写成 CARE 的胜利。不要用这条替换 Taylor Drop-16。

π0.5 GateProbe Drop-7 的 +0.75（94.00 vs 93.25）在噪声带内，仍低于 Fisher Drop-7 的 96.00。Drop-8 的 +2.25（88.00 vs 85.75）是真的，但仍低于经营点 Fisher-6，不能把 GateProbe 抬成主选择器。

### 11.4 加权不是通用的校准改进

CARE Fisher 在 Molmo Drop-16 上翻到 Taylor；**CARE IGIA 不翻**（L5 = 104.28 vs L24 = 131.32），Drop-16 仍删 L5、留 L24，历史 **72.75**。同一套 α，换一个指标就不救悬崖。CARE 不能写成「校准加权普遍提高冗余度量」。

### 11.5 固定校准帧之后，α 根本不改 Drop-k

第 9 节 E1：九个 (栈 × 指标) 的 J(CARE, uniform) = 1.0。v1 相对 Stage-1 的差异来自 480 帧共享集和逐帧估计器。把「集合变了」算成 CARE 的贡献，是把校准集更换算进了方法。

顺带：bootstrap c 在多行上加权更差，不是更稳。

### 11.6 「突变帧更准」不成立

第 10 节 D/S：Taylor 对 U/D/S 不变。π0.5 Fisher-6 上平滑帧映射到最好的 97.75，突变帧映射到 97.00。Molmo Fisher-16 上方向相反（突变/均匀 98.10，平滑 72.75）。没有跨模型的时间戳规则。

### 11.7 高 α 分位也不是新经营点

只保留 480 帧里 α 最高的 120 帧、等权（不是 α 加权均值）。π0.5 GateProbe Drop-6 变成 `[2, 3, 14–17]`，400 集 **97.25**（389/400）。相对 v2 均匀 / CARE 加权的 97.00 是 +0.25（1 集，噪声），打平完整模型 97.25，仍低于 Fisher-6 **97.75** 的 0.50。Jaccard vs `[2, 13–17]` = 0.71。这是覆盖率消融，不是第二套 CARE。不要覆盖 Stage-1 的 95.00 或 CARE 加权的 97.00。

### 11.8 方法不自洽，且不能用成功率回搜

v1 与 v2 对 Molmo GateProbe Drop-16 给出两个不同的「CARE 集」（Taylor 98.10 vs 新集 `[1, 2, 3, 21, 24–35]`）。H、白化、7 维 / 6 维、分位阈值都是未用 SR 调过的选择；一旦用闭环去选它们，CARE 就变成对成功率的搜索。平局闸门、维度一致性闸门已经说明自动采用不安全。继续扩定义（随机权重、反向权重、留出模型、再换窗口）不会把负结果变成方法。2026-10-10 停止。

---

## 12. 验证实验清单（不得遗漏）

下列全部已结束。Jaccard = 1.0 或已映射到评过集合的，不要重开闭环。历史 Stage-1 目录不要覆盖。

### 12.1 v1 加权打分（Stage-1 校准集 + α，2026-10-05）

| 栈 × 指标 | k | Stage-1 集 | CARE 集 | Jaccard | 闭环 |
|---|---:|---|---|---:|---|
| π0.5 Fisher no-L0 | 6 | `[1, 13–17]` | 同 | 1.00 | 不停；97.75 |
| π0.5 Taylor no-L0 | 6 | `[1, 2, 14–17]` | 同 | 1.00 | 不停；97.25 |
| π0.5 GateProbe no-L0 | 6 | `[2, 4, 14–17]` | `[2, 13–17]` | 0.71 | 成对 400 集：94.25 → **97.00** |
| π0.5 Fisher / Taylor no-L0 | 8 | `[1–3, 13–17]` | 同 | 1.00 | 不停；85.50 |
| π0.5 GateProbe no-L0 | 7 | `[2, 4, 13–17]` | `[2, 9, 13–17]` | 0.75 | 400 集 **94.00** vs 史 93.25（噪声） |
| π0.5 GateProbe no-L0 | 8 | `[2, 4, 10, 13–17]` | `[2, 4, 9, 13–17]` | 0.78 | 400 集 **88.00** vs 史 85.75 |
| Molmo Taylor no-L0 7d=6d | 16 | `[1–4, 24–35]` | 同 | 1.00 | 不停；98.10 |
| Molmo Fisher no-L0 7d=6d | 16 | `[1–4, 5, 25–35]` | `[1–4, 24–35]` | 0.88 | 映射 Taylor 98.10，不重跑 2000 |
| Molmo GateProbe no-L0 | 16 | `[2–4, 6, 24–35]` | `[1–4, 24–35]` | 0.88 | 映射 Taylor 98.10 |
| Molmo Fisher / Taylor no-L0 | 17 | 史集 | 同 | 1.00 | 不停；70.60 / 89.60 |
| Molmo GateProbe no-L0 7d | 17 | `[2–4, 6, 22, 24–35]` | `[1–4, 6, 24–35]` | 0.89 | 2000 集 **19.80**（反例） |
| Molmo GateProbe no-L0 6d | 17 | 同上 | `[1–4, 22, 24–35]` | 0.89 | 2000 集 **97.65**；闸门拒绝 |
| SmolVLA Taylor | 4 | `[12–15]` | 同（整条 redundancy_order 也同） | 1.00 | 不停；78.50 |
| SmolVLA GateProbe | 4 | `[7, 13–15]` | 同 | 1.00 | 不停；72.00 |
| SmolVLA Fisher | 4 | `[10, 11, 12, 14]` | `[10, 12, 13, 14]` | 0.60 | 成对 400 集：69.75 → **71.25** |
| π0.5 IGIA | 6 / 7 | `[2, 13–17]` / `[2, 5, 13–17]` | 同 | 1.00 | 不停；97.00 / 93.00 |
| Molmo IGIA no-L0 7d=6d | 16 | `[1–4, 5, 25–35]` | 同 | 1.00 | 不停；**不是** 98.10 |
| Molmo IGIA no-L0 | 17 | `[1–5, 24–35]` | 同 | 1.00 | 不停；70.60 |
| SmolVLA IGIA | 4 | `[10, 12, 14, 15]` | 同 | 1.00 | 不停；73.75 |

闭环目录（均 `seed=1000`，`compile_model=false`，keep-KV，`EXIT:0`）：

- `results/eval_drop/k6_gateprobe_{reval,action_discontinuity}/`（400）
- `results/eval_drop/k7_gateprobe_action_discontinuity/`、`k8_gateprobe_action_discontinuity/`（400）
- `results/smolvla_libero/drop4_fisher_{reval,action_discontinuity}/`（400）
- `results/molmoact2_libero/drop17_gateprobe_action_discontinuity{,_arm6}/`（2000）

打分：`results/stage2_action_weight/`。α=1 sanity：π0.5 Fisher / GateProbe / Taylor / IGIA 在 batch 0 上 `rel_abs_diff=0`。

### 12.2 v2 共享 480 帧（2026-10-09–10）

| 实验 | 做法 | 闭环 | 结果 | 目录 |
|---|---|---|---|---|
| E1 均匀 vs α | 同一 480 帧；9 个 (栈 × Fisher/Taylor/GateProbe) | 无 | J(CARE, uni) = 1.00（九行全是）；Taylor 经营点不动；π0.5 Fisher-6 的 v2 均匀 ≠ Stage-1，未替换 | `results/stage2_care_v2/e1.{md,json}` |
| 高 α 分位 | 480 帧中 α 最高的 120 帧等权，不是 α 均值；π0.5 GateProbe-6 | 400 集 GPU1 **97.25**（389/400） | 相对 97.00 为噪声；仍低于 Fisher-6 97.75 | `results/eval_drop/k6_gateprobe_v2_hialpha/` |
| D/S 时间戳 | 同 480 episode；U 随机、D 最大 d、S 高于 p10 的最小 d；等权无 α；经营点 k 与悬崖 k | 无新闭环 | Taylor 不变；π0.5 Fisher-6 上 S 更好；Molmo Fisher-16 上 D/U 更好 | `results/stage2_care_v2/composition/` |

高 α 的 drop 是 `[2, 3, 14, 15, 16, 17]`，套件 97 / 99 / 93 / 100。`EXIT:0` 2026-10-10T09:33:02Z。不要覆盖 `k6_gateprobe` 或 `k6_gateprobe_action_discontinuity`。

### 12.3 闸门本身也是验证

采纳新集必须同时：维度一致（若该栈打了 7 维和 6 维）、切口不是数值平局（分差 ≥ 中位相邻差的 10%）、Jaccard < 1。Molmo GateProbe Drop-17 三条里前两条失败，所以 19.80 / 97.65 都不进经营点。闸门是在闭环**之前**写进方法的，不是事后用 19.80 发明的。它验证的是「不能自动采用 CARE 新集」，不是「CARE 可用只要加上闸门」。

### 12.4 明确没有做、也不再做

- 随机权重、反向权重（低 d 权重大）。
- 用闭环 SR 调 α、H、7 维 / 6 维、高 α 分位阈值。
- 为 E1 / D/S 里标记 `new` 的层集开闭环。
- 为 Jaccard = 1.0 的集合重评（含三个经营点、π0.5 Fisher/Taylor Drop-8、Molmo Fisher/Taylor Drop-17、三栈 CARE IGIA、SmolVLA GateProbe Drop-4）。
- 把已经变成 Taylor Drop-16 的 Molmo CARE Fisher / v1 CARE GateProbe-16 再跑 2000 集。
- 留出模型、π0 / GR00T / RoboTwin 上的 CARE。
- 把 CARE 或 D/S 写回主贡献。

LEDGE 用的是 CARE 分数曲线上的漂移，本身另见 `ledge_libero.md`。LEDGE 也不是方法（无留出，大约只省一档闭环）。不要把 LEDGE 的区间写成 CARE 的成功。

## Checks

Weight dumps: `results/stage2_action_weight/weights_{pi05,smolvla,molmoact2}.pt`. π0.5 α=1 sanity: Fisher, GateProbe, Taylor, and IGIA `rel_abs_diff=0` on batch 0. Molmo / SmolVLA CARE Fisher, Taylor, and GateProbe divide by N=32; CARE IGIA does not (Stage-1 IGIA has no 1/N). Historical `k6_gateprobe`, `k7_gateprobe_nol0`, `k8_gateprobe_nol0`, and `drop4_fisher` directories were not overwritten. π0.5 CARE GateProbe Drop-6 / Drop-7 / Drop-8 wrote `eval_info.json` with `n_episodes=400` and `EXIT:0` (97.00 / 94.00 / 88.00). CARE v2 high-α GateProbe Drop-6 (`k6_gateprobe_v2_hialpha`) wrote `eval_info.json` with `n_episodes=400` and `EXIT:0` (97.25, 389/400), drop `[2, 3, 14, 15, 16, 17]`. Molmo CARE GateProbe Drop-17 wrote `eval_info.json` with `n_episodes=2000` and `EXIT:0` (7-d 19.80, arm 6-d 97.65). CARE IGIA operating-point sets are Jaccard 1.0; do not re-eval 97.00 / 72.75 / 73.75. SmolVLA CARE GateProbe Drop-4 is Jaccard 1.0; do not re-eval 72.00.

v2 E1: `results/stage2_care_v2/e1.json` has nine rows, all `jaccard_care_vs_uniform == 1.0`, n=480, ESS=384.1. D/S: `results/stage2_care_v2/composition/composition.json` protocol `care_v2_composition`, no closed loop; `arms.json` has `n_same_disc_smooth=0`, `n_smooth_fallback=0`. Do not launch evals for composition tags marked `new`. Do not retune α or H on SR. CARE is a negative result as of 2026-10-10.
