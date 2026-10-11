# CARE：动作不连续加权校准（Calibration with Action-discontinuity REweighting，2026-10-05）

权威表：`functional_necessity/main_results/care_libero.md`。本文件是打分草稿，数字以那张表为准。

定义：对损失所用的**归一化动作**，取窗口 H=10 内相邻步的平均 L2。再除以全体均值，使均匀场精确为 1。α 在损失外面。没有用闭环成功率调权重。操作点不开闭环；次优 π0.5 GateProbe / SmolVLA Fisher 后来成对评了 400 集。

三套栈的张量不一样，论文里不要写成同一种：

| 栈 | 权重张量 | 损失张量 |
|---|---|---|
| π0.5 | 已存 50 步块的前 10 个有效步 | 同一 50 步块 |
| SmolVLA | 已存 50 步块的前 10 个有效步 | 同一 50 步块 |
| MolmoAct2 | 同 episode 后续 10 帧，经与损失相同的 masked-normalizer + clamp | 原来那一步 `(1,1,32)` |

Molmo 第一步与存盘动作的 max abs diff = 0。π0.5 有 5/512 帧窗口不足 2 步，用中位数填充。

| 栈 | n | ESS (7d) | Spearman 7d vs 臂 6d | GPU |
|---|---:|---:|---:|---|
| π0.5 | 512 | 386.80 | 0.9375 | 只打 7d |
| SmolVLA | 32 | 22.70 | 0.9824 | 只打 7d |
| MolmoAct2 | 32 | 23.30 | 0.7005 | 7d 和 6d |

π0.5 Fisher 仍按 64 个原始 batch：每个 batch 4 次 mb=2 前向，对 per-sample loss 加权求和得到加权平均梯度，再平方；64 个 batch 等权。α=1 时 batch 0 的 L0 Fisher 贡献与 Stage-1 原归约 **逐位相同**（`rel_abs_diff=0`）。禁止把 512 帧拉平再平方。

## 操作点 Jaccard

对照钉在 Stage-1：π0.5 Fisher Drop-6 97.75；Molmo Taylor Drop-16 98.10；SmolVLA Taylor Drop-4 78.50。0.25–0.75 仍当噪声。π0.5 / Molmo 已在天花板附近，集合变了也只问新层集是否还守住，不求更高。

| 栈 | Stage-1 Drop | 加权 Drop | Jaccard | 闭环 |
|---|---|---|---:|---|
| π0.5 Fisher k=6 no-L0 | `[1,13–17]` | `[1,13–17]` | 1.0 | 不停、不跑 |
| Molmo Taylor k=16 no-L0（7d） | `[1–4,24–35]` | `[1–4,24–35]` | 1.0 | 不停、不跑 |
| Molmo Taylor k=16 no-L0（臂 6d） | 同上 | 同上 | 1.0 | 不停、不跑 |
| SmolVLA Taylor k=4 | `[12–15]` | `[12–15]` | 1.0 | 不停、不跑 |
| SmolVLA GateProbe k=4 | `[7,13–15]` | `[7,13–15]` | 1.0 | 不停、不跑（72.00） |

SmolVLA Taylor 的完整 `redundancy_order` 与 Stage-1 相同。SmolVLA GateProbe Drop-4 同集；Drop-8 会 L12 ↔ L9，不开闭环。π0.5 / Molmo 的操作点集合不变，更靠后的名次有对换（例如 π0.5 在第 7 名之后 L4 与 L10）。这不够构成新的 Drop 集。

## 次优方案 Jaccard

同一套权重，下面三条 Stage-1 次优的 k 切口都落在相邻名次上，加权后各对换一层。π0.5 / SmolVLA Fisher 随后成对闭环；Molmo 新集等于已评 Taylor，不重跑。SmolVLA GateProbe Drop-4 不是近邻切口，Jaccard 1.0，不跑。

| 栈 | Stage-1 Drop | 加权 Drop | Jaccard | 闭环 |
|---|---|---|---:|---|
| Molmo Fisher k=16 no-L0（7d） | `[1–4,5,25–35]` | `[1–4,24–35]` | 0.882 | 等于已评 Taylor Drop-16（98.10），不重跑 |
| Molmo Fisher k=16 no-L0（臂 6d） | 同上 | 同上 | 0.882 | 同上 |
| π0.5 GateProbe k=6 no-L0 | `[2,4,14–17]` | `[2,13–17]` | 0.714 | 原集重评 94.25；加权新集 **97.00**（+2.75） |
| SmolVLA Fisher k=4 | `[10,11,12,14]` | `[10,12,13,14]` | 0.600 | 原集重评 69.75；加权新集 **71.25**（+1.50） |

对换都是 k 边界上的近邻：Molmo L5 ↔ L24（Stage-1 Fisher 分数 0.00319 vs 0.00327）；π0.5 GateProbe L4 ↔ L13（Stage-1 redundancy 第 6 / 7 名，L0 仍在最后，no-L0 等于 native）；SmolVLA L11 ↔ L13（第 4 / 5 名）。π0.5 GateProbe 仍按 64×8 原 batch、4× mb=2；α=1 时 batch 0 的 signed inner 与 Stage-1 **逐位相同**（`rel_abs_diff=0`）。Molmo / SmolVLA Fisher 的 α 在损失外，`(1/N) Σ α_i (g_i² θ²)`。

## 次优闭环（2026-10-05，成对 400 集）

Stage-1 原集另开目录重评，不覆盖 95.00 / 69.25。0.25–0.75 仍当噪声。

| 对比 | 历史 | 同期重评 | 加权新集 | 新集 − 同期重评 |
|---|---:|---:|---:|---:|
| π0.5 GateProbe Drop-6 | 95.00 | 94.25 | **97.00** | **+2.75** |
| SmolVLA Fisher Drop-4 | 69.25 | 69.75 | **71.25** | **+1.50** |

π0.5 加权新集相对本机锚点 97.25 为 −0.25，守住天花板。SmolVLA 加权新集仍低于 Taylor 78.50（−7.25）和锚点 83.00（−11.75）。原集重评与历史的差（−0.75 / +0.50）在噪声内，所以 +2.75 / +1.50 不是把一次坏评拿来对比。

产物：`results/stage2_action_weight/`；闭环 `results/eval_drop/k6_gateprobe_{reval,action_discontinuity}/`、`results/smolvla_libero/drop4_fisher_{reval,action_discontinuity}/`。

## 悬崖切口（π0.5 Drop-8 / Molmo Drop-17，2026-10-05）

同一套权重。Jaccard 1.0 不开闭环。π0.5 GateProbe Drop-7 / Drop-8 与 Molmo GateProbe Drop-17（7 维、臂 6 维）都已闭环。

| 栈 | Stage-1 Drop | CARE Drop | Jaccard | 闭环 |
|---|---|---|---:|---|
| π0.5 Fisher k=8 no-L0 | `[1–3,13–17]` | `[1–3,13–17]` | 1.0 | 不停、不跑（85.50） |
| π0.5 Taylor k=8 no-L0 | `[1–3,13–17]` | `[1–3,13–17]` | 1.0 | 同 Fisher，不跑 |
| π0.5 GateProbe k=8 no-L0 | `[2,4,10,13–17]` | `[2,4,9,13–17]` | 0.778 | L10 ↔ L9；闭环 **88.00** vs 历史 85.75（+2.25） |
| π0.5 GateProbe k=7 no-L0 | `[2,4,13–17]` | `[2,9,13–17]` | 0.750 | L4 ↔ L9；闭环 **94.00** vs 历史 93.25（+0.75，噪声） |
| Molmo Fisher k=17 no-L0（7d=6d） | `[1–5,24–35]` | `[1–5,24–35]` | 1.0 | 不停、不跑（70.60） |
| Molmo Taylor k=17 no-L0（7d=6d） | `[1–4,23–35]` | `[1–4,23–35]` | 1.0 | 不停、不跑（89.60） |
| Molmo GateProbe k=17 no-L0（7d） | `[2–4,6,22,24–35]` | `[1–4,6,24–35]` | 0.889 | L22 ↔ L1；闭环 **19.80** vs 历史 82.65（−62.85） |
| Molmo GateProbe k=17 no-L0（臂 6d） | 同上 | `[1–4,22,24–35]` | 0.889 | L6 ↔ L1；闭环 **97.65** vs 历史 82.65（+15.00） |

Drop-16 上 Molmo Fisher 的 L5 ↔ L24 对换，到 Drop-17 两层都进删除集，所以 Fisher Drop-17 与 Stage-1 相同。

## IGIA（操作点，2026-10-05）

三栈操作点 Jaccard 全是 **1.0**，不开闭环。π0.5 α=1 时 `rel_abs_diff=0`。Molmo IGIA **没有**随 Fisher 翻到 Taylor：L5=104.28 vs L24=131.32，Drop-16 仍删 L5、留 L24（72.75）。

| 栈 | Stage-1 Drop | CARE Drop | Jaccard | 闭环 |
|---|---|---|---:|---|
| π0.5 IGIA k=6 | `[2,13–17]` | `[2,13–17]` | 1.0 | 不停、不跑（97.00） |
| π0.5 IGIA k=7 no-L0 | `[2,5,13–17]` | `[2,5,13–17]` | 1.0 | 不停、不跑（93.00） |
| Molmo IGIA k=16 no-L0（7d=6d） | `[1–4,5,25–35]` | `[1–4,5,25–35]` | 1.0 | 不要写成 98.10 |
| SmolVLA IGIA k=4 | `[10,12,14,15]` | `[10,12,14,15]` | 1.0 | 不停、不跑（73.75） |
