# LEDGE: Layer-drop Endpoint by Drift-Guided Evaluation

中文名：**漂移引导的删层崖线**。沿冗余序从低到高删语言模型层时，闭环成功率先保持平台，到某一层才掉下去。开环动作漂移把这段崖线收成一个小区间，区间内的闭环给出该模型在当前任务上可接受的最大删层数。

LEDGE 不训练，也不用成功率去拟合漂移阈值。表格里的层数来自闭环。漂移只决定测哪些 k。

局部锚点：π0.5 **97.25**，MolmoAct2 **97.50**，SmolVLA **83.00**。跳层为 keep-KV。低于锚点 5 分视为离开平台（π0.5 以 92.25 为界，MolmoAct2 以 92.50 为界）。0.25–0.75 的平均分差是噪声。

---

## 现象

按重要性从低到高、一层层去掉语言模型层时，闭环成功率不是匀速变差。前面若干层几乎停在锚点附近，形成平台；删到下一层，成功率才明显掉下去。平台上的最后一层，就是该模型在这一任务、这一删除序上可接受的最大删层数。

| 模型 | 删除序 | 平台上的最后一档 | 掉下去的下一档 | 可接受的最大 k |
|---|---|---|---|---:|
| π0.5 | Fisher CARE | k=7，96.00 | k=8，85.50 | 7 |
| π0.5 | GateProbe CARE | k=7，94.00 | k=8，88.00 | 7 |
| π0.5 | GateProbe 均匀 | k=7，93.25 | k=8，85.75 | 7 |
| MolmoAct2 | Taylor CARE | k=16，98.10 | k=17，89.60 | 16 |
| MolmoAct2 | Fisher 均匀 | k=15，98.10 | k=16，72.75 | 15 |

π0.5 Fisher 在掉下去之前更平：k=6 为 97.75，k=7 为 96.00，到 k=8 才落到 85.50。MolmoAct2 Taylor 在 k=12 到 k=16 保持 97.70–98.10，k=17 落到 89.60。Fisher 均匀序的崖线更陡，k=15 仍是 98.10，k=16 降到 72.75，k=17 为 70.60，再删一层并没有比第一下掉得更多。

同一模型上，崖线的位置随删除序变化，因为不同的序在不同的 k 放进关键层。要找的是每一条序自己的平台尽头，不是一个对所有序通用的层号。

SmolVLA 的 Taylor 序没有这段平台。只删最后一层时开环动作与完整模型相同，闭环仍是 79.25，相对锚点 83.00 低了约 4 分；k=2 到 k=4 为 80.50、77.75、78.50。这几分落在单次评测的起伏里，不能当成崖线。

---

## 方法

开环动作漂移衡量「删到第 k 层时，执行动作离开完整模型多远」。它沿同一条冗余序是大体上升的，但上升最快的那一刀通常落在崖线上或崖线之后：那一刀放进来的是靠前的关键层。自身动作噪声则把还几乎不改变动作的尾部层标出来。两者把平台尽头夹在一个区间里，闭环只在这个区间里查找。

设完整模型在冻结推理噪声 ε 下、前 10 个执行步的动作为 a(ε)，删层前缀 D_k 的动作为 a_k(ε)。CARE 的帧权重为 α_i（均值定为 1）。有多份噪声时先对噪声取平均：

$$
r(k)=\mathbb{E}_{\varepsilon}\left[\sum_i \alpha_i\,\|a_k(\varepsilon)-a(\varepsilon)\|_2\;\middle/\;\sum_i \alpha_i\right].
$$

完整模型换一份噪声再走一遍，得到自身波动 σ。它用同样的加权 L2，只是两边都是完整模型。区间的两端都不读成功率：

$$
k_{\mathrm{lo}}=\max\{k:r(k)\le \sigma\},
\qquad
k_{\mathrm{hi}}=\arg\max_k \big(r(k)-r(k-1)\big)-1.
$$

k_lo 是动作改变还没有超过模型自己换噪声时的波动的最后一刀。k_hi 是动作漂移一步增量最大的那一刀的前一刀。可接受的最大删层数是区间 [k_lo, k_hi] 内、闭环成功率仍留在平台上的最大 k。沿这条序，成功率掉下去之后不再恢复，因此可以在区间里二分，不必从 k=1 逐档评到序的末尾。

π0.5 用 512 帧、2 份噪声；MolmoAct2 用 32 帧、4 份噪声。权重是 CARE 的 7 维权重。k=0 时 r=0。

在已经测过崖线的两条序上，测量到的 k 都落在区间内：

| 模型 | 删除序 | k_lo | k_hi | 区间内测得的 k |
|---|---|---:|---:|---:|
| π0.5 | Fisher CARE | 3 | 8 | 7 |
| MolmoAct2 | Taylor CARE | 9 | 19 | 16 |

π0.5 的 σ 为 0.60，k=3 时 r 仍低于 σ，k=4 起超过；增量最大在第 9 刀（进入 L4），所以上界是 8。MolmoAct2 的 σ 为 0.079，上界来自第 20 刀进入 L6 时的增量。

若不对漂移做任何使用、只假设成功率沿序单调，对全部 k 二分，π0.5 大约要 4 次闭环，MolmoAct2 大约要 5 次。LEDGE 把范围收成上面的区间后，大约是 3 次和 4 次。省下的是一档左右。曲线的作用是说明崖线落在噪声底和最大增量之间，头部几乎不动动作的层和尾部关键层不必再逐档闭环。

---

## 适用前提

1. 删除序上要有一段成功率平台。SmolVLA 没有这段平台，而且 GateProbe 第 2 刀的漂移仍低于 σ，闭环已经是 72.75。下界在这里不成立，LEDGE 不给出它的最大删层数。
2. 沿该序，成功率离开平台后不恢复。现有的 π0.5 与 MolmoAct2 序上没有反例。
3. 区间由漂移定义，层数由区间内的闭环读出。不把 k_hi 或最大增量那一刀写成可接受的最大删层数：π0.5 的上界 8 对应的闭环是 85.50，MolmoAct2 的上界 19 已经在第 17 刀的 89.60 之后。
4. 容忍的漂移大小因模型而异，不在模型之间共用一个倍数，也不把一条序上读出的 r 拿去判定另一条序。

---

# Lab notes (not for the paper)

Stage-3 of the flow-matching VLA pruning study. Chinese name: **漂移引导的删层崖线（LEDGE）**. Date: 2026-10-07.

The name refers to the flat edge before the drop. An earlier offline rule (largest one-step drift increment, accepted only when it exceeds 3× the median of earlier increments) is not the method above. On the single-seed curves it returned k_safe = 8 for π0.5 Fisher (closed-loop at k=8 is 85.50; the plateau ends at 7) and k_safe = 19 for MolmoAct2 Taylor (the plateau ends at 16). The same curves show why a first-jump rule and a flow-matching-loss rule do not replace it: π0.5 already jumps at k=4 and k=6 while SR is 97.75; MolmoAct2 jumps by about 10× at k=10 (L1) while k=12..16 stay at 97.7–98.1; at the Taylor cliff (k=17, SR 89.60) the loss goes down, while the large loss increase is at k=16 where SR is still 98.10.

Two other offline cuts were computed and do not locate k either. r(k) ≤ 1·σ stops too early (π0.5 at 3–4, MolmoAct2 at 9–11). A 100-step closed-loop object-position pilot (20 episodes, c=1 against the full policy's own state spread) also put the still-safe side above the spread: π0.5 k=7 was 2.1×σ_state and k=8 was 2.4×; MolmoAct2 k=16 was 1.9× and k=17 was 4.6×. Those negative results are why the paper procedure keeps closed-loop inside the bracket instead of emitting k from drift alone.

The bracket ends were adopted after those rules missed in opposite directions (noise floor too early, largest increment too late). On the two orders in the table the measured k falls inside. That containment was not shown on a held-out model, and it is not claimed as a guarantee. A tolerance transferred from one order's last passing r to other orders was also checked offline and is not the method: the test orders are not independent, the calibration order changes the answer by about one layer, and MolmoAct2's pass and fail drift values sit inside each other's bootstrap intervals (0.746 vs 0.753).

Raw curves: `results/stage3_ledge/pi05_fisher.json`, `molmoact2_taylor.json`, `smolvla_taylor.json` (single seed). Multi-seed r and σ: `results/stage3_ledge/v2/`. Object-state pilot: `results/stage3_ledge/pilot_state/`.

Vision LEDGE on π0.5 language Fisher Drop-6, finished 2026-10-08. Reference is that Drop-6 model with full SigLIP, not the intact VLM. Identity skip of whole SigLIP blocks, 512 frames, 2 noises, NFE=10, CARE-weighted L2 of the first 10 executed steps. Curves: `results/stage_nfe/vision_drop6_ledge.json`.

| order | σ | k_lo | k_hi | where r leaves σ | largest step |
|---|---:|---:|---:|---|---|
| Taylor 6d | 0.695 | 0 | 2 | r(1)=0.841 already above σ | k=3, L0 enters, 0.90→6.25 |
| Fisher 6d | 0.695 | 3 | 14 | r(3)=0.659, r(4)=0.986 | k=15, L10 enters, 4.71→6.38 |
| Taylor 7d | 0.714 | 0 | 1 | r(1)=0.873 already above σ | k=2, L0 enters, 0.87→6.31 |
| Fisher 7d | 0.714 | 3 | 14 | r(3)=0.671, r(4)=0.962 | k=15, L10 enters, 4.71→6.36 |

Taylor k_lo=0 is the empty-set fallback: no prefix stays under σ, because L0 is ranked near the front. Fisher k_lo=3 is the same early cut the language curves had; language closed-loop still held past that noise floor. These brackets do not set the vision k.
