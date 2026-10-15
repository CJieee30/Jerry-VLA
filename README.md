# Training-Free Efficiency for Flow-Matching VLA

**研究主线：系统性分析 VLM 层冗余，结合事件感知的离线 NFE 校准，在不做恢复训练的条件下压缩并加速 Flow-Matching VLA。**

更新：2026-10-14。本文依据 Jerry-VLA 提交 `eab85156dd53d89cc47e41e9f42676c9311eca61` 中的结果，以及最新确定的研究计划整理。服务器项目目录名是 `Skill_Efficient_VLA`，对应 GitHub 仓库是 [CJieee30/Jerry-VLA](https://github.com/CJieee30/Jerry-VLA)。本文介绍研究；新对话接手实验前，请读 [HANDOFF.md](HANDOFF.md)。

当前论文保留两项主贡献：**VLM backbone 的 no-recovery 层冗余分析与剪枝；CARE_event 支持的 action expert 少步推理诊断。** 面向相机角色的视觉 Token 剪枝 CROP 暂不列为主贡献。目标是形成有竞争力的顶会工作，近期优先补齐两条主线的核心证据。

## 1. 我们要解决什么问题

Flow-matching VLA 通常先编码图像和语言，再由动作专家通过多次速度场求值生成动作块。模型中有两类不同的压缩机会：

| 对象 | 研究问题 | 主要收益 | 主要风险 |
|---|---|---|---|
| VLM backbone 的层计算 | 哪些层的计算对动作功能并非必要？ | 减少参数、FLOPs 和前缀计算 | 表征相似的层可能仍承担关键功能 |
| Action expert 的迭代计算 | 默认求解步数能否减少？ | 减少重复调用，降低动作块推理延迟 | 平均动作很相似，但少数关键命令可能改变 |

我们的研究逻辑是：**不能只凭表征相似来删层，也不能只凭平均动作余弦来减少求解步数。需要分别检查功能保留情况，并通过闭环任务验证。** 两个组件互补：层剪枝主要压缩前缀计算和参数；少步推理主要减少动作专家的重复计算。

本文中的 training-free / no-recovery 指：在已有、面向 LIBERO 微调的检查点上压缩，不再更新权重，不进行恢复微调或蒸馏。计算梯度、使用示范校准集属于离线分析，因此这不是 data-free 方法，也不是从未经微调的基础模型直接部署。

## 2. 主贡献一：VLM 层冗余分析与 no-recovery 剪枝

我们对微调后的 flow-matching VLA backbone 进行系统性的层冗余研究，比较表征相似度、梯度相关重要性及位置删层等判据，检查它们能否预测无恢复剪枝后的闭环行为。现有实验表明，**representation similarity 不能直接替代 functional necessity**：在所研究的检查点和删层算子下，CosSim、CKA 可能选择到功能关键的层；Fisher、Taylor、GateProbe 等已有梯度判据在部分预算上更有效，但不存在跨模型、跨预算的单一冠军。

在 π0.5 和 MolmoAct2 上，我们分别找到 6/18 和 16/36 层的压缩配置，在已有 LIBERO 评估中保持较高成功率。这里的贡献是系统性的实验发现、对算子与判据关系的分析，以及可靠的无恢复压缩实践；Fisher/Taylor/GateProbe 本身不是本项目新发明的算法。“首个”主张尚需完整文献核查，暂不作为已确认事实。

### 2.1 为什么必须说明 keep-KV

这些模型的动作专家依赖 VLM 前缀产生的逐层 K/V。当前主路径保留被选层的 K/V 生成与缓存，让该层的残差更新恒等跳过；动作专家本身不删层。

因此，Drop-k 表示跳过 k 个 VLM block 的指定计算，**不代表彻底删除该层全部参数、移除全部 K/V，或复用前一层的 K/V**。即使连续层收到相同残差输入，各层的投影权重也不同，不能据此断定 K/V 相同。

实现有历史版本差异：π0.5 使用 DTR Appendix B prefix keep-KV 路径；MolmoAct2/SmolVLA 的早期 keep-KV 仍执行部分随后被丢弃的注意力计算，后续 K/V-only 进一步跳过 Q、attention、输出投影和 MLP。新版效率不能直接与旧版成功率拼接成同一实测配置。

### 2.2 已有层剪枝结果

以下均为 NFE=10、无恢复训练的本机 LIBERO 结果。SR 为成功率，n 为闭环回合数。

| 模型 | 配置 | 删除层数 | 完整模型 SR | 剪枝 SR | n/配置 |
|---|---|---:|---:|---:|---:|
| π0.5 | Fisher `[1,13,14,15,16,17]` | 6/18，33.3% | 97.25% | 97.75% | 400 |
| π0.5 | 同上，扩大评估 | 6/18 | 96.80% | 97.10% | 2000 |
| MolmoAct2 | Taylor `[1,2,3,4,24–35]` | 16/36，44.4% | 97.50% | 98.10% | 2000 |
| SmolVLA | Taylor `[12,13,14,15]` | 4/16 | 83.00% | 78.50% | 400 |

SmolVLA 是重要边界：当前 Drop-4 有明显成功率损失，不应写成三个模型都能无损剪掉三分之一层。小幅正负差异应结合逐任务结果和不确定性解释，不能默认都是噪声，也不能将较高点估计解释为剪枝提升能力。

已有核心对照足以支撑第一阶段的主要判断：

| 对照 | 结果 | 支持的判断 |
|---|---|---|
| π0.5 Drop-6：CosSim / CKA | 0.75% / 2.00%，各 400 回合 | 表征相似度不足以判断这种算子下的可删性 |
| π0.5 Last-6 / Fisher Drop-6 | 73.50% / 97.75%，各 400 回合 | 直接删末六层明显弱于经判据选层 |
| MolmoAct2 Last-16 / Taylor Drop-16 | 58.25% / 98.10%，各 2000 回合 | 同等删层数量下，层集合很重要 |
| MolmoAct2 Fisher / Taylor Drop-16 | 72.75% / 98.10%，各 2000 回合 | 梯度方法也有模型及预算依赖 |
| π0.5 同一 Drop-6，K/V 写零 / keep-KV | 26.25% / 97.75%，各 400 回合 | 保留条件 K/V 对该配置很重要；写零不是物理删除或跨层复用实验 |

Last-k 对照说明不能简单把尾层都当成冗余；它没有否定所有可学习或自适应早退方法。部分 MolmoAct2 CosSim/PPL 实验提前终止，不能把未执行的回合当成实测失败。

详细记录：[π0.5](functional_necessity/main_results/eval_pi05_libero_v044.md)、[MolmoAct2](functional_necessity/main_results/eval_molmoact2_libero.md)、[SmolVLA](functional_necessity/main_results/eval_smolvla_libero.md)。

### 2.3 语言模块的压缩收益

下表来自 [efficiency_analysis.md §3](functional_necessity/main_results/efficiency_analysis.md)，仅计算 language prefix，不含视觉编码与动作专家。参数指参与当前路径计算的 active parameters。

| 模型 | Active params（M） | 语言模块延迟（ms） | 语言模块 FLOPs（G） |
|---|---|---|---|
| π0.5，18→12 个残差更新层 | 2508.53→1854.21（−26.1%） | 78.10→53.43（−31.6%） | 3974.95→2662.15（−33.0%） |
| MolmoAct2，36→20 个残差更新层 | 4029.68→2498.71（−38.0%） | 88.68→71.01（−19.9%） | 3820.07→2206.99（−42.2%） |

这些不是整模型参数或整段推理时间的降幅。只装 skip hook 不会自动减小存储权重；项目另有 materialized pruning 实现用于实际释放权重。MolmoAct2 本表使用新版 K/V-only，已观察到相对旧算子的 bf16 动作数值差异，闭环结果与具体实现版本仍需对齐。

## 3. 主贡献二：Event-Aware Calibration for Few-Step Flow Matching

**中文：面向少步流匹配的事件感知校准；代码名称：CARE_event。**

模块诊断发现，参数较少的动作专家可能占据主要延迟：在本机 RTX 3090、eager、NFE=10 的一次模块测量中，π0.5 和 MolmoAct2 的动作专家分别约占三模块时间之和的 74% 和 90%。多次调用及运行时开销使参数量、FLOPs 与延迟并不成比例。因此，仅压缩 VLM 不一定能显著降低整个动作块的生成延迟。

我们提出用动作事件引导离线校准：优先选择运动命令变化较大、夹爪开合决策附近的观测，在相同输入与噪声下比较少步和默认 NFE 输出，检查手臂误差、夹爪命令时机及误差尾部，为固定 NFE 配置提供依据。部署时不逐块计算 CARE 分数，不进行额外在线路由，也不改变已选 NFE 内的 Euler 更新公式。

**当前已完成选样分析、离线探针、独立回合验证和等预算四组对照；推荐规则已有 v0 草案和开发演练（[协议](functional_necessity/results/registry/care_event_nfe_protocol.md)），尚未冻结，也尚未完成“先冻结推荐规则，再预测未知验证结果”的完整证据链。** 已有 π0.5/MolmoAct2 一步、SmolVLA 两步闭环结果早于 CARE_event，不能倒过来宣称它们是由 CARE_event 自动发现的。

### 3.1 方法与公式

对示范轨迹中的观测时刻 τ，取从 τ 开始的 H=10 步有效动作命令。动作有七个有效维度。用固定的逐维动作差分标准差标准化后，计算平均变化强度：

$$
d_\tau=\frac{1}{m_\tau-1}\sum_{j=\tau}^{\tau+m_\tau-2}
\left\|(a_{j+1}-a_j)\oslash\sigma_{\Delta a}\right\|_2,
\qquad \alpha_\tau=\frac{d_\tau}{\operatorname{mean}_{\mathrm{cal}}(d)}.
$$

若窗口内原始夹爪命令的开合状态 `g > 0` 发生切换，记 Gτ=1，否则为 0。事件增强分数为：

$$
\alpha_{\mathrm{event},\tau}=
\frac{\alpha_\tau+\lambda G_\tau}
{\operatorname{mean}_{\mathrm{cal}}(\alpha+\lambda G)},\qquad \lambda=1.
$$

当前每任务从 12 个候选回合的观测中选前三名，共 40×3=120 个观测，同时保留最高与较高分样本。分数用于选样，不用于训练或损失重加权。归一化统计来自校准数据，独立验证时固定使用。

评分针对动作命令，探针从同一观测 τ 开始。实际夹爪张开、物品掉落可能稍后显现，但不因此把探针起点后移：我们要检查模型能否在合适的位置和时机发出命令。低分表示命令较平滑，不等于任务不重要。这里的执行时刻 τ、执行窗口 H 和 flow 求解时间/NFE 是不同概念。

### 3.2 现有等预算选样结果

原候选池 480 个观测，每任务 12 个不同 episode；各策略选 120 个观测，探针使用两组初始噪声。原始 α 与事件分数的计算不使用模型的低 NFE 误差。

| 指标 | 随机选样 | 原始 CARE α | CARE_event |
|---|---:|---:|---:|
| 示范夹爪事件窗口覆盖 | 19/56 | 34/56 | 55/56 |
| SmolVLA NFE=1 夹爪偏差捕获 | 8/15 | 11/15 | 14/15 |
| π0.5 NFE=1 夹爪偏差捕获 | 5/10 | 7/10 | 10/10 |
| SmolVLA NFE=1 整体尾部偏差捕获 | 26/73 | 36/73 | 44/73 |
| π0.5 NFE=1 整体尾部偏差捕获 | 23/71 | 38/71 | 38/71 |

后四行合并两组噪声的比较数，不是独立回合数。结果支持事件加分对夹爪偏差的诊断价值；π0.5 的整体尾部捕获没有优于原始 α，手臂误差覆盖也存在预算取舍。当前随机组只有一个选样种子，且这些比较复用了已探测的全部 480 个观测，属于回放分析。

### 3.3 独立回合 holdout

另取 120 个新 episode 的观测，与原 480 个候选涉及的 episode 无交集。每观测两组噪声，每个 NFE 有 240 次动作比较。比较的是输出前 10×7 个动作，以 NFE=10 为模型参考。

| 模型 | NFE | 平均动作余弦 | 平均 RMSE | 尾部偏差数 | 夹爪命令不一致数 |
|---|---:|---:|---:|---:|---:|
| π0.5 | 1 | 0.99782 | 0.0560 | 13/240 | 0/240 |
| π0.5 | 2 | 0.99890 | 0.0395 | 3/240 | 0/240 |
| SmolVLA | 1 | 0.99342 | 0.0942 | 22/240 | 2/240 |
| SmolVLA | 2 | 0.99747 | 0.0568 | 2/240 | 2/240 |
| SmolVLA | 3 | 0.99849 | 0.0420 | 2/240 | 2/240 |

RMSE 在策略归一化空间中计算；尾部界限来自各模型自身的原校准 NFE=1 分布，不是任务失败阈值。SmolVLA 从一步到两步明显减少整体偏差，但两例夹爪命令时机偏差在两步、三步中仍存在。因此，平均余弦高不能证明安全，增加一步也不保证修复每个样本。

详见 [CARE_event 完整研究记录](functional_necessity/main_results/care_event_action_expert_nfe.md)及 [holdout 报告](functional_necessity/results/stage2_care_v2/nfe_probe_care_event_holdout/REPORT.md)。这里的独立仅指 episode 隔离，不代表模型训练时未见过该示范数据集。

## 4. 已有少步闭环与联合压缩结果

下表汇总已有闭环事实，不归因为 CARE_event。保留不同评估规模和复跑，避免选择性报告。

| 模型 / n | Full，NFE=10 | 仅层剪枝，NFE=10 | Full，低 NFE | 层剪枝＋低 NFE |
|---|---:|---:|---:|---:|
| π0.5 / 400 | 97.25% | 97.75% | 97.00%（1） | 96.50%（1） |
| π0.5 / 2000 | 96.80% | 97.10% | 尚未列入已核验结果 | 95.70%；复跑 96.25%（1） |
| MolmoAct2 / 2000 | 97.50% | 98.10% | 97.80%（1） | 97.60%（1） |
| SmolVLA / 400 | 83.00% | 78.50% | 82.00%（2） | 77.00%（2） |

括号为低 NFE 数值。MolmoAct2 另有早期 400 回合的 Full NFE=1 98.25%、Drop-16 NFE=1 99.25%，它们与 2000 回合结果是不同评估，不能互相替换。SmolVLA Full NFE=1 为 79.25%，另一次 paired 复跑为 78.25%，支持优先考察两步而非直接固定一步。

本次文档核查从 MolmoAct2 原始 JSON 复算出 Full NFE=1 **1956/2000**、Drop-16 NFE=1 **1952/2000**，两者均为 40 任务×50 回合；旧 `eval_nfe_libero.md` 的“未跑 Molmo 2000 集 NFE=1”已经过时。来源：[Full JSON](functional_necessity/results/stage_nfe/molmoact2_full_nfe1_2000/eval_info.json)、[Drop-16 JSON](functional_necessity/results/stage_nfe/molmoact2_drop16_taylor_nfe1_2000/eval_info.json)。元数据记录 keep-KV；与新版 K/V-only/materialized 实现的对应关系仍需追溯，不能仅凭目录名认定相同。

### 4.1 推理效率的已有证据与边界

下面是 RTX 3090 上已有的单输入配对 eager 诊断，计时范围为一次 `predict_action_chunk`。它展示互补趋势，不代替最终多输入、独占卡测量。

| 模型 | Full NFE=10 | 仅剪层 NFE=10 | Full 低 NFE | 剪层＋低 NFE |
|---|---:|---:|---:|---:|
| π0.5，低 NFE=1 | 524.3 ms | 502.0 ms | 181.1 ms | 155.9 ms |
| MolmoAct2，低 NFE=1，旧 keep-KV | 1166.6 ms | 1153.0 ms | 253.4 ms | 240.4 ms |
| SmolVLA，低 NFE=2，旧 keep-KV | 556.3 ms | 573.0 ms | 173.8 ms | 181.8 ms |

例如 π0.5 联合配置相对 Full NFE=10 的 chunk 时间减少约 70.3%、FLOPs 减少约 33.1%；其中大部分时间收益来自 NFE，不能全部算作层剪枝收益。SmolVLA 的层剪枝在该轮同 NFE 测量中没有加速。

运行时优化会显著改变倍数。另一轮 MolmoAct2 实体化测量中，Full NFE=10→1 的 eager 时间为 1392.8→309.6 ms；启用模型现有 expert CUDA graph 后为 273.9→183.6 ms。相应 Graph 联合配置为 171.5 ms。上述数字各自来自同一轮配对，不能跨轮相减。Graph 代表样本动作误差为 0，但尚不能代替所有输入与闭环的等价性验证。

因此，最终主表必须包含合理运行时基线，不能只选择有大量 eager 启动开销的配置来宣传最高加速倍数。详见 [效率记录 §4–§6](functional_necessity/main_results/efficiency_analysis.md)。

## 5. 泛化边界与暂缓方向

**分布内保持成功率不等于分布外无损。** 已完成的六个 LIBERO-plus 维度上，π0.5 Full NFE=10 与 Drop-6 NFE=10 的维度宏平均为 85.89% 与 81.65%，下降 4.24 个百分点；不含未完成的 Sensor Noise。Drop-6 的 NFE=1 已完成 Language / Init / Camera / Layout，分别为 84.67 / 74.47 / 61.24 / 80.98。Light/Background 的 NFE=1 在快照中仍为进行中，实际状态需查服务器。详见 [LIBERO-plus](functional_necessity/main_results/eval_libero_plus.md)。

- **CROP：**面向腕部相机的任务相关信息和全局相机的多样性信息进行视觉 Token 保留。已有实现与闭环，但选样、scout 与 prefix 处理的开销限制了收益，目前不作为主贡献。详见 [CROP](functional_necessity/main_results/crop_libero.md)。
- **早期 layer-pruning CARE：**研究动作变化加权能否改善层排序，有局部收益与不稳定结果；不能作为当前 NFE CARE_event 已成功的证据。详见 [CARE 层剪枝记录](functional_necessity/main_results/care_libero.md)。
- **LEDGE、任务条件层选择、进一步 KV 复用：**保留历史记录，暂不扩大实验。
- **跨设备、RoboTwin、真机：**属于扩展与边界验证，不在当前一个月的最小必做清单中；另一台服务器结果独立呈现。

## 6. 接下来只补五项核心证据

| 优先事项 | 具体目标 | 当前状态 |
|---|---|---|
| 1. 第一贡献结果规范化 | 对齐 checkpoint、算子版本、NFE、回合数、种子、层集合与效率范围；复用可比结果 | **完成（CPU）**：[结果清单](functional_necessity/results/registry/core_results_registry.md)；Molmo 历史算子版本仅为推断 |
| 2. CARE_event 最小对照 | 同为每任务 3 个观测，比较随机、α、G-only、α_event；重复随机选样 | **完成（CPU）**：四组＋20 种子，见 `nfe_probe_event_care/budget_compare/` |
| 3. 冻结规则的前瞻验证 | 在新验证结果揭晓前保存 NFE 推荐，再验证；补 MolmoAct2 探针 | **进行中**：[v0 协议草案](functional_necessity/results/registry/care_event_nfe_protocol.md)、分阶段入口和开发演练已完成，待审阅后冻结；新验证、新闭环和 Molmo 探针均未启动 |
| 4. 联合配置验证 | 检查剪枝后低 NFE 风险，补最终实现对应的闭环缺口 | 历史四格已有大量结果，先核查再补跑 |
| 5. 正式效率测量 | 独占 3090，多输入配对，报告 median/P95、FLOPs、实际参数与显存；纳入可用 Graph 基线 | 现有诊断不足以替代最终测量 |

不要求把所有历史 400 回合实验重跑成 2000；统一协议与标注，比统一所有样本量更重要。新关键闭环至少每任务 10 回合，优先复用完整且配置一致的结果。

我们希望最终证明的是：**针对 VLM 深度与动作专家求解步数这两类冗余，采用功能导向的离线分析，可以获得可验证的无恢复压缩与效率收益；事件感知校准能让有限探针预算更关注容易被平均指标掩盖的动作偏差。** 其中层剪枝证据较充分，CARE_event 的前瞻推荐能力及最终部署收益仍需完成上述验证。

## 7. 阅读与代码入口

| 内容 | 入口 |
|---|---|
| 新对话任务、下一步、运行约束与坑 | [HANDOFF.md](HANDOFF.md) |
| 论文结果与原始实验索引 | [functional_necessity/main_results/](functional_necessity/main_results/) |
| π0.5 层评分、skip、闭环 | [scripts/pi05_libero/](functional_necessity/scripts/pi05_libero/) |
| MolmoAct2 对应实现 | [scripts/molmoact2_libero/](functional_necessity/scripts/molmoact2_libero/) |
| SmolVLA 对应实现 | [scripts/smolvla_libero/](functional_necessity/scripts/smolvla_libero/) |
| CARE_event 选样与 NFE 探针 | [scripts/stage2_care_v2/](functional_necessity/scripts/stage2_care_v2/) |
| NFE、实体化剪枝与部署测量 | [scripts/stage_nfe/](functional_necessity/scripts/stage_nfe/) |

`paper/` 中的“层剪枝→CARE→LEDGE 三阶段”是历史组织方式，不能作为当前论文贡献顺序。最新叙事和工作优先级以本文与 HANDOFF 为准；具体数值仍回溯原始结果、配置及相应源码版本。
