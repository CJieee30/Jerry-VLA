# HANDOFF：Flow-Matching VLA 的无恢复压缩与离线少步校准

> 写给完全没有前文的新对话。更新：2026-10-14。  
> 数据核查快照：Jerry-VLA `eab85156dd53d89cc47e41e9f42676c9311eca61`。  
> 服务器项目根目录：`/home/guorj/Skill_Efficient_VLA`。本文路径未特别说明时均相对此目录。  
> 本次更新只整理文档、核查已有代码与结果，没有启动新 GPU 实验。

## 0. 接手先读这一页

我们准备一篇面向顶会的 **training-free / no-recovery Flow-Matching VLA efficiency** 论文。用户希望近期一个月集中完成必要证据，避免扩展成庞大的机制分析或 benchmark 项目。论文当前确定两条主线：

1. **系统性 VLM 层冗余分析与无恢复剪枝。** 在 π0.5、MolmoAct2 等微调检查点上比较表征相似度与梯度判据；说明 representation similarity 不能直接代表 functional necessity，找到有效的 keep-KV 层压缩配置。
2. **CARE_event：Event-Aware Calibration for Few-Step Flow Matching。** 用动作命令变化和夹爪命令事件选择离线校准观测，检查低 NFE 对关键动作的影响，为部署时固定 NFE 提供诊断依据。

**进度判断：第一条已有较充分实验；第二条已有选样、探针和独立回合结果，但“未知验证结果前能否给出有效 NFE 建议”仍是最重要缺口。** 旧有低 NFE 闭环先于 CARE_event，不能归功于本轮校准方法。

**进度（2026-10-15）：**

- **已完成（CPU）**：结果规范化（`results/registry/core_results_registry.md`）；G-only 和 20 种子随机对照。
- **已有草案，未冻结**：NFE 协议 v0 文档（`results/registry/care_event_nfe_protocol.md`）；实现以 **`nfe_protocol_v1.py` + `nfe_en_metrics.py`** 为准（主规则 E_N≤0.085，两改善规则独立对照）。
- **冻结前收口（2026-10-15 CPU）**：`dev_nfe_rules_cpu_v2/`（`three_rule_budget_table.csv`、`PRE_FREEZE_REPORT.md`、`FREEZE_BLOCKERS.md`）；E2E `nfe_protocol/dev_e2e_three_rules_v8/`；前瞻模板 `nfe_protocol_config.prospective_prefreeze.json`。
- **下一步**：顾问审阅后执行 `prospective_prefreeze_v1` 的 `freeze`（本轮未做）；再 split / 前瞻探针 / 闭环导入。

不要先重跑整套层剪枝，不要开始训练。

以下旧指令已被用户最新决定取代：只保留一个贡献、效率只能作支撑、所有 CARE 都只能是负结果、永远不考虑 CUDA graph、等待所有 GPU 全空才能做闭环、所有实验必须 2000 回合。旧 `paper/` 文档与历史结果说明可能仍含这些说法，勿据此恢复旧路线。

## 1. 研究目标、范围与表达边界

### 1.1 论文要讲的故事

Flow-matching VLA 中，VLM 层计算与动作专家多步求解是两种不同的冗余。层剪枝能减少参数和前缀 FLOPs，但整段延迟可能仍受反复调用的动作专家限制。我们先研究什么层计算可以省，再研究如何用有限离线探针检查求解步数能否减少，最后验证二者结合后的任务性能与部署收益。

这是“系统性分析＋功能导向的无恢复效率优化”。不必临时改成深入因果机制论文，也不必发明新的 Fisher/Taylor 才能推进；但 CARE_event 要有超出事后解释的诊断价值，效率要使用可信基线。

### 1.2 定义及不能夸大的内容

- Training-free / no-recovery：压缩期间不更新已微调模型权重、不恢复微调、不蒸馏。允许离线梯度计算与校准数据；不是 data-free。
- Fisher、Taylor、GateProbe 是已有判据，不称为本项目新算法；暂不使用未经完整查新的“首个”。
- π0.5 删除 6/18 是恰好三分之一；MolmoAct2 删除 16/36 是 44.4%。这是指定算子下被跳过的 block 计算，不是这些 block 的全部权重与 KV 都消失。
- representation similarity 不等于 functional necessity 的结论限定于实测检查点、数据和算子，不推广成“所有 VLA 的 CosSim 都无效”。
- Last-k 对照表现差，不等于所有动态/可学习早退都不可行；L0 敏感也是当前模型的经验观察，不是普遍定理。
- CARE_event 分数衡量示范命令变化与事件，不是重要性的真值，也没有证明它导致 flow 轨迹弯曲。
- NFE=10 是参考模型输出，不是真实正确动作或闭环成功的 oracle。
- 参数量、FLOPs、语言模块时间、完整 chunk 时间和机器人任务完成时间必须分开。

视觉 Token 剪枝 CROP 暂不列入主贡献。早期 layer-pruning CARE、LEDGE、任务条件删层、视觉层剪枝与进一步 KV 复用暂不增加工作量。真机、第二设备、新 benchmark 不是当前最小必做项。

## 2. 已完成什么：可复用的结果

### 2.1 VLM 层分析

完整记录：[π0.5](functional_necessity/main_results/eval_pi05_libero_v044.md)、[MolmoAct2](functional_necessity/main_results/eval_molmoact2_libero.md)、[SmolVLA](functional_necessity/main_results/eval_smolvla_libero.md)。

| 模型 | 主配置，层编号从 0 开始 | NFE=10 完整→剪枝 SR | 评估规模 |
|---|---|---|---|
| π0.5 | Fisher Drop-6 `[1,13,14,15,16,17]` | 97.25→97.75%；扩大评估 96.80→97.10% | 前者各 400，后者各 2000 |
| MolmoAct2 | Taylor Drop-16 `[1,2,3,4,24,25,26,27,28,29,30,31,32,33,34,35]` | 97.50→98.10% | 各 2000 |
| SmolVLA | Taylor Drop-4 `[12,13,14,15]` | 83.00→78.50% | 各 400；是压缩边界，不是无损成功案例 |

重要对照已存在，不需要再做一轮小规模功能干预：

- π0.5 Drop-6：CosSim 0.75%、CKA 2.00%、Last-6 73.50%、Fisher 97.75%；Taylor 原生含 L0 为 0%，明确保护 L0 的变体为 97.25%。不得把保护 L0 后的结果标为原生 Taylor。
- MolmoAct2：Last-16 58.25%、Fisher Drop-16 72.75%、Taylor Drop-16 98.10%；Taylor Drop-17 为 89.60%。没有跨预算的统一最优判据。
- MolmoAct2 Drop-8 的 Taylor/Fisher/IGIA/GateProbe 选出相同集合 `[28–35]`，共享一轮 2000 回合的 97.75%；不是四轮独立验证。
- π0.5 Drop-6 将 K/V 写零为 26.25%，keep-KV 为 97.75%，各 400 回合。该实验既不是物理删除 KV，也不是复用上一层 KV。
- 末层 keep-KV 的 32 帧开环检查中，π0.5 L17、SmolVLA L15 在 NFE=1/10 的动作差异为 0；MolmoAct2 有小数值差异。不要扩展成所有末层 KV 可删或恒等的证明。
- 早期 layer-pruning CARE 存在局部收益、集合不变与不稳定结果，不能概括为全无作用，也不能作为新 NFE 方法的成功实验。

部分低成功率方法提前终止。论文规范化时按实际完成回合与 `stopped` 状态标记；不要沿用旧文档把未运行部分补零的规则，也不必为了表格整齐补跑明显失败配置。

### 2.2 少步及联合闭环

| 模型 | 配置 | NFE | SR | n |
|---|---|---:|---:|---:|
| π0.5 | Full | 10 / 1 | 97.25 / 97.00% | 各 400 |
| π0.5 | Drop-6 | 10 / 1 | 97.75 / 96.50% | 各 400 |
| π0.5 | Full | 10 | 96.80% | 2000 |
| π0.5 | Drop-6 | 10 | 97.10% | 2000 |
| π0.5 | Drop-6 | 1 | 95.70%；同配置另一次复跑 96.25% | 各 2000 |
| MolmoAct2 | Full | 10 / 1 | 97.50 / 97.80% | 各 2000 |
| MolmoAct2 | Drop-16 | 10 / 1 | 98.10 / 97.60% | 各 2000 |
| MolmoAct2 | Full / Drop-16 | 1 | 98.25 / 99.25% | 各 400，早期实验 |
| SmolVLA | Full | 10 / 1 / 2 / 3 | 83.00 / 79.25 / 82.00 / 81.00% | 各 400 |
| SmolVLA | Drop-4 | 10 / 1 / 2 / 3 | 78.50 / 73.00 / 77.00 / 76.00% | 各 400 |

SmolVLA 另一次 paired NFE=1 复跑：Full 78.25%、Drop-4 70.25%。π0.5 的两轮 2000 回合 NFE=1 使用相同 seed=1000，不应包装为两个独立随机种子，不能仅挑较高的一轮。

**本次补核实：**MolmoAct2 两份 NFE=1 2000 回合 JSON 已齐，逐任务 `successes` 复算为 Full 1956/2000、Drop-16 1952/2000，每个任务 50 回合：

- [Full NFE=1 原始结果](functional_necessity/results/stage_nfe/molmoact2_full_nfe1_2000/eval_info.json)，四套件 97.4 / 99.4 / 97.4 / 97.0%。
- [Drop-16 NFE=1 原始结果](functional_necessity/results/stage_nfe/molmoact2_drop16_taylor_nfe1_2000/eval_info.json)，四套件 97.8 / 98.6 / 96.8 / 97.2%。
- 同目录 `drop.json` 记录 `operator=keep_kv_identity_residual`、NFE=1、每任务 50；它没有充分锁定源码版本。先追溯该运行究竟使用哪版 skip，再决定是否需要新版闭环。

`main_results/eval_nfe_libero.md` 仍说“未跑 Molmo 2000 集 NFE=1”，属于汇总滞后。不要重复安排这两轮历史实验；先补写结果及元数据。其余 NFE 背景见 [eval_nfe_libero.md](functional_necessity/main_results/eval_nfe_libero.md)。

以上实验支持低 NFE 的可行性，但不足以宣称所有联合配置“成功率完全不变”。尤其 π0.5 联合 2000 回合存在下降，SmolVLA 层剪枝损失明确。已有结果不是 CARE_event 前瞻推荐实验。

### 2.3 效率与实现版本

详见 [efficiency_analysis.md](functional_necessity/main_results/efficiency_analysis.md)。

| 结果 | 已有证据 | 使用限制 |
|---|---|---|
| π0.5 language prefix | Active params −26.1%，时间 −31.6%，FLOPs −33.0% | 模块口径，不是整模型降幅 |
| MolmoAct2 language prefix，新 K/V-only | Active params −38.0%，时间 −19.9%，FLOPs −42.2% | 与旧 keep-KV SR 的实现对应关系待核实 |
| π0.5 单输入 eager 配对 | Full10 / Drop10 / Full1 / Drop1：524.3 / 502.0 / 181.1 / 155.9 ms | 诊断；不能替代多输入正式测速 |
| MolmoAct2 旧 keep-KV 配对 | 1166.6 / 1153.0 / 253.4 / 240.4 ms | 与新版 K/V-only 和其他会话分开 |
| MolmoAct2 expert Graph 配对 | Full10 / Drop10 / Full1 / Drop1：273.9 / 254.0 / 183.6 / 171.5 ms | 已有可用运行时基线，不能从效率讨论中删去 |

必须区分三层实现：

1. **历史 keep-KV skip：**残差不更新，但 MolmoAct2/SmolVLA 可能仍算 Q/attention 后丢弃；旧 SR 与 `efficiency_analysis.md §2/§4` 属于对应历史路径。
2. **K/V-only skip：**保留必要 norm、K/V、位置编码和缓存，跳过 Q/attention/输出投影/MLP。MolmoAct2 从 fused projection 改为 K/V 切片投影后有 bf16 差异，代表帧动作 max-abs 约 0.0078，不能直接认定闭环等价。§3/§5 使用此类测量。
3. **Materialized pruning：**真正释放未使用的权重，而非只装 hook。§6 与 `scripts/stage_nfe/materialize.py`、`measure_deployment.py` 对应；stored/active 参数和 allocated/peak 显存分别报告。

`torch.compile` 与 expert CUDA graph 也不是同一种设置。历史 compile 可能导致动作漂移、绕过动态 hook、触发重编译；Molmo expert Graph 在代表样本上动作差异为 0。新主表应纳入可用 Graph 对照，不要求先开展大规模编译工程。改变运行时后先检查实际调用与动作一致性，再确定可复用哪些 SR。

### 2.4 泛化与旁支现状

- LIBERO-plus：六个完整维度的 Full10 / Drop-6 NFE10 宏平均为 85.89 / 81.65%，差 −4.24 pp；Sensor Noise 只有 Spatial/Object，不能凑成七维完整平均。
- Drop-6 NFE=1 的 Language / Init / Camera / Layout 已完成，分别 84.67 / 74.47 / 61.24 / 80.98%。Light/Background 在本快照的文档中标记进行中，需查服务器，勿从本文推断此刻卡是否空闲。
- [eval_libero_plus.md](functional_necessity/main_results/eval_libero_plus.md) 有把分布内 π0.5 Drop-6 NFE=1 写成 **76.00%** 的错误；400 回合正确值是 **96.50%**，2000 回合为 **95.70 / 96.25%**。规范化时修正文案，不改原始 JSON。
- CROP：Pi Drop-6、NFE=1 的 role / shared-mix 为 95.75 / 96.00%，对应 baseline 96.50%，各 400。后续 NFE=10 raw / trim-only / trim-grid 为 97.75 / 96.50 / 94.75%，各 400。当前不足以作为主贡献，见 [crop_libero.md](functional_necessity/main_results/crop_libero.md)。
- `another_server_results`、RoboTwin、OFT、CogACT、DTR recovered checkpoint 各有不同机器或协议，不能混入本机三条主栈的主表。已有 OOD 损失是需要保留的研究边界。

## 3. CARE_event：接手必须理解的方法与数据

完整介绍与公式见 [care_event_action_expert_nfe.md](functional_necessity/main_results/care_event_action_expert_nfe.md)。以下足以开始下一阶段，不依赖旧对话。

### 3.1 当前分数固定定义

对示范观测 τ，取同一 episode 内从 τ 开始的最长 H=10 个七维动作。对相邻差分使用固定 `sigma_delta_a` 标准化，平均其 L2 范数得到 d，再除以校准均值得到 α。完整窗口计算九个相邻差分。

```text
z_j = (a[j+1] - a[j]) / sigma_delta_a
d_tau = mean_j ||z_j||_2
alpha_tau = d_tau / 2.0472331711711984
G_tau = 窗口内原始夹爪命令 (g > 0) 是否发生任意开合切换
alpha_event = (alpha_tau + G_tau) / 1.1166666665580123
H = 10，lambda = 1
```

`sigma_delta_a` 以 `dump_shared.py` 和现有 `shared.pt` 为准；原始 v2 固定值为：

```text
[0.05375851375258111, 0.045300022456052406, 0.06353572610428114,
 0.007690927155167576, 0.012292487402144343, 0.008634497491330419,
 0.2400744514055428]
```

G 同时含开与合，仅检查窗口内部，不含窗口前一帧到窗口起点的切换。历史候选可能有尾端短窗口，当前 holdout 全为完整 H=10。不要对 holdout 重估均值或差分标准差，也不要与早期 layer-pruning CARE 的归一化口径混用。

视频金色窗口是评分动作命令窗口；探针输入就是窗口起点 τ 的观测。高分常对应夹爪命令决策，实际机械响应或物品落下可能出现在窗口之后。用户审阅后认可当前事件定义，**不要整体后移时间戳，不要擅自提高夹爪权重或改公式**。物理延迟未由关节状态定量测定，不能写成确定的 3–10 帧常数。较高 α 的对齐/调整与最高 α_event 的开合决策都应被覆盖。

### 3.2 已完成的选样和探针

- 原候选池：40 任务×12 不同 episode，每 episode 一个观测，共 480。
- 等预算选样：每任务三个观测，共 120；现有 random(seed42)、α top3、α_event top3。
- 两个初始噪声，各 NFE 匹配同一噪声、观测、语言与状态；模型 reset 后执行。
- 当前 CARE 探针使用完整模型，没有 Drop-6/Drop-16 或 Token 剪枝。π0.5 NFE={1,2,10}，SmolVLA={1,2,3,10}。
- 比较前 10 个执行动作×7 个有效维度；Pi/Smol 输出 chunk=50，不能把未执行的后 40 步或 padded 32 维算进主指标。
- RMSE/余弦基于 policy-normalized MEAN_STD 动作；夹爪符号先反标准化再判断原始 `g > 0`，不能对归一化动作直接取符号。
- 当前两噪声种子规则：`1000 * 1000003 + sample_index * 2 + noise_id`。样本 index 必须和 manifest 对应。
- 代表样本 NFE=10 与官方路径 max-abs=0、reset 重复一致；CSV 的调用次数与 NFE 一致。代表样本检查不等于所有输入都重跑过官方对照。

现有同预算结果：

| 指标，合并两噪声时不是独立回合数 | Random | α | α_event |
|---|---:|---:|---:|
| 56 个示范夹爪事件窗口覆盖 | 19 | 34 | 55 |
| SmolVLA NFE1 夹爪偏差捕获 | 8/15 | 11/15 | 14/15 |
| π0.5 NFE1 夹爪偏差捕获 | 5/10 | 7/10 | 10/10 |
| SmolVLA NFE1 整体尾部捕获 | 26/73 | 36/73 | 44/73 |
| π0.5 NFE1 整体尾部捕获 | 23/71 | 38/71 | 38/71 |

分数中本来就含 G，事件覆盖提高不是充分贡献；应重点考察模型动作偏差捕获及前瞻建议价值。事件项也会挤占手臂变化样本的预算，不应宣称各指标全面优于原 α。

### 3.3 已完成的独立回合验证

目录：[nfe_probe_care_event_holdout/](functional_necessity/results/stage2_care_v2/nfe_probe_care_event_holdout/)。seed4242 排除原 480 候选涉及的全部 episode，每任务三个新 episode，各取一个完整窗口，总计 120 个观测、16 个示范夹爪事件窗口。每个 NFE 比较 240 次，来自 120 个 episode，不是 240 个独立 episode。

| 模型 | NFE | 平均余弦 / RMSE | 尾部偏差 | 夹爪不一致 |
|---|---:|---|---:|---:|
| π0.5 | 1 | 0.99782 / 0.0560 | 13/240 | 0/240 |
| π0.5 | 2 | 0.99890 / 0.0395 | 3/240 | 0/240 |
| SmolVLA | 1 | 0.99342 / 0.0942 | 22/240 | 2/240 |
| SmolVLA | 2 | 0.99747 / 0.0568 | 2/240 | 2/240 |
| SmolVLA | 3 | 0.99849 / 0.0420 | 2/240 | 2/240 |

尾部规则：`cosine < cal_NFE1_P5 OR RMSE > cal_NFE1_P95 OR gripper_disagree_n > 0`，按**对应模型、对应噪声**的原 480 候选 NFE1 结果定阈值。它只是描述性异常标记，**不是可直接上线的 NFE 接受规则**。分模型阈值也不适合直接比较跨模型“安全度”。

两例 SmolVLA 夹爪偏差：holdout #16/noise0 提前一步发出松爪命令；#48/noise1 延迟一步闭合；NFE=1/2/3 都存在。π0.5 #7 是 G=0 的手臂偏差，#33 的较低事件分数也未排除尾部风险。支持同时看运动误差和事件，但未证明这些偏差必然导致任务失败。

原 holdout 每任务只有三个候选，同时选择预算也是三个，所以 `selected_*` 全为 1；这里不能比较选样策略，也不能把 100% 覆盖当作 CARE 优势。独立回合验证不是未见任务或新模型验证，更不是尚未看结果的前瞻推荐。

### 3.4 修复记录与剩余小问题

- 已修复：π0.5 误用 SmolVLA 尾部阈值。正确 Pi NFE1 为 7/120 + 6/120，即 13/240，不再使用旧的 2/240。
- 已修复：异常图混入普通样本；现按真实 tail 筛选，SmolVLA 包含 NFE3，Pi 有独立报告。来源见同目录 `FIXES.md`、`summary.json`、各模型 CSV。
- 仍存在工程风险：`holdout_validation.py::archive_stale_tail_curves` 若同名备份已存在，会先 `unlink()` 再覆盖。再次重绘前改成时间戳目录或不覆盖备份；这是文件归档问题，不是上述数值失效的证据。
- 跨阶段定位样本用 `global_index` 或 dataset/episode/frame；不要把各目录局部的 `shared_idx` 当成全局身份。

## 4. 下一步：只保留五项核心任务

以下是已确定的最小方案，不能提前写进论文为已完成。

当前状态：

| 任务 | 状态 |
|---|---|
| 1 结果规范化 | 完成（CPU） |
| 2 等预算对照 | 完成（CPU） |
| 3 前瞻验证 | 进行中：v0 协议与开发演练已完成，待审阅后冻结；未生成新模型输出 |
| 4 联合配置闭环 | 未启动 |
| 5 正式效率 | 未启动 |

| 顺序 | 任务与对应贡献 | 最小做法 | 验收/交付 |
|---|---|---|---|
| 1 | 第一贡献结果规范化 | 汇总已有主配置与核心对照；记录检查点、源码/算子版本、层集、L0 保护、NFE、回合数、种子、运行时、原始路径 | 一个可追溯的结果清单；修正 Molmo NFE1 2000 遗漏与 Plus 76.00 误写；列出真正缺失的配对 |
| 2 | CARE_event 等预算最小对照 | 复用已有 480 候选与探针；各 120、每任务 3 个；补 G-only，随机选样重复约 20 个种子 | Random/α/G-only/α_event 的夹爪偏差、整体尾部及手臂偏差捕获；随机均值和离散程度；通常无需新 GPU 推理 |
| 3，核心 | 冻结规则的前瞻 NFE 验证 | 旧结果只作开发；先固定规则，再用新候选 episode 校准、保存建议，之后看独立验证与闭环；覆盖 Pi/Smol，补 Molmo 探针 | 可审计的配置、数据分割、推荐及时间记录；允许拒绝低 NFE；证明建议是否与后续验证一致 |
| 4 | 最终联合配置闭环 | Pi/Molmo 做 Full/Pruned × NFE10/低 NFE；先检查剪枝后选中样本的动作偏差；核查已有结果后只补缺口 | 最终代码版本与 SR 对应；每个新增关键配置至少 40×10=400 回合；确有需要才扩大关键项 |
| 5 | 正式效率 | 独占 RTX 3090，多输入配对测四格；纳入已可用的 Molmo expert Graph 基线 | chunk median/P95、FLOPs、stored/active 参数、allocated/peak 显存、配置与原始日志；离线校准成本另报 |

### 4.1 不要求把 400 全部改成 2000

π0.5 的方法比较已有 400 回合、关键配置已有 2000；MolmoAct2 大部分已有 2000。两种规模都可以在论文中使用，只要每行明确 n，核心对比的环境、初始状态/种子、动作执行、NFE、模型和算子一致，并报告合适的不确定性。

不要靠统一显示位数来假装精度相同。需要同规模时优先补少量关键格；只有能核对同一预定义评估子集与初始状态，才考虑从现有逐回合记录取配对子集，不能按结果挑回合。统计时保留每任务结果，优先按任务及配对回合处理相关性；不把同一输入的两组噪声当独立闭环回合。

### 4.2 G-only 与随机对照怎样保持简单公平

候选池、预算、任务平衡都与 α/α_event 相同。G-only 只按是否有事件优先选；事件内部或不够三个时，用预先固定随机种子抽取/补足，**不偷偷用 α 作为第二排序键**。这能直接检验运动变化项是否比“只挑开合事件”增加价值。随机重复只重选已有探针结果，无需重复模型推理。

暂不增加 λ 网格、不同 H、大量预算、各种混合采样或新公式。先完成这个最小比较。

### 4.3 前瞻实验必须避免事后解释

1. **开发阶段：**已有 480 候选、当前 holdout 和已有闭环都已经看过，只能用于方法开发与背景解释。先写清 NFE 候选、误差指标、接受标准和回退规则；现有 P5/P95 不能直接充当成功保证。v0 草案（`results/registry/care_event_nfe_protocol.md`）的做法：
   - 以 NFE=10 两次噪声之间的随机基准为参照，比值取 1.0，检查手臂中位、手臂 P95、夹爪比例和严重夹爪四项；
   - 推荐通过的最低候选 NFE，没有通过者则回退 10；
   - 开发演练中，四种策略对 Pi 和 Smol 都推荐 2。

   草案在审阅、冻结之前不算定稿。
2. **冻结协议：**保存 H、λ、归一化、选样预算、任务分配、模型及算子版本、候选 NFE、噪声种子、判断标准和配置 hash。允许没有任何低 NFE 被接受而回退 10；不要强制输出 Pi=1、Molmo=1、Smol=2。
3. **划分新数据：**候选校准与独立验证按整个 episode 隔离，并排除已用于开发的 episode；若数据不足，明确报告约束。可沿用 40×12 候选、每任务选 3 个的规模。验证集不能用于调公式、阈值或采样种子。
4. **先给建议：**只能用各策略实际选中的 120 个观测的探针输出形成建议。未选中候选的模型输出若用于后续评估，要在建议落盘后才揭晓；不能先看全 480 个输出再声称校准预算只有 120。
5. **再验证：**运行独立集诊断和必要闭环，检查推荐是否通过预先定义的质量条件，报告偏差、拒绝、误推荐及其成本。若看过验证后修改规则，该集转为开发数据，下一次应另设验证。
6. **复用相同配置：**Random 与 CARE 若给出相同 NFE、相同最终模型，可以共用一轮闭环；不同选样策略不能凭相同部署配置得到不同“方法 SR”。CARE 的优势可能是更好地揭示风险或更少的离线探针预算，而不是提高固定 NFE 的动作质量。

目前三个检查点的低 NFE 结果已知。新 episode 实验可以验证冻结协议在新状态上的表现，不能包装成第一次盲猜三个模型的最优 NFE。Molmo 目前缺的是 **CARE_event 探针**，不是所有 NFE1 闭环。完整模型建议也不能自动移植到剪枝模型，联合配置需要对应的离线复查。

### 4.4 未来四周的安排原则

第一周优先完成任务 1/2 与协议冻结；第二周进行新的离线校准和验证；第三周补真正缺失的关键闭环，同时争取正式测速时段；第四周完成表图、论文与复核。写作可以立即开始，未完成实验标记待验证，不预填结果。

不追加小规模功能干预、所有层预算的重跑、完整 CROP 系列或多个新 benchmark。若前瞻验证不支持自动 NFE 推荐，就把 CARE_event 定位为离线风险诊断工具，如实缩小主张，不用事后阈值维持预想结论。

若最终要声称优于 ProbeFlow/AdaVLA，必须做同检查点、硬件、质量口径和计时范围的直接对照；未完成时只讨论离线固定策略与在线自适应的差别。不能把在线方法的两次速度场求值都称作纯额外开销，它们也用于求解。

## 5. 固定实验栈与服务器环境

### 5.1 三条主栈

| 项目 | π0.5 | MolmoAct2 | SmolVLA |
|---|---|---|---|
| 检查点 | `lerobot/pi05_libero_finetuned_v044` | `allenai/MolmoAct2-LIBERO-LeRobot`，本地 `eval_ckpt` | `lerobot/smolvla_libero` |
| VLM 层数 | 18 | 36 | 16 |
| NFE 配置字段 | `num_inference_steps` | `flow_matching_num_steps` | `num_steps` |
| chunk / 实际执行步数 | 50 / 10 | 10 / 10 | 50 / 10 |
| 旧主结果运行时 | eager，compile=false | continuous，bf16，expert graph=false | eager，compile=false |
| 等待环境稳定步数 | 历史默认 10 | 50 | 沿用已验证 launcher |

主 benchmark 是 LIBERO Spatial/Object/Goal/Long（Long 即 LIBERO-10），共 40 任务。历史闭环通常 seed=1000、batch=1；新实验必须记录种子，不可只继承未保存的默认值。主线不需要训练。

SmolVLA：相机 `image→camera1`、`image2→camera2`，`empty_cameras=0`，state 使用现有 normalizer 的 8 维，不用过时配置的 6 维。`HuggingFaceVLA/smolvla_libero` 的 32 层版本是另一模型，不能替换本研究 16 层版本。

Molmo：`load_policy.py` 按本地 HF config 建图、从 LeRobot 权重加载，避免误下大型 base shard。打分使用的 `num_flow_timesteps=1` 不是推理 NFE，不能据此声称已经设置为一步推理。

层评分历史校准通常为 32 帧、seed42、冻结 noise/time；CARE_event 的 480 候选是另一实验，不能混淆校准预算。高重要性分数意味着保留；保护 L0 要显式记录原排序和过滤后集合。只用校准数据评分，不依据验证 SR 回搜层或改 CARE 权重。

### 5.2 环境

主服务器历史记录：`miaa-hpc2`、用户 `guorj`，**4×RTX 3090 24GB**。该服务器没有 A800；不要生成 A800 专用任务。真实机械臂是一对 Piper 双臂，但当前计划不要求启动真机实验。

```bash
cd /home/guorj/Skill_Efficient_VLA
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/guorj/lerobot/src
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python
```

以上是服务器环境，不是任何本地机器都存在的路径。开始工作先检查实际文件和当前环境，不自动重装 LeRobot、不替换已有模型依赖。`HF_ENDPOINT` 必须带 `https://`。

### 5.3 GPU 调度

- 先读当前 GPU、tmux、日志状态；文档中的 GPU 编号与“进行中”只是快照，不是实时占用信息。
- 闭环成功率、离线正确性允许在资源足够的共享 GPU 上执行，不要求四张卡全部空闲。先检查可用显存和资源限制，不影响别人的任务。
- 保守地每张卡只开一个本项目 LIBERO 评估进程。不要把历史同卡并行的两轮直接当作严格可复现对照；新实验固定配置并记录共享状态。
- 共享卡可以测诊断，但正式效率结论须独占目标 3090。闭环总耗时含仿真与失败步数，不能作为模型加速指标。
- 不杀其他用户进程，不用 `tmux kill-server`。自己的任务也按具体 PID/会话处理，不用宽泛匹配。
- 模型加载可能在 1–3 分钟后才显示显存，先检查日志和进程，不因此重复启动。
- `CUDA_VISIBLE_DEVICES=<物理卡号>` 后只用逻辑 `cuda:0`；CUDA event、stream、同步必须指向同一目标设备。不要双重映射误测 GPU0。

## 6. 绝对不要再踩的坑

| 类别 | 必须遵守 |
|---|---|
| 结果可信度 | 同一个实验的 JSON、配置、源码版本共同确定结果。Markdown 不是比原始数据更高的真相；修正文案时保留旧原始记录。未完成回合不算失败，不虚构结果。 |
| 复用与统计 | 相同层集/模型/NFE/协议可复用一轮闭环并明确注明；不是方法名相同就可复用，也不能将一轮共享结果说成多次验证。400 与 2000 明确分母；小差异不自动等于噪声或无损。 |
| skip 算子 | keep-KV、K/V-only、KV 写零、整层恒等、前层 KV 复用不同。连续层残差相同不代表 KV 相同。不要偷偷改正式 Drop-6 keep-KV 路径。 |
| NFE | 一步/两步必须用独立步长覆盖完整 flow 时间区间，不能从十步轨迹提前停在中间状态。NFE 与执行动作步数 H 不是同一概念。核验实际速度场调用次数。 |
| 校准泄漏 | 按 episode 隔离；新验证前冻结规则和建议；只用所选观测输出做建议。不按后验 SR 调层、λ、阈值或种子。 |
| 探针比较 | 匹配输入和初始噪声，reset 队列/缓存；只比较前 10×7 有效动作；夹爪先反标准化；每模型自己的阈值。平均高余弦不等于关键动作未变。 |
| 时序对齐 | 金色命令窗口与物理响应不能混淆；不整体后移 CARE 时间戳。检查 episode 边界，不能把不同 episode 的高排名窗口拼成同一任务连续阶段。 |
| 时延口径 | 同设备、同运行时、同输入的配对；完整 chunk 包含必要处理，分段 profile 另跑。不同会话毫秒不可相减；CUDA Graph 与 eager 分开。不能以 GFLOPs 比例推导实测加速。 |
| 参数口径 | active params 不是实际存储参数；只有实体化释放后才报告对应权重/显存节省；NFE 本身不降低参数量。 |
| 编译 | 不直接打开 `compile_model=true` 就沿用旧 SR；检查 hook 是否被绕过、实际层调用、TF32、动作漂移与重编译。也不能因旧禁令永远排除已可用 Graph 基线。 |
| 模型输入 | Smol 图像需正确 float 预处理、action 要有 batch 维、noise pad 到 max_action_dim=32；最终误差仍只算有效 7 维。该栈不总走 `DecoderLayer.forward`，hook 要核对真正执行路径。 |
| 脚本兼容 | π0.5 `eval_drop.py` 历史 CLI 对类型注解敏感，不随意加 `from __future__ import annotations`。余弦探针 batch 范围是 `(row,row+1)`，不是把终点写成常数 1。 |
| 数值评分 | PPL 在本项目是 action Δloss，不是语言 perplexity；CKA 排序挤在数值噪声时标记不可用；Hessian OOM 就记录 OOM，不能编排名。 |
| 文件与进程 | 新实验独立输出目录、保存 config/log/manifest；不覆盖老结果。不要自动 commit/push。tmux 名称避免互为前缀；等待条件绑定 GPU/用户/模型，不能全局等待任意 `run_drop`。 |

若之后恢复 Token 旁支，另注意：π0.5 通常有两路真实相机＋一路 empty camera，每路 256 token 是 16×16 token 网格；前缀常为 3×256+200=968，不能假设一直是两路。语言 padding 按 mask 删除，不能因切片错误删掉有效语言。移除 masked token、视觉减半及 scout 选择开销必须分别对照；保留正确位置编码和 expert 偏移。

CROP 历史已修过 scout 意外跑完整 VLM、相邻计时点 `zip(strict=True)` 长度不等、CUDA 设备混用、多样性选择循环 `.item()` 同步等问题。相关 CPU 自测通过不等于真实模型等价；先离线检查再闭环。Token 暂缓，不为这些旁支安排新扫描。

## 7. 关键文件地图

| 用途 | 文件 / 目录 |
|---|---|
| 当前研究介绍与主要数字 | [README.md](README.md) |
| 各模型层剪枝结果 | `functional_necessity/main_results/eval_{pi05_libero_v044,molmoact2_libero,smolvla_libero}.md` |
| NFE 历史闭环；有已注明的汇总滞后 | [eval_nfe_libero.md](functional_necessity/main_results/eval_nfe_libero.md) |
| 多轮效率与实现变化 | [efficiency_analysis.md](functional_necessity/main_results/efficiency_analysis.md)，注意拼写 |
| CARE_event 完整记录 | [care_event_action_expert_nfe.md](functional_necessity/main_results/care_event_action_expert_nfe.md) |
| 事件定义及等预算回放 | [event_care_nfe_diagnostic.py](functional_necessity/scripts/stage2_care_v2/event_care_nfe_diagnostic.py) |
| 共享示范、差分统计 | [dump_shared.py](functional_necessity/scripts/stage2_care_v2/dump_shared.py) |
| 两个已实现探针 | [probe_pi05_nfe.py](functional_necessity/scripts/stage2_care_v2/probe_pi05_nfe.py)、[probe_smolvla_nfe.py](functional_necessity/scripts/stage2_care_v2/probe_smolvla_nfe.py) |
| 独立集采样、统计、异常绘图 | [holdout_validation.py](functional_necessity/scripts/stage2_care_v2/holdout_validation.py) |
| 现有 holdout 启动入口 | [run_care_event_holdout_phase2.sh](functional_necessity/scripts/stage2_care_v2/run_care_event_holdout_phase2.sh)，读参数后使用，不原目录盲重跑 |
| 原校准探针与视频 | `functional_necessity/results/stage2_care_v2/nfe_probe_{smolvla,pi05,event_care}/` |
| 修复后独立集全部材料 | [nfe_probe_care_event_holdout/](functional_necessity/results/stage2_care_v2/nfe_probe_care_event_holdout/)，含 REPORT/config/manifest/summary/FIXES 与各模型输出 |
| 三模型加载、打分、skip、闭环 | `functional_necessity/scripts/{pi05_libero,molmoact2_libero,smolvla_libero}/` |
| 实体化与部署测量 | [materialize.py](functional_necessity/scripts/stage_nfe/materialize.py)、[measure_deployment.py](functional_necessity/scripts/stage_nfe/measure_deployment.py) |
| 历史配对效率 | [measure_sota_efficiency.py](functional_necessity/scripts/stage_nfe/measure_sota_efficiency.py)、`functional_necessity/results/stage_nfe/sota_efficiency_*.json` |
| LIBERO-plus | [eval_libero_plus.md](functional_necessity/main_results/eval_libero_plus.md)、`functional_necessity/results/eval_libero_plus/` |

GitHub 不一定包含服务器上所有权重、校准 `.pt`、视频和日志；缺文件先核对服务器与 `.gitignore`，不要把缺上传当成没做实验。`paper/README.md` 的三阶段结构是历史记录，不能覆盖本交接的两贡献决策。

## 8. 新 Cursor 对话的启动顺序与交付要求

1. 阅读本文件和 README，核对仓库版本、未提交改动、上述关键文件及当前进程；不假定服务器与此快照完全相同。
2. 建立“已有可复用 / 仅需规范化 / 真正缺失”的清单。优先补写 Molmo NFE1 2000 结果、修正 Plus 错别数字，追溯 Molmo skip 版本，保护旧原始结果。
3. CPU 完成 G-only 和随机重复选样，修复重绘归档覆盖风险；若统计变化，说明原因和来源，不为了预设结论调规则。
4. 制定并冻结前瞻推荐协议，新增输出目录。v0 草案已有具体数值（`results/registry/care_event_nfe_protocol.md`），审阅通过后用不含 DRAFT 的 `run_name` 执行 `nfe_protocol.py freeze`，必须在看到新验证结果之前完成。
5. 再依次开展新校准/验证、剪枝后离线复查、必要的至少 400 回合闭环，以及独占卡正式测速。已完成的同配置结果优先复用。
6. 每阶段用简体中文交付：做了什么、配置和代码版本、实际结果、可以/不可以支持的结论、下一步；同步 README/HANDOFF 和对应结果文档，状态明确为未启动/进行中/完成/提前停止。

用户偏好：行动方案清晰、必要实验优先、不要重复征求已经给过的授权，不为了“全面”无限追加消融。新发现与现有结论冲突时如实报告。论文可以润色与提高概括层次，但不能编造数据、隐藏负结果，或把待验证方法写成已经成立的保证。
