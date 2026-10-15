# Event-Aware Calibration for Few-Step Flow Matching

**面向少步流匹配的事件感知校准**

> 更新日期：2026-10-16  
> 开发证据快照：Jerry-VLA `87342bba134aca3f548e37c0f48defeede7ba0c7`  
> 正式前瞻运行：`prospective_v1_20261015`（ledger 已 `validate`；闭环列为历史 `eval_info.json` 复用）  
> 建议存放位置：`Skill_Efficient_VLA/functional_necessity/main_results/care_event_action_expert_nfe.md`

本文记录将 CARE 应用于 action expert NFE 校准的研究思路、开发阶段探针证据，以及 **2026-10-15 冻结的正式前瞻校准与独立开环验证**（π0.5→NFE **1**、SmolVLA→NFE **2**、MolmoAct2→NFE **1**）。开发段数字以修正后的 CSV 为准；正式段以 `results/stage2_care_v2/nfe_protocol/prospective_v1_20261015/` 封存为准。文中的「独立验证」指与校准 episode 隔离的 holdout 观测集，不表示策略在 LIBERO 闭环中从未见过相关任务分布。

## 1. 核心 Idea

**使用 CARE_event 优先选择运动变化和夹爪开合决策附近的观测，在部署前检查低 NFE 是否会改变关键动作，为固定 NFE 配置提供诊断依据。**

Flow-matching VLA 的 action expert 通常通过多次速度场求值生成动作块。此前 π0.5 的轨迹探针观察到，部分输入下速度场沿求解路径的方向变化较小，因此默认 NFE=10 可能存在减少求解步数的空间。

然而，只看大量普通样本的平均动作余弦，容易忽略少数具有较大偏差的状态。例如：

- 抓取时，夹爪闭合是否及时；
- 放置前，模型是否准确调整位置与运动方向；
- 释放时，模型是否在合适的动作步发出松爪命令。

CARE_event 的作用是提高有限校准预算的诊断价值。它本身不直接减少推理计算；实际计算节省来自随后采用较低的 NFE。

预期的部署流程是：

```text
示范轨迹
  → 计算动作变化 α 和夹爪事件 G
  → 按任务选出 α_event 最高与较高的观测
  → 在相同输入、相同初始噪声下比较不同 NFE
  → 形成固定 NFE 候选
  → 使用闭环评估确认质量与效率
```

正式前瞻路径已实现：**冻结划分 → 校准探针 → 三规则推荐 → holdout 开环检验**；闭环成功率来自本机既有 Full 模型评测的 **历史复用导入**（早于校准 freeze，非本轮新跑闭环）。

### 1.1 与早期 Layer-Pruning CARE 的关系

早期 CARE 研究动作变化权重能否改善层重要性估计。本阶段保留动作变化的基本思想，将其用于 NFE 探针的样本排序和选择。

当前 NFE 实验中，α_event 用于选择观测，不用于训练模型、修改 flow-matching 损失、改变求解器更新公式，或在线决定每个动作块执行多少步。层剪枝 CARE 的结果与本报告的 action expert 结果应分别记录。

### 1.2 为什么同时保留“最高”与“较高”分数

现有视频审阅显示，最高 α_event 样本经常包含夹爪命令切换；较高分数样本则可以补充手臂运动变化、对齐和调整等阶段。

只保留最高分可能过度集中在夹爪事件。每任务取前三名，能够保留排名第一以及排名第二、第三的样本。

“最高”和“较高”均指当前候选池内的相对排名，不代表已经搜索整条轨迹并找到了所有关键时刻。

## 2. CARE_event 的计算定义

为避免混淆，使用 τ 表示示范轨迹的执行时间索引，使用 t 表示 flow-matching 求解时间。CARE 分数根据执行时间上的动作命令变化计算，不直接衡量速度场在 flow 时间上的曲率。

### 2.1 原始动作变化分数 α

对观测 τ，读取同一 episode 内从 τ 开始、最长 H=10 个示范动作。每个动作包含七个有效维度：前三维平移命令、随后三维旋转命令，以及最后一维夹爪命令。

原 CARE v2 使用示范动作相邻差分的逐维标准差进行标准化：

$$
z_j = (a_{j+1}-a_j) \oslash \sigma_{\Delta a},
$$

其中 $\oslash$ 表示逐维相除。$\sigma_{\Delta a}$ 来自原 CARE v2 对示范数据同 episode 相邻动作差分的统计，后续实验固定使用该统计量。此处定义以 `dump_shared.py` 与 `shared.pt` 为准，与早期 layer-pruning 文档中的动作归一化口径应分别核对，不能直接混用。

设窗口实际有效长度为 $m_\tau$，则：

$$
d_\tau = \frac{1}{m_\tau-1}\sum_{j=\tau}^{\tau+m_\tau-2}\|z_j\|_2,
\qquad
\alpha_\tau = \frac{d_\tau}{\operatorname{mean}_{\mathrm{cal}}(d)}.
$$

完整 H=10 窗口平均九个相邻差分。原始候选集中 episode 末尾的窗口可能被截短，按实际有效长度计算；本阶段独立 holdout 的窗口均为完整 H=10。

本轮固定校准均值：

```text
mean_cal(d) = 2.0472331711711984
```

α 衡量命令序列变化强度。它不等于实际机械臂速度、夹爪关节运动、物体运动或任务重要性的真值。

### 2.2 夹爪命令事件 G

将原始示范夹爪命令的正负号转换为开合状态：

$$
b_j = \mathbb{1}[g_j > 0].
$$

如果窗口内部任意相邻两个动作的开合状态不同，则：

$$
G_\tau = \mathbb{1}[\exists j:\ b_{j+1}\ne b_j].
$$

当前实现同时包含“张开→闭合”和“闭合→张开”。G 只检查窗口内部的相邻动作，不包含窗口前一动作到起点的切换。

### 2.3 事件增强分数 α_event

$$
\alpha_{\mathrm{event},\tau}
= \frac{\alpha_\tau + \lambda G_\tau}
{\operatorname{mean}_{\mathrm{cal}}(\alpha+\lambda G)},
\qquad \lambda=1.
$$

当前 H=10、事件权重 λ=1。没有在本轮独立验证后重新搜索这些参数。

本轮固定归一化均值：

```text
mean_cal(alpha + G) = 1.1166666665580123
```

Holdout 沿用原校准差分标准差及上述两个均值，不根据 holdout 重新拟合归一化统计。

### 2.4 命令切换与实际夹爪响应的时间关系

高 α_event 窗口可以发生在实际夹爪明显张开之前。因为评分依据是示范命令，实际机械响应可能稍后才显现。

例如在已审阅的 Long 样本中，观测起点为第 262 帧，计分窗口为 262–271 帧，夹爪命令在第 266 帧切换，明显物理变化主要出现在窗口之后。

这不要求将校准起点整体后移。当前目标是检查模型从观测起点出发，能否生成正确的运动和开合命令。实际释放的精确延迟尚未通过夹爪关节状态测量，不能统一假定为固定帧数。

同时，低 α 只能称为“命令变化较平滑”，不能直接称为“不重要”。平滑运输、慢速对齐等动作也可能要求较高精度。

## 3. 离线 NFE 探针与推荐协议

### 3.1 模型与执行窗口

| 项目 | π0.5 | SmolVLA | MolmoAct2 |
|---|---|---|---|
| 检查点 | `lerobot/pi05_libero_finetuned_v044` | `lerobot/smolvla_libero` | `allenai/MolmoAct2-LIBERO-LeRobot` |
| 候选 NFE | 1、2、3、10 | 1、2、3、10 | 1、2、3、10 |
| 初始噪声 | 2 | 2 | 2 |
| 执行窗口 H | 10 | 10 | 10 |
| 有效动作维 | 7 | 7 | 7（Molmo 手臂经分位数后处理映射到物理夹爪） |

分析对象为 **Full 模型**（无 VLM Drop / token pruning 联合压缩）。**H=10** 是执行的前 10 步示范对齐窗口；**NFE=10** 是生成该块时的速度场求值次数。降 NFE 不缩短执行窗口。

选样分数 **α、G、α_event 仅来自示范命令**（§2），**不读取**任何低 NFE 探针误差。正式前瞻的 episode 划分、校准池与验证集在 freeze 时一次性固定（`prospective_v1_20261015`）。

### 3.2 统一误差空间与 $E_N$

三模型探针输出先经各栈后处理，再映射到 **LIBERO 归一化 7 维动作空间**（`nfe_en_metrics` / `dev_metrics_common`），与开发阶段固定的 LIBERO mean/std 一致。

对每个观测 $\times$ 每组噪声、候选 NFE $n$：

1. 固定图像、状态、语言及初始噪声；
2. 生成动作块，取前 $10\times 7$ 维；
3. 以 **同噪声 NFE=10** 输出为参考；
4. 计算 LIBERO 归一化 RMSE，记为一次比较。

$$
E_n = \mathrm{mean}_{\text{obs},\text{noise}}\ \mathrm{RMSE}_{7\text{-d, LIBERO}}(a_n, a_{10}).
$$

正式推荐在 **校准并集**（238 观测）上统计 $E_1,E_2,E_3$；独立验证在 **holdout 480 观测**上复算，**不**用验证集重新选步替代校准推荐。

噪声种子（与探针脚本一致）：

```text
noise_seed = 1000 * 1000003 + pool_index * 2 + noise_id
```

### 3.3 三规则独立推荐（无旧四门槛）

候选集合 $\{1,2,3\}$，参考 NFE=10，若无规则通过则回退 10。

| 规则 | 含义 | 冻结阈值 |
|---|---|---|
| **主规则 abs** | 取最小 $n$ 使 $E_n \le \tau$ | $\tau=0.085$ |
| **abs improve** | 相对 $n{+}1$ 的绝对改善 | $\Delta E \le 0.03$，且单调不增 |
| **rel improve** | 相对改善幅度 | $\Delta E / E_n \le 0.375$，且单调 |

三条规则 **独立报告**，主推荐字段对齐 abs 规则。缺少相邻候选（如 $n{=}3$ 的 improve 规则）记为 **NA**。部署时 **固定** 推荐 NFE；**不在线** 计算 CARE 分数、不做 per-step 路由、**不改变** Euler 更新公式，也 **不做** recovery 训练。

### 3.4 诊断指标（非接受门槛）

- **余弦、手臂分维 RMSE、尾部 3 步 RMSE、夹爪符号不一致率**：用于解释与审阅，**不**恢复历史「四门槛」自动否决。
- 开发段 480 候选池上，尾部标记 $B$ 仍使用各模型 NFE=1 的 P5/P95 阈值（见旧表）；正式前瞻报告中的夹爪/手臂比率为 **相对 NFE=10 的诊断**，与 `OPEN_LOOP_VALIDATE.md` 一致。

| 模型 | noise | cosine P5 | RMSE P95 |
|---|---:|---:|---:|
| π0.5 | 0 | 0.993313 | 0.116572 |
| π0.5 | 1 | 0.993285 | 0.120798 |
| SmolVLA | 0 | 0.984906 | 0.159371 |
| SmolVLA | 1 | 0.983971 | 0.157276 |

不确定性：正式汇总中 episode 为聚簇单位（bootstrap $B{=}2000$，seed 20261015）；**同一 episode 的多个窗口与两组噪声不视为独立 episode**。

---

**以下 §4–§8 为开发阶段证据**（原 480 候选池、120 观测 holdout、π0.5/SmolVLA 两栈）。与 §9 正式前瞻运行分开表述，不混为同一批实验。

## 4. 已完成实验一：探针路径正确性检查

上传的代表性样本校验记录显示：

- 各 NFE 的实际速度场调用次数与配置一致；
- NFE=10 与官方路径输出的 `max_abs` 差异为 0；
- reset 后重复执行 NFE=10，输出一致；
- NFE=1 输出与 NFE=10 不完全相同，确认配置实际生效；
- 比较动作形状为 10×7。

独立 CSV 审查还确认，所有记录的速度调用数与 NFE 一致，主要动作指标有限，样本×噪声×NFE 组合没有缺失或重复。官方路径的逐样本一致性校验记录针对代表性样本，不能扩大表述为对所有输入都重跑了官方对照。

## 5. 已完成实验二：高、较高、低 α_event 视频审阅

扩展预览展示了四个 LIBERO suite 中的八个任务，每任务候选池内排名第 1、2、3 和第 12 的样本。

| 排名 | 展示窗口数 | 包含夹爪命令切换 G=1 |
|---|---:|---:|
| 第 1 名 | 8 | 7 |
| 第 2 名 | 8 | 5 |
| 第 3 名 | 8 | 0 |
| 第 12 名 | 8 | 0 |

人工审阅观察到：

- 最低分样本通常呈现平滑命令对应的运动；
- 最高分样本经常覆盖即将开合夹爪的决策窗口；
- 较高分样本能够补充手臂运动调整等阶段。

这些观察支持“最高＋较高”选样的解释，但不是任务重要性的定量标注。不同排名样本经常来自不同 episode，不能拼成同一次任务的连续阶段。

例如“把碗放到盘子上”的展示样本中，第一名窗口包含开爪命令，第二名包含闭爪命令，第三名没有夹爪切换、得分来自手臂动作变化。这说明 CARE_event 可以覆盖抓取、释放及运动调整，不仅关注松爪。

## 6. 已完成实验三：等预算校准选样比较

### 6.1 设计

原候选池为 480 个观测：40 个任务，每任务来自 12 个回合的候选时间戳。比较三种选择，各取每任务三个观测，共 120 个：

- `random`：固定种子 42 随机选择；
- `care_alpha`：按原始 α 取前三名；
- `care_event`：按 α_event 取前三名。

本阶段复用原 480 观测的已有模型探针输出。选择预算是 120，但为评价选择效果，实际已有探针池包含全部 480 个观测。

### 6.2 等预算四组与 G-only（2026-10-15 CPU 回放）

在不变探针输出的前提下，于 `results/stage2_care_v2/nfe_probe_event_care/budget_compare/` 复现 Random（seed 42）、α、α_event，并新增 **G-only**（每任务优先 G=1，不足或并列时用同种子随机补足，**不用 α 作第二排序**）。合并两噪声的捕获表见 `REPORT.md` 与 `four_group_merged_two_noises.csv`；20 个固定种子的 Random/G-only 波动见同目录 `summary.json`。

相对单次 Random(seed42)，G-only 提高夹爪 NFE1 捕获（Smol 8→13，Pi 5→10），整体尾部介于 Random 与 α_event 之间；α_event 在 Smol 尾部仍最高（44/73）。多随机种子下 Pi 的 G-only 夹爪捕获恒为 10/10（20/20 种子），Random 波动较大（Pi grip 均值 2.45，std 1.56）。

### 6.3 夹爪事件覆盖

原候选池共有 56 个示范夹爪事件窗口。

| 选样方式 | 选择观测数 | 覆盖事件窗口 | 事件覆盖率 |
|---|---:|---:|---:|
| 随机 | 120 | 19/56 | 33.9% |
| 原始 CARE α | 120 | 34/56 | 60.7% |
| CARE_event | 120 | 55/56 | 98.2% |

事件项本身就是选样分数的一部分，因此事件覆盖提高符合设计预期。更有价值的检查是，它是否同时提高模型动作偏差的捕获。

### 6.4 NFE=1 夹爪偏差捕获

下表分母为原 480 个候选中对应噪声出现夹爪开合不一致的观测数量。

| 模型 | noise | 随机 | 原始 CARE α | CARE_event |
|---|---:|---:|---:|---:|
| SmolVLA | 0 | 3/6 | 4/6 | 6/6 |
| SmolVLA | 1 | 5/9 | 7/9 | 8/9 |
| π0.5 | 0 | 3/7 | 5/7 | 7/7 |
| π0.5 | 1 | 2/3 | 2/3 | 3/3 |

两组噪声的样本比较合计：SmolVLA 从随机 8/15、原始 CARE 11/15 提高到 CARE_event 14/15；π0.5 从 5/10、7/10 提高到 10/10。这些合计不是独立回合数。

### 6.5 NFE=1 整体尾部偏差捕获

| 模型 | noise | 随机 | 原始 CARE α | CARE_event |
|---|---:|---:|---:|---:|
| SmolVLA | 0 | 12/36（33.3%） | 19/36（52.8%） | 22/36（61.1%） |
| SmolVLA | 1 | 14/37（37.8%） | 17/37（45.9%） | 22/37（59.5%） |
| π0.5 | 0 | 11/36（30.6%） | 20/36（55.6%） | 21/36（58.3%） |
| π0.5 | 1 | 12/35（34.3%） | 18/35（51.4%） | 17/35（48.6%） |

两组噪声按分子、分母合计：

- SmolVLA：随机 26/73，原始 CARE 36/73，CARE_event 44/73。
- π0.5：随机 23/71，原始 CARE 38/71，CARE_event 38/71。

因此，CARE_event 比本轮随机组捕获更多尾部偏差；相比原始 CARE，优势集中在夹爪偏差，π0.5 的整体合计并未继续提高。

手臂误差最高 5% 的捕获也存在预算取舍：SmolVLA 原始 CARE 两组均为 15/24，事件版本为 14/24、13/24；π0.5 从 14/24、13/24 变为 13/24、12/24。事件加分不能保证所有误差类型同时改善。

这些比较是在既有候选池上的回放分析，随机组使用一个固定采样种子，尚未给出重复随机选样的统计区间。

## 7. 已完成实验四：独立回合 Holdout NFE 探针

### 7.1 独立集与检查范围

- 采样种子为 4242；
- 排除原 480 个候选观测涉及的全部 episode；
- 每任务随机选择三个新的不同回合，每回合随机取一个完整 H=10 窗口；
- 共 120 个观测，来自 120 个不同回合；
- 与原候选集的回合交集为零；
- 包含 16 个示范夹爪事件窗口；
- 每观测运行两组噪声，每个 NFE 共 240 次动作比较。

π0.5 CSV 共 720 行，SmolVLA 共 960 行，包含 NFE=10 参考行。

Holdout 是独立的随机观测集。本轮检查了低 NFE 在新回合上的动作差异，尚未在 holdout 上完成同预算选样策略比较，也没有涉及新任务、跨 benchmark 或闭环访问状态分布的验证。

### 7.2 两组噪声合并结果

| 模型 | NFE | 平均余弦 | 平均 RMSE | RMSE P95 | 尾部偏差比较数 | 夹爪不一致比较数 |
|---|---:|---:|---:|---:|---:|---:|
| π0.5 | 1 | 0.99782 | 0.0560 | 0.1024 | 13/240 | 0/240 |
| π0.5 | 2 | 0.99890 | 0.0395 | 0.0662 | 3/240 | 0/240 |
| SmolVLA | 1 | 0.99342 | 0.0942 | 0.1459 | 22/240 | 2/240 |
| SmolVLA | 2 | 0.99747 | 0.0568 | 0.0864 | 2/240 | 2/240 |
| SmolVLA | 3 | 0.99849 | 0.0420 | 0.0592 | 2/240 | 2/240 |

合并 P95 从全部 240 条 RMSE 重新计算，不平均两个噪声切片的 P95。240 次比较来自 120 个观测，不能按 240 个独立回合解释。

### 7.3 分噪声的异常数量

| 模型 | NFE | noise0 尾部 | noise1 尾部 | noise0 夹爪不一致 | noise1 夹爪不一致 |
|---|---:|---:|---:|---:|---:|
| π0.5 | 1 | 7/120 | 6/120 | 0/120 | 0/120 |
| π0.5 | 2 | 2/120 | 1/120 | 0/120 | 0/120 |
| SmolVLA | 1 | 10/120 | 12/120 | 1/120 | 1/120 |
| SmolVLA | 2 | 1/120 | 1/120 | 1/120 | 1/120 |
| SmolVLA | 3 | 1/120 | 1/120 | 1/120 | 1/120 |

### 7.4 主要观察

**π0.5：**一步的平均动作一致性较高，夹爪符号没有发生不一致，但有少量手臂偏差。两步进一步减少动作误差及尾部数量，因此 NFE=1 是优先闭环候选，NFE=2 可以作为质量与效率取舍的对照。

**SmolVLA：**从一步到两步，平均 RMSE 降低约 39.7%，240 次比较中的 RMSE 均下降，尾部数量从 22 次降到 2 次。两步值得优先验证，但剩余两例夹爪偏差并未消除。三步继续改善平均误差，却没有解决这两例。

## 8. 已完成实验五：异常案例与分数关联检查

### 8.1 SmolVLA #16/noise0：提前松爪

```text
global_index = 65123
episode = 241
frame = 248
task = put both the alphabet soup and the tomato sauce in the basket
G = 1
alpha = 1.1613
alpha_event = 1.9355
```

| NFE | 余弦 | RMSE | 夹爪不一致动作步数 |
|---|---:|---:|---:|
| 1 | 0.91477 | 0.23035 | 1 |
| 2 | 0.95889 | 0.15335 | 1 |
| 3 | 0.94703 | 0.17346 | 1 |

上传曲线显示，三个低 NFE 相对 NFE=10 都提前一个动作步发出松爪命令。这里检查的是命令时机差异，不是实际机械响应延迟。

该例还说明，个别样本从 NFE=2 到 NFE=3 的误差可以增大，不能假设每增加一步都单调改善。

### 8.2 SmolVLA #48/noise1：延迟闭合

```text
global_index = 106748
episode = 418
frame = 48
task = turn on the stove
G = 1
alpha = 1.0285
alpha_event = 1.8165
```

| NFE | 余弦 | RMSE | 夹爪不一致动作步数 |
|---|---:|---:|---:|
| 1 | 0.94284 | 0.24777 | 1 |
| 2 | 0.96705 | 0.18442 | 1 |
| 3 | 0.97351 | 0.16542 | 1 |

上传曲线显示，低 NFE 相对 NFE=10 延迟一个动作步闭合夹爪。与 #16 相同，该例属于高事件分数窗口，但偏差方向不同。

两例支持夹爪事件加分的诊断价值。它们尚不能证明任务必然失败；是否发生功能性影响需要闭环确认。

### 8.3 π0.5 #7：无夹爪事件的手臂偏差

```text
global_index = 90998
episode = 339
frame = 195
task = put the yellow and white mug in the microwave and close it
G = 0
alpha = 1.9168
alpha_event = 1.7165
```

该例最差噪声下，NFE=1 的 RMSE 约为 0.21979，NFE=2 降到约 0.13932，夹爪符号均一致。

这支持同时保留高原始 α 的运动变化样本，不能只检查 G=1 的开合窗口。

此外，π0.5 #33 的 α_event 约为 0.681，没有夹爪事件，但 NFE=1 仍落入余弦尾部。因此，低分样本也可能有偏差，CARE 不能作为排除所有低分状态风险的证明。

### 8.4 独立集上的分数—误差关联

以每个观测两组噪声的 NFE=1 平均 RMSE 为误差，对 120 个观测做 Spearman 等级相关的独立 CSV 复算：

| 模型 | 原始 α 与平均 RMSE | α_event 与平均 RMSE |
|---|---:|---:|
| π0.5 | 0.591 | 0.506 |
| SmolVLA | 0.433 | 0.399 |

这些数字来自原始 CSV 的独立复算，未作为新模型实验。α_event 按原校准固定均值重建，两组噪声先在同一观测内取平均，再计算相关。

结果表明，原始动作变化分数与低 NFE 误差的关联在独立回合上仍然存在。事件增强没有提高整体 RMSE 相关性，它优先强调夹爪切换，因此应依据夹爪与手臂覆盖的共同表现评价。

该关联不能证明动作命令不连续导致速度场弯曲，也不能证明某个高分观测必然需要更多 NFE。机制因果解释仍未完成。

## 9. 正式前瞻校准与独立开环验证（`prospective_v1_20261015`）

运行目录：`results/stage2_care_v2/nfe_protocol/prospective_v1_20261015/`。`recommend` 与 `validate` 已写入 ledger；闭环列为 **`closed-loop-import` 历史复用**（见 §9.5），**不是** freeze 之后新跑的前瞻闭环。

### 9.1 数据划分与探针预算

| 角色 | Episode | 观测 | 说明 |
|---|---:|---:|---|
| 校准池 | 400 | 400（每 episode 1 窗） | 排除开发 `shared.pt` 与旧 holdout episode |
| 每策略选样 | — | 120（40 任务 × top-3） | random / α / g_only / α_event |
| 四策略并集探针 | — | **238**（去重） | 联合速度场 **7616** 次/模型（含 NFE10 参考） |
| 独立验证 | 160 | **480**（每 episode 3 窗） | episode 与校准无交；双噪声 |

每策略仅跑所选 120 窗时，速度场求值为 $120\times 2\times(1+2+3+10)=3840$；并集实际 **7616**。验证阶段 **15360** 次/模型。统计口径均含 NFE=10 参考成本。

### 9.2 正式校准推荐（并集 238 obs，`04_recommend/recommendations.json`）

四策略 **推荐 NFE 完全一致**（主规则与两对照规则相同）。

| 模型 | $E_1$ | $E_2$ | $E_3$ | abs≤0.085 | abs improve | rel improve | 主推荐 |
|---|---:|---:|---:|---:|---:|---:|---:|
| π0.5 | 0.0805 | 0.0533 | 0.0390 | 1 | 1 | 1 | **1** |
| SmolVLA | 0.1202 | 0.0686 | 0.0496 | 2 | 2 | 2 | **2** |
| MolmoAct2 | 0.0428 | 0.0311 | 0.0239 | 1 | 1 | 1 | **1** |

（表中 $E_n$ 取 `alpha_event` 策略的 `E_on_selection`；另三策略数值相同至四位小数。）

### 9.3 独立开环 holdout（480 obs，`05_validation/open_loop_validation.json`）

对 **校准已冻结的 screening NFE** 检验三规则（阈值未调）；**不用**验证集 `holdout_diagnostic_repick` 替代推荐。

| 模型 | 原推荐 | 验证 $E_1$ | $E_2$ | $E_3$ | abs | abs imp. | rel imp. |
|---|---:|---:|---:|---:|---|---|---|
| π0.5 | 1 | 0.0629 | 0.0428 | 0.0321 | pass | pass | pass |
| SmolVLA | 2 | 0.0971 | 0.0566 | 0.0413 | pass | pass | pass |
| MolmoAct2 | 1 | 0.0358 | 0.0249 | 0.0193 | pass | pass | pass |

### 9.4 推荐 NFE 的夹爪 / 手臂诊断（验证集，相对 NFE=10）

比较次数：480 obs × 2 noise；episode 聚簇 160。下表为 **诊断**，非新接受门槛。

| 模型 | $N^\*$ | 夹爪不一致率 | 手臂 median ratio | 手臂 P95 ratio |
|---|---:|---:|---:|---:|
| π0.5 | 1 | 0.0156 | 1.154 | 0.884 |
| SmolVLA | 2 | 0.0052 | 0.697 | 0.531 |
| MolmoAct2 | 1 | 0.0042 | 1.852 | 1.016 |

工程注记：Molmo 验证探针在 GPU 上与历史 `eval_drop` 共卡时，冻结脚本会拒绝运行；实际使用运行目录副本 `probe_molmoact2_nfe_validation.py`（`--ignore-eval-drop`），见 `05_validation/OPERATIONAL_NOTES.md`。**不等于** freeze 字节与运行时探针一致。Molmo 数据分片在校准阶段已补全 `molmo_root`，验证 480 `global_index` 全覆盖。

### 9.5 历史闭环复用（`closed-loop-import`，非新前瞻闭环）

自 `eval_info.json` 整理导入（`success_rate_unit=percent`，seed=1000，四套件）。**早于** `prospective_v1_20261015` freeze；仅用于 `validate` 的闭环列，**不能**写成 CARE 前瞻新闭环。

| 模型 | 配置 | NFE | Spatial | Object | Goal | Long | Avg. | 成功/回合 | 来源 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| π0.5 | full | 10 | 97.0 | 100.0 | 95.0 | 97.0 | **97.25** | 389/400 | `results/eval_drop/baseline/eval_info.json` |
| π0.5 | full | 1 | 97.0 | 99.0 | 97.0 | 95.0 | **97.00** | 388/400 | `results/stage_nfe/pi05_full_nfe1/eval_info.json` |
| SmolVLA | full | 10 | 78.0 | 89.0 | 89.0 | 76.0 | **83.00** | 332/400 | `results/smolvla_libero/lerobot_official_nas10/eval_info.json` |
| SmolVLA | full | 2 | 77.0 | 89.0 | 88.0 | 74.0 | **82.00** | 328/400 | `results/stage_nfe/smolvla_full_nfe2/eval_info.json` |
| MolmoAct2 | full | 10 | 97.4 | 99.6 | 96.8 | 96.2 | **97.50** | 1950/2000 | `results/molmoact2_libero/baseline_eval/eval_info.json` |
| MolmoAct2 | full | 1 | 97.4 | 99.4 | 97.4 | 97.0 | **97.80** | 1956/2000 | `results/stage_nfe/molmoact2_full_nfe1_2000/eval_info.json` |

相对 NFE=10：π0.5 **−0.25 pp**；SmolVLA **−1.00 pp**；Molmo **+0.30 pp**（同为 2000 集协议）。`06_validate/VALIDATE.md` 中 holdout 与闭环列均为 **pass**（闭环规则：Avg. ≥ ref−2 pp 且各套件 ≥ ref−5 pp）。

## 10. 当前能够成立的结论

### 10.1 正式前瞻：冻结规则在新 episode 上通过开环验证

在独立 480 观测上，校准选定的 π0.5 **NFE=1**、SmolVLA **NFE=2**、MolmoAct2 **NFE=1** 均满足三条开环规则（§9.3）。这支持 **固定部署 NFE**，但不等于任务成功保证或通用阈值。

### 10.2 闭环：有可复用 Full 模型证据，但非本轮新实验

§9.5 历史闭环表明推荐 NFE 与 NFE=10 锚点在同协议下 **未出现套件塌缩**（见 `eval_nfe_libero.md`）。该证据 **不**归因于 CARE 选样，也 **不**构成「前瞻闭环」。

### 10.3 开发段：CARE_event 提高夹爪相关诊断覆盖（§4–§8）

在开发 480 候选、120 预算回放中，α_event 相对 random 提高夹爪事件与夹爪偏差捕获；相对纯 α 有手臂尾部取舍。**四策略正式推荐相同**，不能宣称 α_event **提高推荐准确率**或必然节省探针预算；相对 g_only 的优势应只看 §6 的捕获率与 §9 的误差表，不作夸大。

### 10.4 定位

> 用示范命令变化与夹爪事件选择离线压力样本，在冻结阈值下为 Full 模型的固定 NFE 提供校准、开环检验与（历史）闭环对照。

## 11. 目前尚未证明的内容

- **通用阈值 / 跨 benchmark：**0.085 与 improve 阈值仅在本次 LIBERO 前瞻协议下冻结验证。
- **联合压缩：**Full 模型结论不直接外推到 Drop-6 / Drop-16 × 低 NFE。
- **在线 CARE：**部署不计算 α_event，不做 per-chunk 路由。
- **因果机制：**速度场冗余与夹爪时机偏差的相关性仍属诊断层面。
- **效率终局：**未在本章给出新的独占 GPU 端到端效率表；后续重点为已确认联合配置下的部署测量，**不**重复安排三组 Full 闭环筛查。

## 12. 后续推进原则

1. 保持 `prospective_v1_20261015` 冻结清单与推荐不变；新数据用新 `run_name`。
2. 闭环仅复用或补充 **与探针同权重同协议** 的评测；剪枝栈需单独校准。
3. 报告效率时区分速度场次数、chunk 墙钟与闭环 SR（见 `efficiency_analysis.md` / `eval_nfe_libero.md`）。
4. Molmo 探针若需与 `eval_drop` 共卡，在运行目录维护补丁脚本并文档化，勿声称与 freeze 哈希一致。

## 13. 数据来源、版本与工程记录

### 13.1 主要结果与代码

**正式前瞻（`prospective_v1_20261015`）**

- [ledger](../results/stage2_care_v2/nfe_protocol/prospective_v1_20261015/ledger.jsonl)
- [校准推荐](../results/stage2_care_v2/nfe_protocol/prospective_v1_20261015/04_recommend/recommendations.json)
- [开环验证汇总](../results/stage2_care_v2/nfe_protocol/prospective_v1_20261015/05_validation/open_loop_validation.json)
- [正式 validate](../results/stage2_care_v2/nfe_protocol/prospective_v1_20261015/06_validate/validation.json)
- [历史闭环导入](../results/stage2_care_v2/nfe_protocol/prospective_v1_20261015/05_validation/historical_closed_loop_import.json)
- [闭环导入脚本](../results/stage2_care_v2/nfe_protocol/prospective_v1_20261015/05_validation/build_historical_closed_loop_import.py)
- [NFE 闭环背景文档](eval_nfe_libero.md)

**开发阶段（§4–§8）**

以下相对链接按本文件放入 `functional_necessity/main_results/` 后的位置组织：

- [独立验证 REPORT](../results/stage2_care_v2/nfe_probe_care_event_holdout/REPORT.md)
- [独立验证 summary](../results/stage2_care_v2/nfe_probe_care_event_holdout/summary.json)
- [Holdout 采样清单](../results/stage2_care_v2/nfe_probe_care_event_holdout/holdout_manifest.json)
- [π0.5 Holdout 原始 CSV](../results/stage2_care_v2/nfe_probe_care_event_holdout/pi05/frames.csv)
- [SmolVLA Holdout 原始 CSV](../results/stage2_care_v2/nfe_probe_care_event_holdout/smolvla/frames.csv)
- [原始 π0.5 校准探针](../results/stage2_care_v2/nfe_probe_pi05/)
- [原始 SmolVLA 校准探针](../results/stage2_care_v2/nfe_probe_smolvla/)
- [事件 CARE 校准与视频索引](../results/stage2_care_v2/nfe_probe_event_care/)
- [八任务最高与最低样本](../results/stage2_care_v2/nfe_probe_event_care/preview_multitask_event.csv)
- [八任务第二与第三名样本](../results/stage2_care_v2/nfe_probe_event_care/preview_multitask_event_subtop.csv)
- [当前曲线索引](../results/stage2_care_v2/nfe_probe_care_event_holdout/curves/curve_index.json)
- [SmolVLA #16/noise0 曲线](../results/stage2_care_v2/nfe_probe_care_event_holdout/curves/smolvla/holdout_016_noise0_grip.png)
- [SmolVLA #48/noise1 曲线](../results/stage2_care_v2/nfe_probe_care_event_holdout/curves/smolvla/holdout_048_noise1_grip.png)
- [π0.5 既有闭环背景配置](../results/stage2_care_v2/nfe_probe_care_event_holdout/pi05/config.json)
- [SmolVLA 既有闭环背景配置](../results/stage2_care_v2/nfe_probe_care_event_holdout/smolvla/config.json)
- [事件评分代码](../scripts/stage2_care_v2/event_care_nfe_diagnostic.py)
- [α 与差分标准化代码](../scripts/stage2_care_v2/dump_shared.py)
- [独立验证分析代码](../scripts/stage2_care_v2/holdout_validation.py)
- [π0.5 探针代码](../scripts/stage2_care_v2/probe_pi05_nfe.py)
- [SmolVLA 探针代码](../scripts/stage2_care_v2/probe_smolvla_nfe.py)
- [Molmo 探针代码（freeze）](../scripts/stage2_care_v2/probe_molmoact2_nfe.py)
- [前瞻协议 CLI](../scripts/stage2_care_v2/nfe_protocol_v1.py)

### 13.2 采用修正后的数字

早期 holdout 分析误将 SmolVLA 校准阈值应用于 π0.5，导致后者 NFE=1 每组噪声仅报告一个尾部样本。正确结果为 noise0 七个、noise1 六个，共 13/240。本文全部使用修正后的自身模型阈值。

两组噪声及全部低 NFE 已补齐；无效的全 holdout “100% 召回”表述已删除。Holdout 旧 `in_top_alpha/in_random/in_low_alpha` 列由于每任务仅三个候选而全为 1，不参与选样比较。

当前 20 张标注 `tail_nfe1` 的图均满足对应异常条件，SmolVLA 两张夹爪图已包含 NFE=3。统计、异常身份与曲线索引已经复核。

最新工程审查仍发现，重复运行归档函数会覆盖同名历史图备份，建议改用运行时间戳保存新备份。该问题不改变本文的 CSV 数值与本次曲线，应与研究结论分开处理。

### 13.3 审查范围

本文基于仓库源码、原始 CSV 独立复算、采样清单、模型正确性记录及代表性曲线检查。完整 `actions.npz` 存于实验服务器，当前 GitHub 快照未上传，因此未在审查端从完整动作数组重新计算所有逐维误差，也未重跑 GPU 推理。

固定快照：[Jerry-VLA / 87342bb](https://github.com/CJieee30/Jerry-VLA/tree/87342bba134aca3f548e37c0f48defeede7ba0c7)。后续新增结果应单独注明版本、模型配置与证据范围，不覆盖或混用本阶段的统计口径。
