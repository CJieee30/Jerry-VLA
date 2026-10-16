# CARE_event 层剪枝校准选样 — 第一轮 CPU 筛查

**状态：开发诊断**（非论文结论）。未启动 GPU 打分、探针或闭环；未影响 LIBERO-plus 评测。

## 1. 证据链与本次增量

| 已有结论（只读引用） | 含义 |
|---|---|
| `care_libero.md` §9 E1 | **固定 480 帧**后，α **加权**不改变 Drop-k（J(CARE, uniform)=1.0）。 |
| `care_libero.md` §10 D/S | 改时间戳（突变/平滑）可改 Drop，但**无统一方向**，且 Taylor 经营点不动。 |
| 本实验 | **新变量**：在**同一 480 池**上，用 **40/80/120 帧**子集 + **等权**层分数（不乘 α）。 |

选样定义与 NFE 前瞻一致：`α`、`G`、`α_event`（H=10、λ=1、固定示范差分白化），见 `care_event_action_expert_nfe.md` §2；实现复用 `dev_nfe_rules_cpu_v2` 的 `select_per_task_k` / `random_group_k` / `g_only_group_k`。

## 2. 数据核对

| 项目 | 结果 |
|---|---|
| 分数目录 | `results/stage2_care_v2/scores/` — **9/9** 文件齐全 |
| 形状 | π0.5 **480×18**，SmolVLA **480×16**，Molmo **480×36** |
| `shared.pt` | **480** 条 `records`，与分数行一一对应 |
| 对齐 | `global_index` ↔ `lerobot/libero` 表 **task_index / episode_index 一致**（`alignment_ok=true`） |
| 估计器 | 全部为 **V2 逐帧** `score_frame.py`（未混用 Stage-1 batch 梯度平方） |

**说明**：候选池是 CARE v2 的 **40×12=480** 帧，与 Smol/Molmo 历史上 **32 帧**校准不是同一集合；**不能**用「40 帧≈旧法全池」作省算力论断。

## 3. 实验设置（冻结）

- **策略**：Random、α、G-only、CARE_event（`alpha_event`）。
- **预算**：40 / 80 / 120 帧（每任务 **1 / 2 / 3** 帧）。
- **Random / G-only**：种子 **0–19**；G-only 仅在事件内随机，不用 α 作第二排序键。
- **剪枝**：π0.5 **Drop-6**，Smol **Drop-4**，Molmo **Drop-16**；L0 规则同 `protocol.STAGE1`。
- **参考集**：同估计器 **480 帧等权** Drop（非真值，仅 Jaccard 参照）。

产物：`summary.json`、`details.csv`（1134 行）、`drop_frequency.csv`、`selections/*.json`。

## 4. 核心结果（开发诊断）

### 4.1 CARE_event 是否改变删除集合？

相对 **α（同预算）**，CARE_event 在 **6/27** 个（模型×方法×预算）单元格上 drop **不同**（其余与 α 完全相同）：

| 差异单元 | α drop 要点 | CARE_event drop 要点 |
|---|---|---|
| **π0.5 Fisher，预算 40** | Stage-1 `[1,13–17]` | E1 全池 uniform `[2,13–17]`（**非** Stage-1） |
| π0.5 GateProbe，40 / 80 | 与 CARE 不同（见 CSV） | 与 α 不同 |
| Molmo Taylor，**80** | `[1–4,24–35]`（Stage-1） | 另一 16 层集（J(full480)=0.88） |
| Molmo GateProbe，40 / 80 | 与 CARE 不同 | 与 α 不同 |

相对 **Random（20 种子）**，CARE_event 在多数非 Taylor 经营点上 **与所有 random 种子都不同**；Taylor 三栈在预算 120 时常与 **全部 random 相同**（random 也只有 1 种 drop）。

### 4.2 相对 Random / G-only：是否更少「已知不良」或更早命中可靠集？

**π0.5 Fisher-6（Stage-1 经营点）— 唯一有明显筛查信号的方法：**

| 预算 | CARE_event | α | Random 命中 Stage-1 比例 | G-only 命中 Stage-1 | Random 命中 E1-bad `[2,13–17]` |
|---:|---|---|---:|---:|---:|
| 40 | E1-bad 集 | **Stage-1** | 25% | 30% | **50%** |
| 80 | **Stage-1** | Stage-1 | 45% | 30% | 60% |
| 120 | **Stage-1** | Stage-1 | **10%** | 20% | **75%** |

解读（诊断级）：

- **k≥2（80/120 帧）**：CARE_event 与 **α 相同**，且 **稳定等于 Stage-1 Fisher-6**；Random/G-only **多数种子**仍落在 E1 全池 uniform 或更差集合上。
- **k=1（40 帧）**：CARE_event **反而**选到 E1-bad 集，**α 更靠近 Stage-1** — 事件排序在「每任务 1 帧」时**不**优于 α。

**Taylor 经营点（π0.5 / Smol / Molmo）**：CARE_event ≡ α，且在所有预算下 **J(Stage-1)=1.0**（Smol/Molmo/π0.5 Taylor）；Random 在 π0.5 Taylor 上亦常命中 Stage-1。**此处 CARE_event 无额外选层价值**（与 E1「换权重无用」一致，但原因是 **Taylor 排序对子集不敏感**）。

**Molmo Fisher-16**：CARE_event 与 α 始终一致，J(Stage-1)=**0.88**（等于 **480 全池 Fisher uniform**，即历史上翻向 Taylor-16 那类集合），**无法**靠事件选样在 CPU 上回到 Stage-1 Fisher 悬崖集。

**GateProbe**：三种策略 drop 高度分散（random 可达 12–16 种/预算）；CARE_event **无**历史闭环标签命中，**不宜**据此推广。

### 4.3 是否只发生在个别方法？

**是。** 信号集中在 **π0.5 + Fisher**；Taylor 全栈无效差异；Molmo/Smol 的 Fisher/GateProbe 多与 **v2 全池 uniform** 锁定，CARE_event **不**改变相对 α 的结论。

### 4.4 与历史闭环的可比性

- `details.csv` 中 `historical_label` **仅**在 drop **精确匹配** `analyze_ds.KNOWN` 时填写（如 S1 Fisher-6 97.75、S1 Taylor-16 98.10）。
- **更接近全池 Jaccard ≠ 更优闭环**；例如 π0.5 Fisher 预算 40 的 CARE_event J(full480)=1.0 但 **不是** Stage-1。
- 旧 32 帧 Smol/Molmo 闭环 **NA**，未写入可比行。

## 5. 结论（开发诊断）

1. **CARE_event 在层剪枝筛查中并非普适改进**：对 **Taylor 经营点**与 **Molmo Fisher** 子集，与 α 或全池 uniform 无异。
2. **唯一值得继续追问的切口**：**π0.5 Fisher-6**，预算 **80 或 120 帧**（k=2/3），CARE_event/α **确定性**得到 Stage-1 集，而 Random/G-only **大量**落在 E1 uniform 次优集 `[2,13–17]`。
3. **反例**：预算 **40** 时 CARE_event **劣于 α**（选到 E1-bad），说明 **k=1 不适用**事件优先策略（与 NFE 文档「每任务需多条」精神一致，但是层剪枝语境下的新观察）。
4. **不能**从本 CPU 表推出闭环 SR；仅能支持「是否值得开 1–2 个对照 eval」。

## 6. 建议的最小 GPU 验证（未执行，供审阅后决定）

| 优先级 | 配置 | 对照 | 目的 |
|---:|---|---|---|
| **1** | π0.5 **Fisher Drop-6**，预算 **80 帧**，CARE_event 选样 | 同预算 **Random 种子 0**（或 20 种子中未命中 S1 的多数集） | 检验「CPU 上更易命中 S1 层集」是否转化为 **闭环差异** |
| **2** | π0.5 **Fisher Drop-6**，预算 **40 帧**，**α vs CARE_event** | 两删除集 `[1,13–17]` vs `[2,13–17]` | 验证 k=1 时事件选样是否 **系统性更差**（CPU 已指向 α 更优） |

**不建议**为本轮筛查开评：三栈 **Taylor**（已与 Stage-1 一致）、**Molmo Fisher**（CPU 无 Stage-1 信号）、**GateProbe**（集合过于分散且无历史锚点）。

---

生成：`scripts/stage2_care_v2/layer_calibration_screening.py`  
目录：`results/stage2_care_v2/layer_calibration_screening/`
