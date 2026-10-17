# MolmoAct2 前瞻校准池层剪枝选样（v1 rev2）— 正式结果

**开发诊断** · 子集选样后**等权**层分数（非旧 α 加权实验）。**本轮未启动闭环。**

协议：`layer_calibration_molmo_v1_rev2` · 池 `prospective_v1_20261015` **400** 帧，与 CARE v2 开发池 episode 重叠 **0**。

## 运行记录（2026-10-17 UTC）

| 阶段 | GPU | 结果 |
|------|-----|------|
| 冒烟 `verify_molmo_score_smoke.py --with-gate n=2` | 0（整卡） | **IGIA 未通过**（`passed=false`，`max_rel_err≈4.7e-3` > `rtol=1e-3`）；Fisher/GateProbe 在容差内；见 `score_smoke_verify.json` |
| IGIA 路径复核 | 0 | `diagnostics_rev2_igia_20261017/`（2 样本×2 重复 + b=80 原实现选层核实） |
| `score_molmo_prospective_layers.py`（400×IGIA/Fisher/GateProbe） | 0 | **完成** ~02:08–02:43 UTC |
| `score_molmo_prospective_taylor_groups.py`（b=80 四策略） | 2 | **完成** ~02:08–02:33 UTC |
| `analyze_layer_calibration_molmo_v1.py` | CPU | **完成** |

显存：冒烟峰值 **`max_allocated` ≈ 17.5 GiB**（`score_smoke_verify.json` → `mem_after_ref.max_allocated_bytes` ≈ 1.88×10¹⁰ B；checkpoint 有效）。共享卡 OOM 尝试见 `summary.json` 历史 `gpu_attempts`。

## 问题 1：CARE_event vs Random（主预算 80）

**Random 的 20 个种子**：在同一前瞻池上的**重复选样**（非 20 个独立校准池）。历史 **SR%** 来自既有闭环记录，**非**本轮新评测。

### Fisher（b=80 Random，20 种子）

- **CARE_event / α-only**：Drop-16 = `[1,2,3,4,24–35]`，与 **400 帧等权 Fisher** 及 **S1 Taylor-16**（98.10%）**一致**。
- **Random**：**14/20** 种子 → Taylor-16 层集；**6/20** → **S1 Fisher/IGIA-16 悬崖** `[1–5,25–35]`（历史 SR **72.75%**）。

### IGIA（b=80 Random，20 种子）

- **CARE_event / α-only**：与 Taylor-16 层集 **一致**（与 full-400 等权 IGIA 相同）。
- **Random**：**5/20** → Taylor-16；**15/20** → 悬崖 72.75% 层集（IGIA 下 Random 更易抽到悬崖模态）。

### CARE 多预算（Fisher / IGIA）

- **b=40 / 80 / 120** 的 **CARE_event** 均选中 **Taylor-16（98.10%）** 层集 → 支持解读：**小预算子集选样可避免 Random 常命中的已知高损失（悬崖）层集**；不宜再写“相对 Random 没有改善”的笼统结论。

### GateProbe（补充，敏感性）

- CARE 与 full-400 的 Jaccard **0.88**；Random **6** 种 drop、**100%** 种子与 CARE 不同 → 仅作**选样敏感性**报告，不作稳定选层主张。

### Taylor（b=80 分组）

- 仅 **Random / G-only 各 1 个种子**（seed=0）；**不能**据此证明 Taylor 在任意种子下普遍稳定。
- CARE / α / Random(seed=0) 与 S1 Taylor-16 对齐（J=1.00）；G-only(seed=0) J=0.88（含 L23）。

### Drop-16 模态与 IGIA 复核

- full-400 等权 Fisher/IGIA → **Taylor-16**；Random b=80 常见另一模态为 **悬崖**（72.75% 历史层集，与 Taylor-16 差 **L5 vs L24** 等）。
- 原 `measure_grad` IGIA 在 **b=80** 上对 **CARE_event** 与 **Random seed=0** 重算，Drop-16 与 rev2 分析 **一致**（`diagnostics_rev2_igia_20261017/igia_b80_ref_selection.json`）。
- 逐帧 IGIA 两次重复：**非逐层 bitwise 相同**（`igia_path_diagnostic.json`）；冒烟 `passed=false` 记录**保留**。

## 问题 2：α-only / G-only 能否替代 CARE？

| 方法 | b=80 CARE vs α-only | 说明 |
|------|---------------------|------|
| Fisher | **层集相同** | 尚未证明 **G** 相对 α 的额外收益 |
| IGIA | **层集相同** | 同上 |
| GateProbe | 不同 | G-only / Random 分歧更大 |

G-only 在 Fisher/IGIA b=80 下可产生与 CARE **不同** 的 drop（见 `summary.json` 闭环候选），**未知 SR**，本轮**未**启动闭环。

## 问题 3：IGIA 实现与是否重算

- 冒烟与原实现 IGIA 在 2 帧上为 **小幅数值偏差**（相对误差 ~10⁻³ 量级）；同帧两次逐帧 IGIA **非 bitwise 相同**（见诊断 JSON）。
- **选层结论**基于已落盘的 400 帧逐帧分数 + b=80 原实现子集核实；**不需要**重跑 Fisher、Taylor 或全 400 帧 IGIA。

## 问题 4：历史复用 vs 闭环候选（未启动）

| 历史标签 | SR% | 本轮关系 |
|----------|-----|----------|
| S1 Taylor-16 | 98.10 | Fisher/IGIA **CARE b=40/80/120**、full-400 等权、Taylor 三组 CARE/α/Random₀ → **可复用层集** |
| S1 Fisher/IGIA-16 | 72.75 | Random 部分种子命中；**非** CARE 目标 |
| S1 GateProbe-16 | 87.00 | GateProbe CARE 不同 |

闭环候选（去重 2 个，**本轮未跑**）：见 `summary.json` → `closed_loop_recommendations`。

## 文件

- `scores/molmoact2_{igia,fisher,gateprobe}.pt`（400×36）
- `taylor_groups/taylor_*_b80*.pt`
- `details.csv`, `drop_frequency.csv`, `selections/`
- `score_smoke_verify.json`（原始冒烟，含 `passed=false`）
- `diagnostics_rev2_igia_20261017/`（本轮 IGIA 复核）
- `rev2_launch.log`, `score_run.log`, `taylor_group_run.log`

生成：`analyze_layer_calibration_molmo_v1.py` · 复核说明：`diagnostics_rev2_igia_20261017/REVIEW_CONCLUSION.md`。
