# MolmoAct2 前瞻校准池层剪枝选样（v1 rev2）— 正式结果

**开发诊断** · 子集选样后**等权**层分数（非旧 α 加权实验）。**本轮未启动闭环。**

协议：`layer_calibration_molmo_v1_rev2` · 池 `prospective_v1_20261015` **400** 帧，与 CARE v2 开发池 episode 重叠 **0**。

## 运行记录（2026-10-17 UTC）

| 阶段 | GPU | 结果 |
|------|-----|------|
| 冒烟 `verify_molmo_score_smoke.py --with-gate n=2` | 0（整卡） | **通过**；IGIA/Fisher 与 `score.py` 一致；GateProbe 隐状态 `requires_grad` 有效 |
| `score_molmo_prospective_layers.py`（400×IGIA/Fisher/GateProbe） | 0 | **完成** ~02:08–02:43 UTC |
| `score_molmo_prospective_taylor_groups.py`（b=80 四策略） | 2 | **完成** ~02:08–02:33 UTC |
| `analyze_layer_calibration_molmo_v1.py` | CPU | **完成** |

显存：整卡冒烟峰值约 **12.0 GiB** allocated（`score_smoke_verify.json`，checkpoint 有效）。与此前共享卡 OOM 对比见 `summary.json` 历史 `gpu_attempts`。

## 问题 1：CARE_event vs Random（主预算 80）

### Fisher

- **CARE_event** Drop-16 = `[1,2,3,4,24–35]`，与 **400 帧等权 Fisher** 及 **S1 Taylor-16** 层集 **完全一致**（J=1.00）。
- **Random**（20 种子）：仅 **2** 种 drop；约 **30%** 种子与 CARE 不同。另一模态为 **S1 Fisher/IGIA-16 悬崖层集** `[1–5,25–35]`（历史 SR **72.75%**，n=2000，keep-KV，NFE=10）。
- **解读**：在 Fisher 下，CARE_event **未**相对 Random 改善选层——CARE 与全池/ Taylor-16 同集；Random 有时抽到已知悬崖集，但并非“更优选层”。

### IGIA

- **CARE_event** 与全池 / S1 Taylor-16 层集 **一致**（J=1.00）。
- **Random**：**2** 种 drop，约 **75%** 种子与 CARE 不同（含悬崖集模态）。
- **解读**：IGIA 下 Random 分歧更大，但 CARE 仍锚定在 Taylor-16 层集，**未见**相对 Random 的系统性“更好” drop。

### GateProbe（补充）

- CARE J(full400)=**0.88**；Random **6** 种 drop，**100%** 种子与 CARE 不同 → 选样对 GateProbe 最敏感。

### Taylor（b=80 分组，无 full-400 矩阵）

- CARE / α / Random(seed=0)：**J(S1 Taylor-16)=1.00**，与历史 **98.10%** 层集一致 → **闭环可复用 S1 Taylor-16**，不必重跑。
- G-only(seed=0)：J(S1)=**0.88**，drop 含 **L23**，与 S1 不同。

## 问题 2：α-only / G-only 能否替代 CARE？

| 方法 | b=80 CARE vs α-only | G-only 与 CARE 不同比例 |
|------|---------------------|-------------------------|
| Fisher | **相同** | **30%** |
| IGIA | **相同** | **10%** |
| GateProbe | 不同 | **45%** |

**结论**：Fisher/IGIA 下 **α-only 与 CARE_event 完全等价**（主/补预算均同）。G-only 在 Fisher b=80 下可产生 **与 CARE 不同** 的 drop（见闭环候选），但 **不能**说优于 CARE——未知 SR。

## 问题 3：原 Taylor 稳定性

四策略分组 Taylor 与 S1 Taylor-16 对齐良好，除 **G-only seed=0** 外均为 **1.00** 或 **0.88**（L23 替换 L4）。**无** full-400 Taylor 分数 → `jaccard_vs_full400_equal` / `full400_equal_drop` 在 Taylor 行中为 **NA**（见 `details.csv`）。

## 问题 4：历史复用 vs 闭环候选（最多 2 个，未启动）

| 历史标签 | SR% | 算子 / 协议 | 本轮关系 |
|----------|-----|-------------|----------|
| S1 Taylor-16 | 98.10 | Taylor, keep-KV, NFE=10, n=2000 | Fisher/IGIA/CARE **b=80** 及 Taylor 三组 → **层集相同，可复用** |
| S1 Fisher-16 / IGIA-16 | 72.75 | 同上悬崖层集 | Random 部分种子命中，**非** CARE 主推 |
| S1 GateProbe-16 | 87.00 | GateProbe-16 | GateProbe CARE drop 不同 |

**建议补闭环（去重后 2 个，未知性能）**：

1. **Fisher + G-only**，b=80，drop `[1,2,4,23,24–35]`
2. **IGIA + G-only**，b=80，drop `[1,2,3,23,24–35]`

## 文件

- `scores/molmoact2_{igia,fisher,gateprobe}.pt`（400×36）
- `taylor_groups/taylor_*_b80*.pt`
- `details.csv`, `drop_frequency.csv`, `selections/`
- `rev2_launch.log`, `score_smoke_verify.json`

生成：`analyze_layer_calibration_molmo_v1.py`。
