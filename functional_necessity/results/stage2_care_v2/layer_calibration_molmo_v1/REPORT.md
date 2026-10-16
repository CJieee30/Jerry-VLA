# MolmoAct2 前瞻校准池层剪枝选样（v1）

**状态：开发诊断 · 打分待独占 GPU**（2026-10-16）

## 本轮完成项

| 步骤 | 状态 | 说明 |
|------|------|------|
| 阅读协议 / 与旧池隔离 | ✅ | `prospective_v1_20261015`；`excluded_dev_episodes=600` |
| Episode 重叠核查 | ✅ | 前瞻 400 ep 与 CARE v2 `shared.pt` 480 ep **交集 0** |
| 固定 (noise, time) 校准张量 | ✅ | `calib_molmoact2.pt`，种子基 `20261015 + pool_index` |
| 四指标逐帧 GPU 打分 | ⏸ | 各卡已有 ~10GB 占用，单步反传 OOM（见 `score_smoke.log`） |
| 等预算 CPU 筛查 | ⏸ | 依赖 `scores/molmoact2_*.pt` |

## 数据与口径

- **身份清单**：`nfe_protocol/prospective_v1_20261015/01_split/calibration_pool.json`（400 obs，40 任务 × **10** 候选/任务）。
- **与旧层剪枝开发池**：CARE v2 为 40×12=480 帧、不同 episode 集合；**无 episode 重叠**（脚本 `verify_no_dev_overlap()`）。
- **与旧 α 加权 IGIA 实验**：旧实验在 **32 帧**上 `α·Σg²`；本轮为 **子集选样 + 等权 Σg²/Σ(wg)²/…**，不可混表。
- **损失与预处理**：与 `molmoact2_libero/score.py`、V2 `score_frame.py` 同一路径（flow-matching joint loss，VLM 块梯度）。

## 实验设置（冻结，待打分后执行 analyze）

- 策略：Random、α-only、G-only、CARE_event（`dev_nfe_rules_cpu_v2` 规则；α 策略名 `alpha`）。
- 预算：**80** 主（每任务 2 帧）；补充 40 / 120。
- Random / G-only：种子 **0–19**。
- 聚合：选中行 **等权 mean** 层分数 → Drop-16，**排除 L0**。
- 指标定义（单次 backward 同时写出）：
  - **IGIA**：Σ g²
  - **Fisher**：Σ (g·θ)²
  - **Taylor**：Σ |g·θ|
  - **GateProbe**：现有边界内积实现

## GPU 占用与运行命令

当前机器上 GPU0/1/2 均被 LIBERO-plus 噪声评测占用（各约 **9.8 GiB**），在 GPU2 上试跑 **1** 个样本反传即 **OOM**（与评测进程共享 24GB）。

**请在独占一张 GPU（建议空闲 ≥14 GiB）时执行：**

```bash
cd /home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/stage2_care_v2
# 校准已写好可跳过 dump
CUDA_VISIBLE_DEVICES=<GPU_ID> bash RUN_LAYER_CALIB_MOLMO_V1.sh <GPU_ID>
```

或分步：

```bash
python dump_calib_molmo_prospective.py          # 已完成
CUDA_VISIBLE_DEVICES=<GPU_ID> python score_molmo_prospective_layers.py --device cuda:0
python analyze_layer_calibration_molmo_v1.py    # 写 details.csv / 本 REPORT 数值段
```

产物目录（不覆盖 `stage2_care_v2/scores/` 旧 480 帧缓存）：

`results/stage2_care_v2/layer_calibration_molmo_v1/`

## 历史闭环映射（待 details 填写）

匹配规则：**checkpoint + Drop-16 层集 + keep-KV + NFE=10 + 评测协议** 一致才可标「可比性已核实」。

| 层集（Drop-16） | 历史 SR | 可比性 |
|-----------------|--------:|--------|
| Taylor `[1–4,24–35]` | 98.10% (2000 ep) | **可比性已核实** — 与策略无关，命中则**不重跑** |
| Fisher `[1–4,5,25–35]` | 72.75% | 可比性已核实 |
| IGIA `[1–4,5,25–35]` | 72.75% | 可比性已核实（与 Fisher-16 同集） |
| GateProbe `[2–4,6,24–35]` | 87.00% | 可比性已核实 |
| CARE v2 480 均匀 Fisher | CPU 参考 (=Taylor 层) | 历史参考，非本池 |

未知层集：**尚未评测**；不得因 Fisher 名称或 Jaccard 高而假定低 SR。

## 待 analyze 填写的四个问题（提纲）

1. **CARE_event vs Random（b=80）**：按 `method` 比较 `drop_frequency.csv` 中 CARE 与 20 种子分布；重点 **Fisher**（悬崖），Taylor 作对照。
2. **α / G-only 可替代性**：CARE 与 α 是否同 drop；G-only 不同比例。
3. **IGIA**：是否与 Fisher 分叉（旧 CARE 加权未翻悬崖）；是否在事件选样下更接近 Taylor 层集。
4. **复用 vs 补跑**：层集等于 Taylor-16 → 复用 98.10%；等于已知 Fisher/IGIA 悬崖 → 复用 72.75%；否则最多 **2** 个新闭环候选（脚本 `recommend_closed_loop`，本轮**不启动**）。

## 脚本索引

- `prospective_molmo_layer_common.py` — 路径、重叠检查、历史映射
- `dump_calib_molmo_prospective.py` — CPU 写 `calib_molmoact2.pt`
- `score_molmo_prospective_layers.py` — GPU 四指标
- `analyze_layer_calibration_molmo_v1.py` — CPU 筛查 + 更新本报告数值段
- `RUN_LAYER_CALIB_MOLMO_V1.sh` — 一键（需 GPU 参数）

---

*NFE 前瞻协议与探针结果未修改；本目录独立于 `nfe_protocol/prospective_v1_20261015/`。*
