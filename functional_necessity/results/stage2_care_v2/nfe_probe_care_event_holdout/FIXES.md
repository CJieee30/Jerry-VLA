# CARE_event holdout 统计修复说明

## 错误

1. `run_analysis()` 对 **π0.5 holdout** 误用 **SmolVLA 校准** `frames.csv` 计算 cosine P5 / RMSE P95，
   导致 π0.5 尾部数量严重偏低（例如 noise0 仅 1 例，应为 7）。
2. 报告将 `eval_selection(..., all_holdout)` 的 `tail_recall=1.0` 误解为 CARE 召回；
   实为对全集计数，无选样组对照。
3. 未分模型、分 NFE、分 noise 汇总；合并 noise 时未说明 240 次比较并重算 P95。

## 修复

- 各模型使用各自校准探针 `nfe_probe_{smolvla,pi05}/frames.csv` 的 NFE=1 阈值。
- Holdout 仅报告 **异常数量与比例**；身份键为 `global_index` / `episode+frame`。
- 忽略 holdout `frames.csv` 中无意义的 `in_top_*` 分组列。

## 修复前后（尾部 bad 数量，noise0 / noise1）

| 模型 | NFE | 修复前（π0.5 误阈） | 修复后 | 期望 |
| --- | ---: | --- | --- | --- |
| pi05 | 1 | ['1', '1'] | [7, 6] | [7, 6] |
| pi05 | 2 | ['?', '?'] | [2, 1] | [2, 1] |
| smolvla | 1 | [10, 12] | [10, 12] | [10, 12] |
| smolvla | 2 | [1, 1] | [1, 1] | [1, 1] |
| smolvla | 3 | [1, 1] | [1, 1] | [1, 1] |

核对：`ok=True`
与预期尾部/夹爪计数一致。

## 绘图修复（第二轮）

- `plot_holdout_anomalies()` 现仅对 `anomaly_rows` 中 **NFE=1** 的真实尾部绘图；
  不再按 RMSE 排序全 holdout（误含 π0.5 #93/noise0、#96/noise1 等）。
- 旧 `*_tail_nfe1.png` 移至 `curves/_backup_pre_plotfix/{model}/`。
- SmolVLA 异常图叠加 **NFE 1/2/3/10**；π0.5 为 **1/2/10**。
- 报告夹爪说明仅列 SmolVLA（#16/noise0、#48/noise1）。

