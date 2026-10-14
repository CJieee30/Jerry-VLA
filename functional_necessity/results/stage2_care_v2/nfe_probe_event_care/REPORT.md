# CARE 夹爪事件增强（alpha_event）离线诊断

本轮仅复用 CARE v2 `shared.pt` 与 SmolVLA / π0.5 离线 NFE 探针 `frames.csv`。**动作开合不一致**指相对同帧 NFE=10 参照的执行端映射差异，不是闭环失败率。

## 公式

`G=1` ⇔ 未来 H=10 示范窗口内夹爪开合切换（与探针 `in_grip_switch` 一致）。
`alpha_event = (alpha + 1.0·G) / mean(alpha + 1.0·G)`。

探针 G 与重算一致率：**100.0%**（0 处不一致）。
边界「前一时刻→窗口起点」切换：**5** 条（主对照仍用窗口内 G）。

## 同预算选样（每任务 3/12，共 120）

### smolvla

**noise 0**

| group | grip窗口 | tail捕获/总数 | tail召回 | grip偏差NFE1 | grip偏差NFE2 | arm top5% | RMSE mean | RMSE P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 19 | 12/36 | 33.3% | 3/6 | 1/2 | 8/24 | 0.0964 | 0.1740 |
| care_alpha | 34 | 19/36 | 52.8% | 4/6 | 1/2 | 15/24 | 0.1129 | 0.1855 |
| care_event | 55 | 22/36 | 61.1% | 6/6 | 2/2 | 14/24 | 0.1124 | 0.1855 |

**noise 1**

| group | grip窗口 | tail捕获/总数 | tail召回 | grip偏差NFE1 | grip偏差NFE2 | arm top5% | RMSE mean | RMSE P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 19 | 14/37 | 37.8% | 5/9 | 1/1 | 8/24 | 0.0997 | 0.2122 |
| care_alpha | 34 | 17/37 | 45.9% | 7/9 | 1/1 | 15/24 | 0.1156 | 0.2149 |
| care_event | 55 | 22/37 | 59.5% | 8/9 | 1/1 | 13/24 | 0.1147 | 0.2122 |

全部 G=1 窗口（n=56）动作开合不一致率：NFE=1 **9.8%**，NFE=2 **2.7%**。

### pi05

**noise 0**

| group | grip窗口 | tail捕获/总数 | tail召回 | grip偏差NFE1 | grip偏差NFE2 | arm top5% | RMSE mean | RMSE P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 19 | 11/36 | 30.6% | 3/7 | 1/4 | 6/24 | 0.0604 | 0.1351 |
| care_alpha | 34 | 20/36 | 55.6% | 5/7 | 3/4 | 14/24 | 0.0773 | 0.1661 |
| care_event | 55 | 21/36 | 58.3% | 7/7 | 4/4 | 13/24 | 0.0780 | 0.1935 |

**noise 1**

| group | grip窗口 | tail捕获/总数 | tail召回 | grip偏差NFE1 | grip偏差NFE2 | arm top5% | RMSE mean | RMSE P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 19 | 12/35 | 34.3% | 2/3 | 1/2 | 7/24 | 0.0613 | 0.1410 |
| care_alpha | 34 | 18/35 | 51.4% | 2/3 | 1/2 | 13/24 | 0.0773 | 0.1595 |
| care_event | 55 | 17/35 | 48.6% | 3/3 | 2/2 | 12/24 | 0.0758 | 0.1720 |

全部 G=1 窗口（n=56）动作开合不一致率：NFE=1 **8.9%**，NFE=2 **5.4%**。

## 选样变化（event vs 原 CARE，SmolVLA noise0 指标）

新增 21：[26, 31, 66, 76, 77, 124, 154, 162, 225, 253, 270, 274]
移除 21：[14, 44, 51, 61, 109, 125, 173, 197, 218, 276, 284, 289]

**124、381 在新增列表中**（中等 α 的夹爪切换帧被 event 抬升）；**361、447 在移除列表中**（原高 α 非 G 帧被挤出）。完整 21 对见 `selection_changes.json`。

## 五个问题（简答）

事件 CARE 相对原 CARE：**夹爪动作偏差（NFE=1）捕获数在多数格上增加或持平**（详情见 summary.json）。G=1 窗口入选数从原 CARE 的 34 增至 event 的 55（noise0 SmolVLA），有利于减少「夹爪切换附近低 NFE 偏差」漏检。

手臂误差 top5% 捕获：event CARE 可能 **略低于** 原 CARE（见各表 arm top5% 列），因预算固定为 120 帧，事件加分会用高 α_event 的夹爪帧替换部分纯高 α 手臂突变帧。

两模型、两噪声下 **SmolVLA 与 π0.5 noise0** 上 tail 召回与夹爪偏差捕获均随 event 提升；**π0.5 noise1** 上 tail 召回 event 略低于原 CARE（48.6% vs 51.4%），但夹爪偏差仍从 2/3 增至 3/3。

高分 α_event 视频在候选池内通常对应 **G=1 或较高 α**，可见更多夹爪轨迹变化；但演示视频 **不能** 单独证明低 NFE 策略行为，也不能代替动作误差表。

建议：**保留原 CARE α 作主排序**，夹爪事件作 **+1 固定加分** 的并行候选（或报告双轨）；若目标是减少夹爪相关动作偏差漏检，event 版值得作诊断选样；若优先覆盖手臂极端误差，保留原 CARE。

## 曲线与视频

- `curves/demo_window_shared_124.png`
- `curves/demo_window_shared_381.png`
- `curves/demo_window_shared_361.png`
- `curves/demo_window_shared_447.png`
- 视频索引：[index.html](index.html)
- 原 α 预览：[../preview_smolvla_timestamps/index.html](../preview_smolvla_timestamps/index.html)
