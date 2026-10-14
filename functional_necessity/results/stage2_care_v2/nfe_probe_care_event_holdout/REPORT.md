# CARE_event 离线 NFE 验证（校准回放 + 独立 holdout）

本报告比较的是 **低 NFE 相对 NFE=10 的动作 chunk 偏差**，不是 LIBERO 闭环成功率。

## 固定方案

- `alpha_event = (α + 1.0·G) / mean_cal(α + 1.0·G)`，H=10。
- 校准集：480 帧，每任务 α_event **前三** → `care_event`（120 帧）。
- Holdout：120 帧，种子 4242，排除校准涉及的全部 episode。

## 1. 校准集（复用已有探针，480 帧）

### smolvla
**noise 0**

| 组 | n | 尾部召回 | 夹爪NFE1捕获 | 夹爪NFE2捕获 | 手臂P95捕获 |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 120 | 33.3% | 3/6 | 1/2 | 8/24 |
| care_alpha | 120 | 52.8% | 4/6 | 1/2 | 15/24 |
| care_event | 120 | 61.1% | 6/6 | 2/2 | 14/24 |

**noise 1**

| 组 | n | 尾部召回 | 夹爪NFE1捕获 | 夹爪NFE2捕获 | 手臂P95捕获 |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 120 | 37.8% | 5/9 | 1/1 | 8/24 |
| care_alpha | 120 | 45.9% | 7/9 | 1/1 | 15/24 |
| care_event | 120 | 59.5% | 8/9 | 1/1 | 13/24 |

### pi05
**noise 0**

| 组 | n | 尾部召回 | 夹爪NFE1捕获 | 夹爪NFE2捕获 | 手臂P95捕获 |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 120 | 30.6% | 3/7 | 1/4 | 6/24 |
| care_alpha | 120 | 55.6% | 5/7 | 3/4 | 14/24 |
| care_event | 120 | 58.3% | 7/7 | 4/4 | 13/24 |

**noise 1**

| 组 | n | 尾部召回 | 夹爪NFE1捕获 | 夹爪NFE2捕获 | 手臂P95捕获 |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 120 | 34.3% | 2/3 | 1/2 | 7/24 |
| care_alpha | 120 | 51.4% | 2/3 | 1/2 | 13/24 |
| care_event | 120 | 48.6% | 3/3 | 2/2 | 12/24 |

## 2. 独立 holdout（120 帧）

- Holdout 窗口内 **G=1 事件数**：16 / 120
- 尾部阈值：来自 **校准** SmolVLA NFE=1 的 P5/P95（holdout 上不重算）。

### smolvla
- 全 holdout 上 tail_bad=10, grip_dev_nfe1=1, grip_dev_nfe2=1, arm_bad=4
- 异常曲线：`curves/smolvla/`（5 张）

### pi05
- 全 holdout 上 tail_bad=1, grip_dev_nfe1=0, grip_dev_nfe2=0, arm_bad=1
- 异常曲线：`curves/pi05/`（5 张）

## 3. 问题摘要

### Q1 care_event 相对随机 / 原 CARE 多覆盖了什么？
- 见校准表：`care_event` 通常在 **G 事件窗口** 与 **夹爪 NFE1 不一致** 上高于 `care_alpha`/`random`；
  手臂 RMSE 尾部依赖任务分布，高 α_event 不保证手臂极端误差。

### Q2 Holdout 是否出现校准未见的严重偏差？
- 对比 holdout 全集 tail_bad 与校准组已捕获的 shared_index 集合（不同 universe）；
  若 holdout tail 样本的 α_event 偏低，说明事件选样未外推覆盖。

### Q3 闭环建议（非 SR 保证）
- **π0.5 NFE=1**：仅当 holdout 上夹爪/尾部仍可控时建议小预算闭环；
- **SmolVLA NFE=2**：校准中 NFE2 对 NFE1 尾部有恢复时优先于 NFE1 试点。

## 闭环试点命令（本轮不执行）

```bash
# π0.5：LIBERO-plus / 正式 eval 脚本，Drop-6，num_inference_steps=1，先 20–40 回合 sanity
# SmolVLA：num_steps=2 对照 10；若 holdout 上 NFE2 仍 tail_bad，勿直接上 400 回合
```

## 产物路径

- `holdout_manifest.json` / `holdout_shared.pt`：独立 120 时间戳
- `smolvla/`、`pi05/`：`frames.csv`、`actions.npz`、`correctness.json`
- `curves/{model}/holdout_*.png`：holdout 上最差 5 例（noise0, NFE1）

