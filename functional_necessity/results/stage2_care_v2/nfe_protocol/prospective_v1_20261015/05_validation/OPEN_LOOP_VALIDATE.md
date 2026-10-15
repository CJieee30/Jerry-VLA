# 前瞻开环独立验证 — prospective_v1_20261015

校准推荐已冻结；下表为 **验证集** holdout 规则检验（非闭环）。正式 `validate` 待闭环导入。

## 验证集 E_N（相对 NFE=10，480 obs × 2 noise 均值）

| model | E@1 | E@2 | E@3 |
|---|---:|---:|---:|
| pi05 | 0.0629 | 0.0428 | 0.0321 |
| smolvla | 0.0971 | 0.0566 | 0.0413 |
| molmoact2 | 0.0358 | 0.0249 | 0.0193 |

## 校准选定 NFE 的 holdout 规则（各策略 screening NFE 相同则合并）

| model | rule | screening NFE | holdout | E@N | detail |
|---|---|---:|---|---:|---|
| pi05 | abs_0.085 | 1 | pass | 0.0629 | E_1=0.0629 |
| pi05 | abs_improve | 1 | pass | 0.0629 | delta=0.0201 |
| pi05 | rel_improve | 1 | pass | 0.0629 | rel=0.3193 |
| smolvla | abs_0.085 | 2 | pass | 0.0566 | E_2=0.0566 |
| smolvla | abs_improve | 2 | pass | 0.0566 | delta=0.0153 |
| smolvla | rel_improve | 2 | pass | 0.0566 | rel=0.2708 |
| molmoact2 | abs_0.085 | 1 | pass | 0.0358 | E_1=0.0358 |
| molmoact2 | abs_improve | 1 | pass | 0.0358 | delta=0.0109 |
| molmoact2 | rel_improve | 1 | pass | 0.0358 | rel=0.3042 |

## 推荐 NFE 诊断（验证集，相对 NFE=10）

| model | N* | grip disagree rate | arm median ratio | arm P95 ratio |
|---|---:|---:|---:|---:|
| pi05 | 1 | 0.0156 | 1.154 | 0.884 |
| smolvla | 2 | 0.0052 | 0.697 | 0.531 |
| molmoact2 | 1 | 0.0042 | 1.852 | 1.016 |
