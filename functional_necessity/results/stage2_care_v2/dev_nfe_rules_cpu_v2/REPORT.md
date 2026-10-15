# NFE 选步规则 CPU 分析（开发结果，非冻结协议）

空间：LIBERO 归一化；10 步×7 维；E_N = 对观测与两噪声平均的 RMSE(vs 同噪声 NFE10)。
规则：绝对 E_N≤0.085（敏感性 (0.08, 0.09)）；绝对改善 E_N−E_{N+1}≤0.03；相对改善 (E_N−E_{N+1})/E_N≤0.375。仅相邻候选 NFE；误差不单调标 non_monotonic。

## 1. 三模型全池指标与推荐

### pi05
| NFE | E_N | cos mean |
|---|---|---|
| 1 | 0.0597 | 0.99716 |
| 2 | 0.0407 | 0.99873 |

| 相邻 | E_N−E_{N+1} | 非单调? | 输出间 mean RMSE(N,N+1) |
|---|---|---|---|
| 1-2 | 0.0191 | False | 0.0289 |
NFE10 两噪声夹爪基准（观测级）：{'comparisons_with_grip_disagree': 8, 'comparisons_total': 480, 'unique_obs_with_disagree': 8, 'rate_per_obs': 0.016666666666666666}
推荐（开发）：
- `abs_0.085` → **1**
- `abs_0.08` → **1**
- `abs_0.09` → **1**
- `abs_improve_only` → **1**
- `rel_improve_only` → **1**
- `all_three_abs085` → **1**

### smolvla
| NFE | E_N | cos mean |
|---|---|---|
| 1 | 0.0936 | 0.99394 |
| 2 | 0.0546 | 0.99796 |
| 3 | 0.0398 | 0.99891 |

| 相邻 | E_N−E_{N+1} | 非单调? | 输出间 mean RMSE(N,N+1) |
|---|---|---|---|
| 1-2 | 0.0390 | False | 0.0608 |
| 2-3 | 0.0148 | False | 0.0214 |
NFE10 两噪声夹爪基准（观测级）：{'comparisons_with_grip_disagree': 8, 'comparisons_total': 480, 'unique_obs_with_disagree': 8, 'rate_per_obs': 0.016666666666666666}
推荐（开发）：
- `abs_0.085` → **2**
- `abs_0.08` → **2**
- `abs_0.09` → **2**
- `abs_improve_only` → **2**
- `rel_improve_only` → **2**
- `all_three_abs085` → **2**

### molmoact2
| NFE | E_N | cos mean |
|---|---|---|
| 1 | 0.0346 | 0.99903 |
| 2 | 0.0244 | 0.99951 |
| 3 | 0.0190 | 0.99973 |

| 相邻 | E_N−E_{N+1} | 非单调? | 输出间 mean RMSE(N,N+1) |
|---|---|---|---|
| 1-2 | 0.0102 | False | 0.0234 |
| 2-3 | 0.0054 | False | 0.0118 |
NFE10 两噪声夹爪基准（观测级）：{'comparisons_with_grip_disagree': 1, 'comparisons_total': 480, 'unique_obs_with_disagree': 1, 'rate_per_obs': 0.0020833333333333333}
推荐（开发）：
- `abs_0.085` → **1**
- `abs_0.08` → **1**
- `abs_0.09` → **1**
- `abs_improve_only` → **1**
- `rel_improve_only` → **1**
- `all_three_abs085` → **1**

## 2. 规则对照（全池，独立应用）

| model | abs≤0.08 | abs≤0.085 | abs≤0.09 | 仅绝对改善 | 仅相对改善 | 三者联合 |
|---|---|---|---|---|---|---|
| pi05 | 1 | 1 | 1 | 1 | 1 | 1 |
| smolvla | 2 | 2 | 2 | 2 | 2 | 2 |
| molmoact2 | 1 | 1 | 1 | 1 | 1 | 1 |

## 3. 选样策略与预算（联合规则 all_three_abs085；非闭环准确率）

Random/G-only：20 种子；α / CARE_event：确定性各 1 次。清单：`dev_nfe_rules_cpu_v2/selections/`。

### 推荐稳定性（k=3，all_three_abs085，random 20 种子）
| model | counts |
|---|---|
| pi05 | {1: 20} |
| smolvla | {2: 20} |
| molmoact2 | {1: 20} |

完整预算表见 `budget_comparison.json`。

## 4. 仍阻止正式冻结

- 正式协议未冻结；本报告阈值均为开发结果
- Molmo dev_validation_probe=pending；闭环导入未接真实 JSON
- holdout 前瞻 split/freeze 未完成
- 夹爪仅诊断，未设未经校准的硬门槛
- Pi 无 NFE3 探针，差分规则在 2→3 段为 NA

## 复现

```bash
cd functional_necessity/scripts/stage2_care_v2
export PYTHONPATH=".:../molmoact2_libero:../pi05_libero"
python dev_nfe_rules_cpu_v2.py
```

数据：`nfe_probe_{pi05,smolvla,molmoact2}/actions.npz`；转换见 `dev_three_model_screening/conversion_manifest.json`。
