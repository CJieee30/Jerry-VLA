# 前瞻协议冻结前收口（开发证据）

## 统一口径

- 模块：`scripts/stage2_care_v2/nfe_en_metrics.py` + `dev_metrics_common.py`
- **E_N**：LIBERO 归一化、H=10、7 维；每 (观测×噪声) RMSE vs 同噪声 NFE10，再平均
- **主规则**：E_N ≤ 0.085（0.08/0.09 仅敏感性）
- **绝对改善**：0 ≤ E_N−E_{N+1} ≤ 0.03（无相邻候选 → **不通过**，非默认通过）
- **相对改善**：0 ≤ (E_N−E_{N+1})/E_N ≤ 0.375（E_N=E_{N+1}=0 → rel=0）
- 夹爪/尾部：诊断 only；旧四门槛已从 `recommend` 移除

## 全池三规则（开发）

| model | abs≤0.085 | abs improve | rel improve |
|---|---|---|---|
| pi05 | 1 | 1 | 1 |
| smolvla | 2 | 2 | 2 |
| molmoact2 | 1 | 1 | 1 |

**复算结论未变**（与上轮 joint 一致；本轮严格 NA 后三规则在全池仍一致）。

## 预算对照（378 行）

- `three_rule_budget_table.csv`：每行独立列出 `rec_abs_0.085` / `rec_abs_improve` / `rec_rel_improve`
- 主预算 k=3（120 观测）：四策略在 random 20 种子下推荐一致 → **不能**表述为 CARE 提高“准确率”
- 40/80 消融：同表 `k_per_task`∈{1,2}；尾部/夹爪列见 JSON 内 `tail_nfe1` / `grip_nfe1`

## 端到端演练（开发）

- Run：`results/stage2_care_v2/nfe_protocol/dev_e2e_three_rules_v8/`
- 路径：freeze → select → seal-dev → recommend → validate（validate 在 recommend 落盘之后）
- `verify-ledger`：通过
- Molmo holdout validate：`pending`（配置明示，非阻塞冻结模板）

## 待冻结前瞻配置

- `scripts/stage2_care_v2/nfe_protocol_config.prospective_prefreeze.json`（`run_name: prospective_prefreeze_v1`）
- 入口：`nfe_protocol_v1.py`（正式 run **禁止** `--force-restage`）

## 复现

```bash
cd functional_necessity/scripts/stage2_care_v2
export PYTHONPATH=".:../molmoact2_libero:../pi05_libero"
python regenerate_three_rule_budget.py
PY=python nfe_protocol_v1.py
CFG=nfe_protocol_config.dev_e2e_three_rules.json
$PY --config $CFG freeze && $PY --config $CFG select && $PY --config $CFG seal-dev
$PY --config $CFG recommend && $PY --config $CFG validate && $PY --config $CFG verify-ledger
```
