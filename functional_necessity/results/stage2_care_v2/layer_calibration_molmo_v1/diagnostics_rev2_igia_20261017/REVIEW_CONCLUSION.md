# Molmo CARE_event rev2 — IGIA 冒烟复核结论（2026-10-17）

## 1. 冒烟记录矛盾

- 原始 `score_smoke_verify.json` **未修改**：`passed=false`，`failed_methods=["igia"]`（聚合 IGIA `max_rel_err≈4.68e-3`，略高于 `rtol=1e-3`；Fisher/GateProbe 通过）。
- 本轮诊断输出目录：`diagnostics_rev2_igia_20261017/`（`igia_path_diagnostic.json`、`igia_b80_ref_selection.json`）。

## 2. IGIA 新旧路径（2 固定样本 × 2 次重复）

- 同一权重、固定 calib 中 `noise`/`timesteps`；`score_frame_three`（rev2 逐帧）vs `score.py::measure_grad(..., "igia")`（原实现）。
- **重复性**：同一样本两次逐帧 IGIA **存在小幅数值波动**（两样本 `repeatability_ours.identical` 均为 **false**；逐层相对误差仍约 10⁻³ 量级）。
- **数值**：4 次逐样本比较聚合后 `max_rel≈1.46e-3`（略超 1e-3）、`max_abs≈0.023`（主要在中间层）；**不能宣称 400 帧全量逐层数值等价**，但与冒烟量级一致（~0.3–0.5% 相对误差级）。
- **选层**：与 Drop-16 模态相关的层集差异已由 b=80 子集原实现复算核实（见下）；**未**用 2 帧结果推断全池选层翻转。

## 3. 选层影响（b=80，现有分数 + 原实现核实）

| 指标 | 结论 |
|------|------|
| **full-400 等权** Fisher/IGIA | Drop-16 = S1 **Taylor-16**（98.10%）；Random b=80 另一模态为悬崖层集（72.75%，含 L5、不含 L24） |
| **CARE_event / α-only** b=40/80/120 | Fisher/IGIA 均 **100%** 选中 Taylor-16 层集 |
| **Random 20 种子** b=80 | Fisher：**14/20** Taylor-16，**6/20** 悬崖 72.75%；IGIA：**5/20** Taylor-16，**15/20** 悬崖 |
| **原 `measure_grad` IGIA，b=80 子集** | CARE_event 与 Random seed=0 的 Drop-16 与 rev2 `details.csv` **完全一致**（`drop_match=true`） |

**结论**：IGIA 实现差异属于 **数值一致性** 问题，**不改变**本轮 rev2 已分析的 **选层一致性**；**无需**重算 400 帧 IGIA / Fisher / Taylor。

## 4. 可正式使用的结果

- `scores/molmoact2_{igia,fisher,gateprobe}.pt`（400×36 逐帧）
- `taylor_groups/*_b80*.pt`（分组 Taylor，非 full-400 矩阵）
- `details.csv` / `drop_frequency.csv` / `summary.json` / 本 REPORT（解读已按本轮复核修订）
- 历史 SR 标签仅作 **闭环复用参照**；**未**在本轮重新跑 LIBERO 闭环

## 5. 显存

- 冒烟 `--with-gate` 峰值以 `torch.cuda.max_memory_allocated` 为准：**≈17.5 GiB**（`score_smoke_verify.json` → `mem_after_ref.max_allocated_bytes`），非“约 12 GiB allocated”。

## 6. 本轮未做

- 两个 G-only 闭环候选 **未启动**
- 未改动运行中的 LIBERO-plus 评测
