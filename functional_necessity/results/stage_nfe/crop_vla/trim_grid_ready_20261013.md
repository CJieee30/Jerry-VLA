# trim＋grid 验证准备（NFE=1）

代码 `e957c92`（工作区未提交）。CPU 自测通过。离线检查在共享 GPU1 上通过。**正式效率复测与两组 400 回合未执行：当前无独占 3090。**

## 三项小修

- trim-only 完整计时不再在 `embed_prefix` 里做 `sum().item()`。清理仍用 mask 判断（空相机 / 语言右边界）。`profile=False` 与分段分开。
- 离线检查增加真正剪枝后的 `grid_vs_trim`（前 10 步 × 7 维：max-abs / MAE / rel-L2）。**不套 keep-all 的 2e-2 门限**。动作非有限立即失败。
- 18 层 K/V 长度全部记录并核对，含 Drop-6 `[1,13,14,15,16,17]`。keep-all / trim-vs-raw 门限未放宽。

## 离线检查（共享 GPU1，`--allow-shared`）

`offline_check_trim_grid_nfe1.json`

| 项 | 结果 |
|---|---|
| 实际 NFE | raw=1，trim-only=1，grid=1 |
| trim-only vs raw | max_abs=0.00996 ≤ 2e-2 |
| keep-all vs trim-only | max_abs=0 |
| grid vs trim（分析） | max_abs=0.528，rel_l2=0.199（frame0）；动作有限 |
| 长度 | raw 968；trim 560/569/563/554；grid 304/313/307/298 |
| 18 层 KV | raw 全 968；trim 全 trimmed；grid 全 short；Drop-6 同长度 |

棋盘剪枝会改动作，这是预期，不是 keep-all 失败。

## 未执行

四张卡均被占用（本机 Plus eval、sunxy、yany）。未跑 `--mode formal`，未跑 400 回合。不要把共享卡诊断 `efficiency_diagnostic_trim_grid_nfe1_20261013T101012Z.json` 当作正式效率表。

目标 GPU **完全空闲后**，同卡不要并行闭环。建议顺序：

```bash
cd /home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/pi05_libero
# 1) 正式效率 A=raw B=trim-only C=trim-grid
bash run_measure_trim.sh          # 或 bash run_measure_trim.sh N
# 2) 闭环：先 trim-only，再 trim-grid（各 400 回合）
bash run_eval_trim.sh trim-only   # 或 bash run_eval_trim.sh trim-only N
bash run_eval_trim.sh trim-grid
```

两组都会先跑同一套 `--prefix-trim --selector grid --nfe 1` 离线检查，失败即退出。输出目录：

- `results/stage_nfe/crop_vla/eval_trim_only_nfe1`
- `results/stage_nfe/crop_vla/eval_trim_grid_nfe1`

对照：整体 raw 复用本机 Drop-6 NFE=1 **96.50%**；视觉剪枝对照必须用本机新跑的 trim-only，不复用他机 trim。FLOPs 未测。历史 `eval_crop/` 未覆盖。
