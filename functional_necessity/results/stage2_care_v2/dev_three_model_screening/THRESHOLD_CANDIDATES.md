# RMSE 阈值候选（开发候选，非冻结规则）

仅比较三条固定规则；未做组合搜索以保证指定 NFE。
七维与六维分开；余弦仅作辅助说明。

## `ratio_rmse6_median_le_1`
六维手臂 RMSE 中位 / 全池同 NFE 中位 ≤ 1.0 + 夹爪率不差于 NFE10
- **molmoact2** random: {10: 8, 3: 6, 1: 4, 2: 2}; g_only: {10: 20}; conservative_reject=28, false_accept=0
- **pi05** random: {10: 18, 1: 2}; g_only: {10: 20}; conservative_reject=0, false_accept=2
- **smolvla** random: {10: 15, 2: 3, 3: 2}; g_only: {10: 20}; conservative_reject=0, false_accept=5

## `ratio_rmse7_median_le_1`
七维 RMSE 中位 / 全池同 NFE 中位 ≤ 1.0 + 夹爪率不差于 NFE10
- **molmoact2** random: {3: 7, 10: 6, 2: 4, 1: 3}; g_only: {10: 20}; conservative_reject=26, false_accept=0
- **pi05** random: {10: 18, 1: 2}; g_only: {10: 20}; conservative_reject=0, false_accept=2
- **smolvla** random: {10: 15, 2: 4, 3: 1}; g_only: {10: 20}; conservative_reject=0, false_accept=5

## `abs_rmse6_obs_p95_le_pool_p95_nfe1`
策略内观测级手臂 P95 ≤ 全池 NFE1 观测 P95 + 夹爪率不差于 NFE10
- **molmoact2** random: {3: 8, 1: 7, 2: 5}; g_only: {2: 8, 3: 6, 1: 6}; conservative_reject=0, false_accept=0
- **pi05** random: {10: 16, 2: 2, 1: 2}; g_only: {10: 20}; conservative_reject=0, false_accept=4
- **smolvla** random: {10: 12, 2: 6, 3: 2}; g_only: {10: 20}; conservative_reject=0, false_accept=8

## 局限
- 夹爪率基于 frames.csv 原生 `gripper_disagree_n`，与 LIBERO 统一空间步级不一致计数可能不同。
- 全池 floor 中位未按 episode 聚簇；bootstrap 仅对 random@seed42 示例一条规则。
- 闭环标签 episode 数不一致（Molmo 2000 vs Pi/Smol 400），不可直接比成功率。
