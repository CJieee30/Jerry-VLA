# CARE_event 校准池等预算选样（480 帧，CPU 回放）

- 每任务 3 帧，共 120 帧；合并 noise0+noise1 的捕获数（非独立 episode）。
- 多随机 / G-only 种子：42, 0, 1, 7, 13, 17, 23, 29, 31, 37, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83
- 与历史 `summary.json` 复现：通过

## 四组对照（合并两噪声）

| model | group | grip_nfe1 | tail_nfe1 | arm_top5 |
| --- | --- | --- | --- | --- |
| smolvla | random | 8/15 | 26/73 | 16/48 |
| smolvla | care_alpha | 11/15 | 36/73 | 30/48 |
| smolvla | g_only | 13/15 | 38/73 | 17/48 |
| smolvla | care_event | 14/15 | 44/73 | 27/48 |
| pi05 | random | 5/10 | 23/71 | 13/48 |
| pi05 | care_alpha | 7/10 | 38/71 | 27/48 |
| pi05 | g_only | 10/10 | 32/71 | 12/48 |
| pi05 | care_event | 10/10 | 38/71 | 25/48 |

## 多随机种子波动（合并两噪声，捕获数）

### random

- **smolvla** grip NFE1: mean=4.65 std=2.46 [1,11]; tail: mean=19.30 std=4.55; arm top5%: mean=12.45 std=2.92
- **pi05** grip NFE1: mean=2.45 std=1.56 [0,6]; tail: mean=19.55 std=4.41; arm top5%: mean=13.65 std=3.51

### g_only

- **smolvla** grip NFE1: mean=11.75 std=1.34 [10,14]; tail: mean=35.70 std=2.22; arm top5%: mean=14.80 std=1.91
- **pi05** grip NFE1: mean=10.00 std=0.00 [10,10]; tail: mean=27.15 std=3.23; arm top5%: mean=8.30 std=2.70

