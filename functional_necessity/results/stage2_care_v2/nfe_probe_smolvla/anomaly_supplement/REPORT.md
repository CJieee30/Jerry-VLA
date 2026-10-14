# SmolVLA NFE 探针 — 异常样本补充（离线动作）

用语：**动作偏差**、**偏差缓解**；不将单帧低 NFE 动作差异表述为闭环失败或失败修复。

数据源：`results/stage2_care_v2/nfe_probe_smolvla` 的 `actions.npz` / `frames.csv`。
独立复算与 `frames.csv` 核对：36 条，max |Δcosine|=1.11e-16，max |Δrmse|=8.33e-17。

## 样本列表

| shared | 分组 | α | topα | gripH | 解读 |
| ---: | --- | ---: | ---: | ---: | --- |
| 89 | nfe2_remaining | 2.279 | 1 | 1 | noise0: 开合执行态在 NFE=1–3 均与参照不一致（1→1→1 步），更像切换时机偏移而非单步积分噪声。 noise0: NFE=1 手臂归一化 RMSE 偏大（trans=0.126, rot=0.156），NFE=2 降至 t… |
| 121 | care_miss_examples | 1.155 | 0 | 0 | noise0: NFE=1 手臂归一化 RMSE 偏大（trans=0.156, rot=0.229），NFE=2 降至 trans=0.075, rot=0.119。 noise1: NFE=1 手臂归一化 RMSE 偏大（trans=0… |
| 124 | care_miss_examples | 1.126 | 0 | 1 | noise0: 开合偏差在 NFE=2 已缓解（NFE=1 有 1 步不一致）。 noise1: 开合偏差在 NFE=2 已缓解（NFE=1 有 1 步不一致）。… |
| 381 | nfe2_remaining | 1.006 | 0 | 1 | noise0: 开合执行态在 NFE=1–3 均与参照不一致（1→1→1 步），更像切换时机偏移而非单步积分噪声。 noise0: NFE=1 手臂归一化 RMSE 偏大（trans=0.090, rot=0.175），NFE=2 降至 t… |
| 437 | nfe2_remaining | 1.435 | 1 | 1 | noise1: NFE=2 仍保留开合偏差（1 步），NFE=3 已缓解；低步数下夹爪阈值附近敏感。 noise1: 余弦在 NFE=2 仍偏低（0.929→0.975），以手臂方向偏差为主。… |
| 451 | nfe2_remaining | 1.312 | 1 | 1 | noise1: 开合偏差在 NFE=2 已缓解（NFE=1 有 1 步不一致）。 noise1: NFE=1 手臂归一化 RMSE 偏大（trans=0.130, rot=0.121），NFE=2 降至 trans=0.063, rot=0… |

## NFE=2 后仍偏大的帧（89/381 @ noise0；437/451 @ noise1）

- **89 / 381（noise0）**：NFE=2 余弦已升高，但 **开合执行态仍整段不一致**（`gripper_disagree_n=1` 贯穿 NFE=1–3），连续夹爪值在阈值附近偏移，手臂 RMSE 已明显下降 → **切换时机 / 阈值敏感** 为主，不是单纯手臂大幅跑偏。
- **437（noise1）**：NFE=1 手臂与夹爪偏差都大；NFE=2 **夹爪仍不一致** 且余弦 <0.98 → 低步数下 **夹爪+手臂** 复合偏差。
- **451（noise1）**：NFE=1 余弦最低档（~0.84）且夹爪 1 步不一致；**NFE=2 夹爪已对齐**，余弦升至 ~0.98 → 典型 **NFE=2 偏差缓解**；残余为手臂归一化误差。

## 高 α 未覆盖的代表（124 / 121）

- 二者 **不在 topα 组**（每任务 α 最高 3 帧），但 NFE=1 在 noise1 上余弦/RMSE 仍进入全体尾部（尤其 124）。
- **NFE=2 均明显缓解**（夹爪不一致归零，余弦 >0.99）→ CARE α **漏检** 的是「中等 α 但仍对低 NFE 敏感」的帧，不能仅靠高 α 子集筛尽动作偏差。

## 文件

- `shared_XXX.json` / `shared_XXX_steps.csv` / `shared_XXX_curves.png`
- `verification.json`

