# π0.5 LIBERO-plus Sensor Noise 缓慢诊断（2026-10-16）

**约束**：未终止进程、未改运行中代码/配置；未影响 GPU3 Molmo 与他人任务。  
**取证窗口**：约 20 分钟（日志解析 + 30s GPU 采样 + 进程/环境核对）。  
**诊断时刻（UTC+8）**：2026-10-16 午间。

---

## 1. 真实进度（三进程）

### 进度计数方法（可复现）

- **已完成变体数**：自本次 `==== START` 起，统计日志中 `Stepping through eval batches: 100%|…| 1/1 [` 的出现次数，并对 tqdm **回车刷新导致的相邻重复行**做去重（比较 `[…]` 内整段内容）。
- **不用** `Running rollout … 520/520` 出现次数作为变体完成数（那是**单回合内仿真步**，且 Long 套件上限 520 步）。
- **当前任务 ID**：`Restricting to task_ids=[…]` 列表中第 `已完成数` 项（0-based 下标 = 已完成数）。
- **变体耗时**：从上述 `1/1` 行解析 `MM:SS<` 得到秒数；部分完成行未带 `MM:SS` 格式时，**耗时样本数会略小于完成数**（见下表 gap）。

| GPU | 任务标签 | PID | 已运行 | 套件 | 变体进度 | 当前 task_id（列表位置） | 当前噪声（LIBERO-plus） | 当前回合步速 |
|-----|----------|-----|--------|------|----------|---------------------------|-------------------------|--------------|
| 0 | `plus_full_noise_10` | 3561009 | ~1d 6h | `libero_10` | **312 / 449**（69.5%） | **1796**（313/449） | `glass_blur` severity **7**（`noise_47`） | ~**3.6–3.9 s/step**，上限 **520** 步 |
| 2 | `plus_fisher6_noise_10` | 3565007 | ~1d 6h | `libero_10` | **313 / 449**（69.7%） | **1797**（314/449） | `glass_blur` severity **8**（`noise_48`） | ~**3.6–3.8 s/step**，上限 **520** 步 |
| 1 | `plus_fisher6_nfe1_noise_object` | 361005 | ~9h | `libero_object` | **166 / 422**（39.3%） | **1562**（167/422） | `glass_blur` severity **7**（`noise_47`） | ~**3.4–3.5 s/step**，上限 **280** 步 |

- 三次评测均为 **`eval.n_episodes=1`**、**`batch_size=1`**、**`seed=1000`**；日志中 **仅 1 次** `==== START`（本轮无重启）。
- **提前成功**：当前三条日志尾部 `running_success_rate=0.0%`，当前回合倾向打满步数上限（Long 更慢）。

### 已完成变体的耗时分布（日志子集，n 见 gap）

| 日志 | 完成数 | 可解析耗时样本 | 全样本中位数 | 最近 10 个变体中位数 | 备注 |
|------|--------|----------------|--------------|----------------------|------|
| full NFE10 Long | 312 | 297（gap 15） | **191 s** | **830 s** | 近期多为 fog/glass 段，明显慢于前期 motion/gaussian |
| fisher6 NFE10 Long | 313 | 276（gap 37） | **218 s** | **441 s** | 与 Full 同套件，略快 1 个变体 |
| fisher6 NFE1 object | 166 | 150（gap 16） | **129 s** | **457 s** | Object 步数上限 280，但近期噪声段仍偏慢 |

噪声实现（**已确认**，源码）：`/home/guorj/LIBERO-plus/libero/libero/envs/env_wrapper.py` 每步对 `agentview_image` 做 PIL +（motion 用 ImageMagick/wand）扰动；`noise_41–50` → **glass_blur**。

### GPU1 后续队列（脚本，未改）

`run_eval_plus_fisher6_nfe1_noise.sh` → `run_eval_plus_dim.sh` 顺序：**spatial → object → goal → libero_10**。

- **已完成套件**：`plus_fisher6_nfe1_noise_spatial`（有 `eval_info.json`）。
- **进行中**：`plus_fisher6_nfe1_noise_object`（上表）。
- **待跑**：`…_goal`（Sensor Noise **379** 变体）、`…_10`（**449** 变体）。  
  仅剩余三套件粗算 **828** 变体 + object 剩余 **256** ≈ **1084** 变体（未含 spatial 已完成的 351）。

---

## 2. 剩余时间估算（区间，勿用单一线性外推）

**本套件剩余（Long / Object）**

| 进程 | 剩余变体 | 乐观（全样本耗时中位数） | 保守（最近 10 变体中位数） | 极保守（打满步数 × 当前 ~3.7 s/step） |
|------|----------|--------------------------|----------------------------|----------------------------------------|
| GPU0 Long | 137 | **~7 h** | **~32 h** | 单变体最长 ~32 min × 137 ≈ **3 d**（仅当几乎全部打满 520 步） |
| GPU2 Long | 136 | **~8 h** | **~17 h** | 同上量级 |
| GPU1 object | 256 | **~9 h** | **~33 h** | 280 步 × 3.5 s ≈ **16 min/变体** 上限量级 |

**说明**

- Long 已跑约 **30 h / ~312 变体** → 全程平均约 **5.8 min/变体**；与「最近 10 个 ~14 min」并存，说明 **噪声类型/难度段** 差异大，**不宜**用「当前 rollout 的 520 步进度百分比」外推整套件。
- 原先 **「两三周 / 卡」** 若来自把 **520 步 tqdm** 当成「449 个变体」或把 **单回合最慢情况** 乘全集：与当前 **~70% 套件进度 + 上述区间** **不一致**，应视为 **已否定**。更贴近现状的是：**Long 单作业剩余约 1–2 天（常见段）到数天（若后期全是 glass/fog 且全打满步）**；**GPU1 整条 NFE1 噪声队列** 仍可能是 **数天到 ~1 周+**（object 剩余 + goal + long）。

---

## 3. 瓶颈证据

| 类别 | 结论 | 依据 |
|------|------|------|
| 仿真 + 传感器噪声（CPU） | **已确认：主要贡献** | 每 env step 在 `env_wrapper.step` 对图像做 blur/fog/glass（PIL/wand）；步速 **~3.4–3.9 s/step** 与 NFE10 推理量级同阶；GPU1 **NFE=1** 时 **GPU 利用长期 ~0%**（30s 采样） |
| π0.5 推理（GPU） | **已确认：存在但非独占** | GPU0 **~33–38%** 波动；GPU2 **间歇 0–93%**；进程 **~10 GB** 显存；**NFE=10** 高于 NFE1 |
| 线程/进程结构 | **已确认** | 主进程 **~219** 线程、**~115% CPU**、RSS **~12 GB**；符合 MuJoCo 离屏渲染 + 单线程 Python 扰动 |
| 调用栈采样 | **未做** | 无 `py-spy`；`gdb`/`perf` 受环境限制（`perf_event_paranoid=4`），未安装额外工具 |
| 与他人抢 GPU | **推测：GPU0 轻微** | 同机多用户历史存在；本次 GPU0 仍有非零利用，GPU2 突发利用，需结合 `nvidia-smi` 进程列表自行对照 |

**评测代码路径（已确认）**

- 入口：`functional_necessity/scripts/pi05_libero/eval_drop.py`
- LeRobot：`/home/guorj/lerobot/src`（`eval_policy_all` 顺序跑变体，**变体间** `env.close()`）
- LIBERO-plus：`source ~/.libero_plus/env.sh` → `PYTHONPATH` 含 `/home/guorj/LIBERO-plus`（覆盖 site-packages `libero`）
- 环境：`lerobot` conda；`MAGICK_HOME=/home/guorj/.libero_plus/magick`

---

## 4. 中断与续跑

| 问题 | 答案 |
|------|------|
| 运行中落盘了什么？ | 启动时 **`drop_meta.json`**；**无**逐变体 JSON/视频（`max_episodes_rendered=0`）；**无**运行中 `eval_info.json` |
| `eval_info.json` 何时写？ | **`eval_policy_all` 整次结束**后一次性写入（`eval_drop.py` ~535 行） |
| 现在中断会丢什么？ | **当前变体未完成回合** + **本套件所有已完成变体的聚合结果**（仅在进程内存的 `per_task_infos` 等，**未持久化**） |
| 能否套件内断点续跑？ | **不能**。`run_eval_plus_lang.sh` 仅在目录已有 **`eval_info.json` 时跳过整次作业**；**不能**自动跳过已完成 `task_id` |
| 可靠恢复方式？ | **无现成机制**。理论上需改 `task_ids` 列表手工去掉已完成 ID 并重跑（需自己从日志还原 ID 列表）；否则只能整套件重跑 |
| 其他套件进度 | GPU1 spatial 的 **`eval_info.json` 已存在**，重跑脚本会 **skip 整个 spatial**，**不会**丢 spatial 结果 |

---

## 5. 分进程建议（本轮仅建议，未执行）

| 进程 | 建议 | 理由 |
|------|------|------|
| **GPU0 `plus_full_noise_10`** | **继续跑** | 已完成 ~70%；中断会丢 ~312 变体内存结果；近期 glass 段慢但可接受 |
| **GPU2 `plus_fisher6_noise_10`** | **继续跑** | 与 GPU0 同套件、进度相当；剩余时间略优于 Full（近期中位数更低） |
| **GPU1 `plus_fisher6_nfe1_noise_*`** | **继续跑**；**勿**为「加速」杀进程 | 队列最长（object+goal+long）；spatial 已安全落盘；中断代价高 |

**不建议本轮**：在线改 `env_wrapper` 扰动、增 GPU 推理任务、或依赖「两周」旧估时做资源决策。

---

## 6. 无法可靠确定的事项

- 精确到 **每一个** 已完成变体是否与「去重后的 312/313/166」完全一致（**15–37** 条完成行缺少可解析 `MM:SS`，可能存在日志截断/格式差异）。
- 各噪声族在未来剩余变体中的 **精确占比**（仅对已完成子集统计了 mix，与列表顺序相关）。
- GPU 争用对 **逐步延迟** 的定量拆分（缺栈采样）。

---

*生成方式：只读解析 `results/logs/eval_plus_*.log`、`/proc/<pid>/environ`、nvidia-smi 周期采样；未修改运行中评测。*
