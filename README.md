# Which Layers Does a Fine-tuned Flow-Matching VLA Actually Need?

微调后、training-free、零恢复的 flow-matching VLA 层冗余研究。主问题不是再提出一个剪枝算法，而是：在 action expert 逐层读取 K/V 的模型上，哪些语言层在功能上可以删掉，以及删层判据必须和删层算子匹配。

数字来自本机 LIBERO 闭环，`seed=1000`，`compile_model=false`。权威表在 `functional_necessity/main_results/`。`HANDOFF.md` 是给后续实验的约束。`functional_necessity/another_server_results/` 是另一台机器的记录，不进入下面的表。

## Idea

π0.5、MolmoAct2、SmolVLA 的 action expert 读取骨干每一层的 K/V，不读取该层残差，也不读取骨干末层隐状态。因此被删的 VLM 层可以保持残差恒等，同时仍写出 K/V（keep-KV）。在这个算子下：

- Fisher / Taylor 能在分布内 LIBERO 上删掉深层并保住成功率。
- CosSim / CKA，以及从末尾连续删同样多的层，会塌。
- 第 0 层是必要的。末层残差对动作几乎是恒等的。
- 若把同一组层的 K/V 写成零，成功率明显下降。keep-KV 是成功率条件。

读末层隐状态、并且整层恒等跳过的 OpenVLA-OFT 结论相反：CosSim Drop-4 为 86.60，Fisher / Taylor 为 0。所以「余弦不适合 VLA」只对逐层读 K/V 的 flow-matching 模型成立。

主张范围是分布内深度冗余。LIBERO-plus 语言维和 RoboTwin 上的掉点是限制，不是迁移后的压缩结果。

## 方法

算子（三条主栈相同，action expert 不删，不做恢复微调）：

- 被删层：输入归一化后只写 K/V，残差不更新，不跑 Q、attention 输出和 MLP。
- π0.5 对应 DTR Appendix B 的 prefix skip。
- 高分 = 保留。Drop-k 取该指标冗余序中最低的 k 层。π0.5 与 MolmoAct2 的高 k 先从排序里去掉 layer 0，再取 k。

判据在同一套连续 flow-matching loss 上计算。校准集 32 帧，`seed=42`，每帧冻结一对 `(noise, time)`。PPL 是动作 loss 的增量，不是 token perplexity。选层只用校准 loss，不用闭环成功率回搜层。

闭环协议：

| 模型 | 权重 | 集数 |
| --- | --- | ---: |
| π0.5 | `lerobot/pi05_libero_finetuned_v044` | 400；完整模型与 Fisher Drop-6 另有 2000 |
| MolmoAct2 | `allenai/MolmoAct2-LIBERO-LeRobot` | 2000 |
| SmolVLA | `lerobot/smolvla_libero` | 400 |

四套件为 Spatial / Object / Goal / Long（LIBERO-10）。SmolVLA 使用 `n_action_steps=10`、相机重命名和 8 维 state。MolmoAct2 关闭推理 CUDA graph。

经营点：π0.5 Fisher Drop-6 `[1,13–17]`；MolmoAct2 Taylor Drop-16 `[1–4,24–35]`；SmolVLA Taylor Drop-4 `[12–15]`。SmolVLA 没有成功率平台，作边界。

## 成果

成功率是百分比。Δ 相对本机完整模型。0.25–0.75 的平均分差视为噪声，写持平。

### 分布内主结果

| 模型 | 完整模型 | 经营点 | CosSim | 连续删尾 |
| --- | ---: | --- | ---: | --- |
| π0.5（400） | 97.25 | Fisher Drop-6 **97.75** | 0.75 | Last-6 73.50 |
| π0.5（2000，NFE=10） | 96.80 | Fisher Drop-6 **97.10** | — | — |
| MolmoAct2（2000） | 97.50 | Taylor Drop-16 **98.10** | Drop-8 0.35；Drop-16 0.00 | Last-16 58.25 |
| SmolVLA（400） | 83.00 | Taylor Drop-4 **78.50** | 30.50 | Taylor 的 Drop-4 本身就是末四层 |

π0.5 上 Fisher Drop-9 降到 76.75，Last-9 为 19.50。MolmoAct2 上 Fisher Drop-16 为 72.75，Taylor 要到 Drop-17 才降到 89.60。不存在跨模型、跨 k 的单一冠军。

只删 layer 0：

| 模型 | Spatial | Object | Goal | Long | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| π0.5 | 0 | 0 | 0 | 0 | 0.00 |
| MolmoAct2 | 0 | 0（记 0） | 0（记 0） | 0（记 0） | 0.00 |
| SmolVLA | 11.0 | 13.0 | 4.0 | 7.0 | 8.75 |

MolmoAct2 的 Spatial 为 0 后，其余三套按停止规则记 0。

π0.5 Fisher Drop-6 不保留 K/V（同一层集，K/V 写零，400 集）：Spatial 34 / Object 36 / Goal 34 / Long 1，平均 **26.25**。同一层集的 keep-KV 是 97.75。

末层 keep-KV 的开环动作（32 帧，NFE=1 和 10）：π0.5 L17 与 SmolVLA L15 的 max-abs 为 0。MolmoAct2 L35 的均值约为 1.6e-3 / 3.7e-3。

### 效率（支撑，不是贡献）

eager、同 NFE 下，删语言层几乎不改变整段 `predict_action_chunk` 的墙钟，因为时间主要在 action expert。可报告的是激活参数、显存、FLOPs，以及和已有少步方法的叠加。例如 π0.5 Drop-6 × NFE=1 相对完整模型 NFE=10，chunk 时间约 −70%，FLOPs 约 −33%，400 集成功率为 96.50。同 NFE=1 时，剪枝本身约 −13.9% 时间、−28.1% FLOPs。不要把少步倍数写成层剪枝的加速。

### 没有做成方法的实验

- CARE：用动作不连续给校准帧加权。三个经营点的删层集合不变。近邻平局上会翻到已经评过的更强集合。
- 突变帧 / 平滑帧校准：Taylor 不变。MolmoAct2 Fisher-16 上，突变帧映射到已评的 98.10 集合，平滑帧映射到已评的 72.75 集合。
- LEDGE：用分数漂移缩小需要闭环的 k 区间，大约少测一档。规则在 π0.5 和 MolmoAct2 上拟合，没有留出模型。粒度是模型 × 基准，不是任务。
- 任务条件删层：π0.5 上 Fisher 的任务额外层为 43.00，打乱后为 44.50。层会变，但不能按任务决定最大 k。

### 限制

LIBERO-plus 只完成了 Language Instructions（每变体 1 trial）：完整模型 89.20，Fisher Drop-6 85.24，Δ −3.96。Robot Initial States 与 Camera Viewpoints 的对照还在评。另外四个维度没测。

RoboTwin 上 Fisher Drop-2 为 69.63，相对完整模型 71.75；Drop-6 会明显下降。深度冗余不外推到 LIBERO 以外。

## 目录

```
HANDOFF.md                          实验约束与禁止项
functional_necessity/main_results/  论文用表
functional_necessity/scripts/       三条主栈的打分、skip、闭环
functional_necessity/results/       eval_info.json 与打分；权重和视频不入库
paper/                              论文用的精简表
```

权重、校准 `.pt`、视频、日志和 parquet 在 `.gitignore` 里。不要把 `another_server_results` 的数字写进主表。
