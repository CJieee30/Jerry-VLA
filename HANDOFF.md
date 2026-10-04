# HANDOFF：微调后 Flow-matching VLA 的 Training-free Layer Pruning

写给**完全没有上下文的新会话**。先读本文，再读 `functional_necessity/main_results/` 里的表。**不要重跑**已完成的闭环。日期：2026-10-04。原机：`miaa-hpc2`，用户 `guorj`。回复用**简体中文**。不要 commit，除非用户明确要求。

**一句话现状：** Stage-1（flow-matching VLA 的规范化层冗余检测）已结束。主基线是 π0.5、MolmoAct2、SmolVLA（小模型辅助）。下一步是 **Stage-2：定义动作重要性，提高重要 timestamp 在校准集里的权重**，让 Stage-1 已选定的打分（Fisher / Taylor）排出更准的冗余层。算法还没选定，先调研再实现，用户批准后再开闭环。Molmo last-16 对照仍在 GPU1 上跑，结论已经够用，跑完只补一行，不要杀进程。

权威数字只信这些文件，不要凭记忆改表：

- `functional_necessity/main_results/eval_pi05_libero_v044.md`
- `functional_necessity/main_results/eval_molmoact2_libero.md`
- `functional_necessity/main_results/eval_smolvla_libero.md`
- `functional_necessity/main_results/efficientcy_analysis.md`（文件名就是这个拼写，不要另建 `efficiency_analysis.md`）
- 旁支，**不要写进 FM 主结论**：`eval_openvla_oft_libero.md`、`eval_db_cogact_libero.md`、`eval_dtr_drop9_recovered.md`、`eval_pi05_libero_v044_task_aware.md`、`eval_dtr_drop9_recovered_task_aware.md`

本文件是唯一给新会话的源。

---

## 0. 新会话 60 秒

1. 冲 **2026-11 左右的 CVPR**。工作名仍可用：**Function Before Similarity: Training-Free Post-Tuning Layer Pruning for VLAs**。
2. Stage-1 要回答的是：微调后的 flow-matching VLA，用什么 training-free 判据可以零恢复地剪语言主干。答案已经有了，见第 1.2 节。**不要再扫 k、不要再跑九法。**
3. Stage-2 要回答的是：均匀抽 32 帧不够贴「关键动作」。给校准 timestamp 加权之后，同一套 Fisher/Taylor 能否选出比均匀校准更好的层。**权重不能用闭环成功率来拟合。**
4. 主栈只有三条，全是 keep-KV、seed=1000、本机锚点当裁判：π0.5 **97.25**、MolmoAct2 **97.50**、SmolVLA **83.00**。OFT 与 DB-CogACT 是整层恒等，数字不能和这三条横比。
5. 推荐操作点（Stage-2 的对照就钉在这三行上）：π0.5 Fisher Drop-6 **97.75**；Molmo Taylor Drop-16（不含 L0）**98.10**；SmolVLA Taylor Drop-4 **78.50**（−4.50，用户已接受）。
6. 协议、公式、skip 算子都不要改。新实验单独成行。PPL = **action Δloss**，不是 token perplexity。高分 = 保留。

---

## 1. 任务目标

给**微调后、training-free、零恢复**的 flow-matching VLA 立一套层冗余标准，并说明它省的是语言前缀 FLOPs，不是 LIBERO 墙钟。

### 1.1 论文里可以写的四条（2026-10-04 已和用户对齐）

1. Flow-matching VLA 的语言主干有可观冗余，training-free、零恢复即可剪。π0.5 删 6/18 仍 +0.50；Molmo 删 16/36 仍 +0.60；SmolVLA 删 4/16 付 −4.50（小模型预算更薄，不是反例）。
2. 表示相似度不是功能必要性。`1−cos` / CKA / Magnitude 会选到早期层并把策略打崩。这是对 EfficientVLA 的直接反驳。
3. 匹配 keep-KV 的功能类打分才是选层器。实用配方：**默认从 redundancy_order 去掉 L0**；中等预算用 Fisher，贴悬崖用 Taylor（保关键中后层，例如 Molmo 的 L24）。不要切 last-k。
4. 剪枝降低的是语言前缀 FLOPs。Action 去噪占 chunk 墙钟，所以 chunk 延迟只降几个点，LIBERO 圈时间几乎不变。收益写成 FLOPs，不要写成仿真加速。

「第一篇」的口径：据我们所知，**首次对 flow-matching VLA 主干**做系统的 training-free / no-recovery 层冗余研究。禁止写成「首次对 VLA 做 training-free layer pruning」（EfficientVLA 已经做过）。

禁止再写：

- 「Fisher 永远最好」——Molmo Drop-16 上 Taylor 98.10，Fisher/IGIA 72.75。
- 「DySL 是 training-free」——它要 adapter + controller + KD。
- 「CLP 是 ICPR 的 CNN CKA」——指的是 VLA 那篇微调前 CKA + 下游微调。
- Task/Skill-aware 是主方法。诊断已否定「每个技能一套层」。
- 把 OFT / DB-CogACT 整层恒等的崩盘写成 FM keep-KV 的结论，或写成「因为 LoRA」。
- 0.25–0.75 点的 Avg. 差是方法差异。π0.5 / SmolVLA 上 1 集 = 0.25。

### 1.2 Stage-1 已站得住的句子

- Recoverability ≠ functional necessity。CLP/DTR 的 CKA/GateProbe 服务「删完再训」；零恢复时 CosSim/CKA 在 π0.5、Molmo 上接近 0。
- 判据必须匹配删层算子。keep-KV：被删 VLM 层仍写 K/V，残差不更新，action expert 不删。
- L0 是结构必要层，不是打分噪声。π0.5 只跳 `[0]` = **0.00**（0/400）。Taylor 原生 Drop-6 含 L0 = 0.00；去掉 L0 后 `[1,2,14–17]` = **97.25**。
- 排名 ≠ 切尾巴。π0.5 last-6 `[12–17]` = **73.50**，Fisher 同 k `[1,13–17]` = **97.75**。SmolVLA Taylor Drop-4 恰好等于 last-4 `[12–15]`，所以小模型上不必再跑 last-4。
- 不存在跨模型、跨 k 的单一冠军。中等 k 用 Fisher；悬崖边用 Taylor。
- 独立 PPL / Greedy-PPL 在 keep-KV FM 上不够。π0.5 Drop-9：Greedy-PPL 47.00，独立 PPL 43.00，Fisher 76.75。不要把 OFT 上「greedy 更好」搬进主文。
- 动作专家主导延迟。π0.5 Fisher-6：语言 FLOPs −33.0%，chunk FLOPs −26.13%，chunk 时间 −3.70%。Molmo Taylor-16：语言 FLOPs −31.6%，chunk FLOPs −22.99%，chunk 时间 −1.03%。SmolVLA Taylor-4：语言只占 chunk FLOPs 的 9.6%，整段 chunk FLOPs 只 −1.96%。

### 1.3 Related work（必须写对）

| 工作 | 实际是什么 | 相对我们 |
|---|---|---|
| **EfficientVLA** (NeurIPS 2025, [arXiv:2506.10100](https://arxiv.org/abs/2506.10100)) | 已训练官方 CogACT（SIMPLER）上 training-free LLM 层剪枝；`1−cosine`；另有视觉 token 与 action cache | 最直接竞争者。不能写「首次」。他们的宿主没有公开 LIBERO 微调权重。我们的反驳是：同一类 `1−cos` 在 FM-VLA 上选层会崩 |
| **CLP** ([arXiv:2606.20246](https://arxiv.org/abs/2606.20246)) | 微调前用相邻层 CKA 找 twin block，永久删层后再 downstream FT | 删完再训 |
| **DTR** ([arXiv:2606.27755](https://arxiv.org/abs/2606.27755)) | GateProbe 选可恢复块，必须 recovery FT | 同 CLP 一侧。栈 B 用的是他们 Drop-9 recovered ckpt |
| **MoLe-VLA** (AAAI 2026) | 训练期把 LLM 层当 expert；评 RLBench | 需要训练 |
| **DySL-VLA** ([arXiv:2602.22896](https://arxiv.org/abs/2602.22896)) | cosine 分 static/dynamic；跳层靠 adapter + controller + KD；LIBERO 宿主是 OpenVLA-OFT | 不是 training-free。trajectory continuity（连续性低则动作更重要）只是 Stage-2 的**例子**，不是已选定的定义 |
| **AtomicVLA / SkillNet / SkillsCrafter** | 技能标注或 action-expert / LoRA MoE | 只做过负对照。没有证明不同技能需要不同 LLM 层 |

---

## 2. 实验栈（绝对不要混）

绝对值不能横比。删层算子不同。

| | A. π0.5 v044 | D. MolmoAct2 | E. SmolVLA | B. recovered（旁支） | C. OpenVLA-OFT（旁支） | CogACT（旁支） |
|---|---|---|---|---|---|---|
| 权重 | `lerobot/pi05_libero_finetuned_v044` | `allenai/MolmoAct2-LIBERO-LeRobot` | `lerobot/smolvla_libero` | `s1ghhh/VLADrop_pi05_LIBERO_Drop9Block_GateProbe` | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | `Dexmal/libero-db-cogact` |
| 代码 | `scripts/pi05_libero/` | `scripts/molmoact2_libero/` | `scripts/smolvla_libero/` | `dtr_drop9_recovered/` | `scripts/openvla_oft_libero/` | `scripts/cogact_libero/` |
| Python | lerobot | lerobot | lerobot | serve：`.venv`；client：lerobot | `openvla-oft` | 见该目录 |
| seed | **1000** | **1000** | **1000** | **7** | **7** | **7** |
| 闭环 | 4×10×**10** = 400 | 4×10×**50** = 2000 | 4×10×**10** = 400 | 400 | 500/套 | 2000 |
| 跳层 | Appendix B：留 prefix K/V，丢 Q/attn/O/MLP | keep-KV：仍写 K/V，残差不更新；expert 不删 | 同 Molmo，16 层 VLM | 同 A | **整层恒等** | **整层恒等**（DiT 读最后隐状态） |
| 表 | `eval_pi05_libero_v044.md` | `eval_molmoact2_libero.md` | `eval_smolvla_libero.md` | `eval_dtr_drop9_recovered.md` | `eval_openvla_oft_libero.md` | `eval_db_cogact_libero.md` |
| 本机锚点 | **97.25** | **97.50** | **83.00** | **95.00** | **95.95** | **92.95** |

Stage-1 主文只用 A / D / E。B 用来说明「再全局删已经没有廉价层」。C / CogACT 用来说明整层恒等不是 keep-KV，**不要扩展**。

选层只用校准 loss。**禁止**用闭环成功率回搜层或回搜 Stage-2 的权重。

SmolVLA 评测必须：`n_action_steps=10`（checkpoint 默认 50 不用）、`num_steps=10`、`chunk_size=50`、相机 `image→camera1`、`image2→camera2`、state 形状按已存的 8 维 normalizer 来，不要用配置里过时的 6。`HuggingFaceVLA/smolvla_libero`（32 层）是另一个 checkpoint，不在表里。

Molmo 评测：`inference_action_mode=continuous`、`dtype=bfloat16`、`enable_inference_cuda_graph=false`、`chunk_size=n_action_steps=10`。高 k 从 `redundancy_order` **去掉 0 再取 k**。

---

## 3. 已经完成什么（不要重跑）

### 3.1 栈 A：π0.5（18 层，0–17）

目录：`results/eval_drop/`。

- Drop-6：Fisher **97.75** `[1,13–17]`；Taylor no-L0 **97.25** `[1,2,14–17]`；IGIA 97.00；PPL 96.25；GateProbe 95.00；Hessian 86.50；**Last-6 73.50**；CKA 2.00；CosSim 0.75；Taylor 原生（含 L0）/ Magnitude = 0。
- Drop-7 no-L0：Fisher=Taylor **96.00**；GateProbe 93.25；IGIA 93.00。
- Drop-8 no-L0：GateProbe 85.75；Fisher=Taylor **85.50**。
- Drop-9：Fisher **76.75**；GateProbe 73.00；IGIA 62.00；Greedy-PPL 47.00；PPL 43.00；其余接近 0。
- **Skip L0 only `[0]`：0.00**（0/400）。墙钟 3h 28m，因为几乎每集都跑满步数。
- Drop-3 九法闭环**暂停**，没有 `eval_info.json` 的目录不要补数。不要为了补全九法去跑。

### 3.2 栈 D：MolmoAct2（36 层，0–35，keep-KV）

目录：`results/molmoact2_libero/`。锚点 97.50。Hub 98.25 不当裁判。

- Drop-8 `[28–35]`：Taylor/Fisher/IGIA/GateProbe 同集 **97.75**。PPL 50.15。Magnitude 32.15。CosSim 约 0.35（Long 未跑完，其余记 0，无 `eval_info.json`）。CKA 不评。Hessian OOM，没有层号。
- Drop-12：Taylor=Fisher **97.75**。
- Drop-14 no-L0：Taylor=IGIA **98.00**；Fisher 97.45。
- Drop-15 no-L0：Fisher=IGIA **98.10**；Taylor 97.70。
- Drop-16 no-L0：Taylor **98.10** `[1–4,24–35]`（保住 L24）；GateProbe 87.00；Fisher=IGIA **72.75**；PPL 中途 Spatial 已崩，剩余记 0。
- Drop-17 no-L0：Taylor **89.60**；GateProbe 82.65；Fisher=IGIA 70.60。
- 悬崖：Fisher 在 15→16 掉约 25 点；Taylor 在 16→17 才掉。

**未写入表、仍在跑：** Last-16 对照，drop `[20–35]`，tmux / pid 以现场 `nvidia-smi` 为准（2026-10-04 晚些时候在 GPU1，pid 2754175）。中途 Goal 已经远低于 Taylor Drop-16 的 Goal 98.6，所以「36 层上也不是切尾巴」这个方向已经成立。**不要把中途成功率写进 md。** 出现 `results/molmoact2_libero/drop16_last/eval_info.json` 之后，只追加一行 Last-16，不要改旧行。

### 3.3 栈 E：SmolVLA（16 层，0–15，keep-KV）

目录：`results/smolvla_libero/`。锚点 83.00（332/400），不是论文 87.3。用户已定性：**Drop-4 有冗余，成功率略降可接受。**

Taylor 的 k=1…4 就是切尾巴：`[15]` 79.25、`[14,15]` 80.50、`[13–15]` 77.75、`[12–15]` **78.50**。Drop-2 高于 Drop-1、Drop-4 高于 Drop-3 都是几集噪声。

Drop-4 全九法：Taylor 78.50；PPL 74.75；IGIA 73.75；GateProbe 72.00；Fisher 69.25；Hessian 65.50；CosSim 30.50；CKA 22.75；Magnitude 4.75。

Drop-1 同层合并，不要拆开重跑：`[15]` 覆盖 Taylor/GateProbe/PPL；`[14]` 覆盖 IGIA/Fisher。Drop-3 Fisher=IGIA `[10,12,14]`，74.75。

L15 在 PPL/GateProbe 上分数为 0：keep-KV 下专家读的是 L15 的 K/V，残差更新不进 action loss。这是架构结果，不是 bug。

### 3.4 效率（`efficientcy_analysis.md`）

RTX 3090，`compile_model=false`，CUDA event warmup=5 / repeats=20，一条校准观测，`FlopCounterMode`。Stored params 不变；表里的 Param 是 **active**（只减去该栈 skip 真正不跑的权重）。

- 模块拆分是**未剪枝 baseline**。π0.5 Action 占时间 74.2%、FLOPs 只占 7.7%。Molmo Action 占时间 89.5%。SmolVLA Vision 占 FLOPs 72.0%，Language 只占 9.6%。
- 共享剪枝表：π0.5 Fisher-6 active 3489.08（−15.79%），521.19 ms（−3.70%），3711.6 G（−26.13%），SR 97.75。Molmo Taylor-16 active 4246.78（−21.97%），1158.76 ms（−1.03%），4043.3 G（−22.99%），SR 98.10。SmolVLA Taylor-4 active 416.86（−7.37%），485.10 ms（−1.64%），586.7 G（−1.96%），SR 78.50。
- JSON：`results/eval_drop/efficiency_{pi05_baseline_modules,fisher_k6,pi05_drop6_language_only}.json`，`results/molmoact2_libero/efficiency_taylor_drop16.json`，`results/smolvla_libero/efficiency_taylor_drop4.json`。

### 3.5 旁支（完成即可，不要扩展、不要重跑）

- 栈 B extra Drop-1：最便宜 L2 **87.00**；L0 = 0。extra Drop-3 最好也只有 **18.75**。没有还能再全局删的廉价 3 层。
- Task / Suite / Skill-aware：层会变，但没有可迁移的 skill→layer。原始输出在 `results/eval_task_extra/`、`results/eval_skill_extra/`、`results/drop6_extra3/`、`results/drop9_skill/`。引用前读 `eval_info.json`。
- OFT 整层恒等：含 L0 的梯度法大量是 0；去掉 L0 后 Fisher Drop-4 仍 15.00。Greedy-PPL Drop-3/4/8 = 94.05 / 91.80 / 68.80。这是「删策略主干」，不是 keep-KV。
- DB-CogACT Drop-8：Fisher no-L0 与 GateProbe 在 Spatial 开头就是 0，剩余按停止规则记 0。不要再排队。

### 3.6 校准协议（Stage-2 的对照基线）

- 32 帧，`lerobot/libero` 或 Molmo 对应集，**seed=42**，连续 flow-matching，每条样本冻结一对 `(noise, time)`。
- 九法公式在三条 FM 栈上一致，只换层 hook。Greedy-PPL 没有在 Molmo/SmolVLA 上打。
- 高分 = 保留。`importance_ranking`：按分数从高到低保留，Drop-k = 最低的 k 个。Molmo/π0.5 的 no-L0 是先从 `redundancy_order` 去掉 0 再取 k。
- 打分产物：π0.5 `results/ranks/ranks.json`；Molmo `results/molmoact2_libero/scores/`；SmolVLA `results/smolvla_libero/scores/`（`*.pt` 里有 `redundancy_order`）。

---

## 4. 当前问题（Stage-2 的起点）

Stage-1 的选层器用的是**均匀**校准帧。闭环已经证明「哪些层冗余」在这个均匀集合上成立，但用户认为关键 timestamp 对任务成败更重要，均匀 32 帧会把平庸帧和关键帧当成一样重。

Stage-2 的目标（用户 2026-10-04 定的，算法未选）：

**定义动作重要性，提高重要 timestamp 在校准集里的权重，让 Fisher/Taylor 测出更准的冗余层。**

「更准」的操作定义，写死，避免新会话用闭环去调权重：

1. 只改校准样本的**权重或抽样**，不改九法公式，不改 skip，不改 seed=1000 的闭环协议。
2. 用加权校准重打 **π0.5 的 Fisher、Molmo 的 Taylor、SmolVLA 的 Taylor**。先看 `redundancy_order` 前 k 是否变了。
3. 若与均匀校准的 Drop 集相同，写下来并停。不要为了同一层集再烧 400/2000 集。
4. 若不同，只闭环**新集合**，k 钉在 Stage-1 操作点：π0.5 k=6（且不含 L0）、Molmo k=16（不含 L0）、SmolVLA k=4。对照就是表里已经写好的 97.75 / 98.10 / 78.50。
5. 权重的选择依据只能是动作/轨迹本身或校准 loss，**不能**是这批闭环成功率。

DySL 的 trajectory continuity（帧间动作变化大 ⇒ 更重要）只是例子。开写代码前先调研，给用户 1–2 个候选和对照（均匀重加权 vs 不加权），**批准后再打分，打分结果不同再闭环**。

安全删层预算（估计当前动作最多能删几层）**不是这一阶段**。不要同时开。

---

## 5. 下一步计划

不要另起 seed、loss 或成功定义。新结果追加到对应 md，不要覆盖 Stage-1 的行。

### 现在就做

1. `nvidia-smi`。GPU1 若仍是 Molmo last-16，不要动。它写完 `eval_info.json` 后，把 Last-16 一行补进 `eval_molmoact2_libero.md`（层集 `[20–35]`，对照 Taylor Drop-16 98.10）。
2. 调研动作重要性定义。候选至少和 DySL continuity 分开写清：定义、需要哪些已有字段（action chunk / state / 时间）、会不会泄漏闭环标签。
3. 把选定定义先做成**打分对照**（均匀 vs 加权），输出新的 `redundancy_order` 和与 Stage-1 Drop 集的 Jaccard。不闭环。
4. 集合变了，再向用户确认要不要开闭环。一张卡一个 LIBERO。优先 π0.5（400 集，约 2–3 小时），不要一上来排 Molmo 的 17 小时。

### 明确不要做

- 重跑任何已有 `eval_info.json` 且 EXIT:0 的闭环。包括 π0.5 Drop-6/7/8/9、last-6、Skip L0；Molmo Drop-8/12/14/15/16/17；SmolVLA Drop-1/2/3/4；效率 JSON。
- 为了「找悬崖」再跑 SmolVLA Drop-5/8，或 π0.5 last-8，或 Molmo 只跳 L0。
- 在 Molmo/SmolVLA 上补 Greedy-PPL，或把 π0.5 Drop-3 九法跑完。
- 把中途停掉的 CosSim/PPL（Molmo、CogACT）重跑。
- 开 OFT 全量微调、训 CogACT、做 OFT「只跳前缀」、做技能路由。
- 用 `eval_s` 讨论加速。效率只认 `efficientcy_analysis.md`。
- 同一张 GPU 上同时跑两个 LIBERO 闭环。cuDNN 选择可能改 SR。别人的非 LIBERO 进程（历史上 GPU0/1 有 yany 的作业）不要杀，但不要再往那张卡上叠第二个 LIBERO。
- `tmux kill-server`。commit。改 git config。降级共享 lerobot 环境里的 MuJoCo（现为 3.8.1）。

---

## 6. 环境与机器

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/guorj/lerobot/src
```

- 栈 A / D / E：`/home/guorj/.miniforge3/envs/lerobot/bin/python`。
- OFT：`/home/guorj/.miniforge3/envs/openvla-oft/bin/python`，仓库 `/home/guorj/openvla-oft`。
- 栈 B：serve 用 `dtr_drop9_recovered/.venv`，client 用 lerobot。不要拷 `.venv`。
- 本机 4×RTX 3090 24GB。全量微调不在这台机器上做。
- 同一张卡两个 LIBERO 会互相影响成功率，禁止并卡。π0.5 / Molmo 加载后 1–3 分钟才出现在 `nvidia-smi` 里，这不是没挂上，不要因此再启动一份。

栈 A 的 `eval_drop.py` **禁止** `from __future__ import annotations`，且 `compile_model=false`。栈 B 的 Gemma 补丁必须在：dropped 层在 `use_cache` 时仍 `past_key_value.update(K,V)`。

---

## 7. 踩过的坑

1. **三种跳层不要混。** π0.5 Appendix B 留 K/V、丢 Q/O/MLP。Molmo/SmolVLA keep-KV：仍算 attention 以写出 K/V，Molmo 跳的是 MLP+ff_norm，SmolVLA 跳的是 VLM `o_proj`+MLP+post-LN。OFT/CogACT 是整层恒等。效率脚本里的 active params 必须按该栈真正跳过的子模块减，不能套 π0.5 的 Q/O/MLP 公式。
2. **同一 GPU 两个 LIBERO 闭环可能改 SR。** 不并卡。非 LIBERO 的小进程可以共存，但不要杀别人的进程。
3. **L0。** 只跳 L0 闭环是 0。高 k 先从 redundancy_order 去掉 0。GateProbe 的 L0 本来就排在末尾，所以它的 no-L0 Drop-7/8 等于 native，不要重跑。Taylor no-L0 的 Drop-7/8 与 Fisher 同集，已经复用，不要重跑。
4. **同层集合只评一次。** SmolVLA Drop-1 `[15]` 与 `[14]`、Drop-3 Fisher=IGIA、Molmo 多个 † 行，都是这个规则。
5. **SmolVLA 打分坑：** 图像必须 float 不能 uint8；action 要有 batch 维；noise 要 pad 到 `max_action_dim=32`。CosSim/CKA/GateProbe 不能 hook `DecoderLayer.forward`（这条栈不走它），要 hook `input_layernorm` 和 final RMSNorm。
6. **Molmo 加载：** `load_policy.py` 用本地 HF config 建图，权重来自 LeRobot 文件，避免下 20GB base shard。`num_flow_timesteps=1` 只用于打分脚本里的训练 loss；**效率与闭环的去噪步数是 10**（`flow_matching_num_steps`）。不要把打分用的 1 步当成推理步数。
7. **Hessian 24GB OOM、CKA 在 32/36 层上挤在 1e-6。** 没有分数就不要编层号。SmolVLA 的 Hessian 例外：16 层能跑完，Drop-4 已有 65.50。
8. **选层禁止看闭环 SR。** 停掉的作业（Molmo PPL Drop-16/17、CosSim、CogACT Drop-8）按用户当时的规则把剩余记 0，不要补跑，也不要写成完整 `eval_info.json`。
9. **墙钟随失败变长**是因为失败集跑满步数上限，不是挂死。成功的剪枝墙钟≈基线。
10. **不要 `tmux kill-server`。** 新作业用新 tmux 名。调度脚本若用全局 `pgrep` 等「任意 run_drop」，会误等另一张卡。要按 GPU + 用户 + 命令行里的模型名等。
11. **1 集 = 0.25（400 集）或 0.05（2000 集）。** 不要把 SmolVLA Taylor Drop-2 高于 Drop-1 解释成「少删更差」。
12. **效率测量：** 模块时间之和可以略小于整段 `predict_action_chunk`（Molmo 差在 packing / depth-gate）。表 1 的占比用三模块之和做分母。π0.5 表 1 的 Language 78.6 ms 与表 3 的 78.10 ms 是两次 CUDA event，各自文件都对，不要强行合成一个数。
13. **换 GPU 会改点。** 锚点永远用本机数字，不用 Hub/论文数字。

---

## 8. 关键文件

```
Skill_Efficient_VLA/HANDOFF.md
functional_necessity/
  main_results/eval_pi05_libero_v044.md
  main_results/eval_molmoact2_libero.md
  main_results/eval_smolvla_libero.md
  main_results/efficientcy_analysis.md
  scripts/pi05_libero/          # 栈 A：eval_drop.py, layer_skip.py, measure_*.py
  scripts/molmoact2_libero/     # 栈 D：score.py, layer_skip.py, load_policy.py, measure_efficiency.py
  scripts/smolvla_libero/       # 栈 E：同上 + adapt_lerobot_official.py
  results/eval_drop/            # 栈 A 闭环与效率 JSON
  results/molmoact2_libero/     # 栈 D；drop16_last/ 仍可能在写
  results/smolvla_libero/       # 栈 E；scores/*.pt, calib/calib_32_seed42.pt
  results/ranks/ranks.json      # 栈 A 排序
```

旁支表和脚本仍在 `main_results/eval_{openvla_oft,db_cogact,dtr_drop9_recovered}*.md`，不要当成 Stage-2 的起点。

---

## 9. 新会话开机检查

1. 读第 0、1.2、2、4、5 节，以及三张 FM 的 `eval_*.md` 和 `efficientcy_analysis.md`。不要凭记忆改数字。
2. `nvidia-smi`。GPU1 上的 Molmo last-16 没写完 `eval_info.json` 之前不要杀。不要抢别人的卡。
3. 不要重跑已有结果，不要 commit，不要 `tmux kill-server`，不要并卡跑两个 LIBERO。
4. Stage-2 先出「加权 vs 均匀」的排序差异，再问用户要不要闭环。
5. 简体中文。
