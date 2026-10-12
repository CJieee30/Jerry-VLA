# HANDOFF：微调后 Flow-matching VLA 的 Training-free Layer Pruning

写给**完全没有上下文的新会话**。先读本文，再读根目录 `README.md` 和 `functional_necessity/main_results/` 里的表。**不要重跑**已完成的闭环。日期：**2026-10-11**。原机：`miaa-hpc2`，用户 `guorj`。回复用**简体中文**。不要 commit、不要 push，除非用户明确要求。用户自行把本仓库推到 `https://github.com/CJieee30/Jerry-VLA.git` 的 `main`。

**一句话现状：** 论文收成一篇发现（2026-10-10 与用户对齐，2026-10-11 因果数字已齐）。唯一主贡献是：微调后、零恢复、keep-KV 的 flow-matching VLA 上，删层算子必须和判据匹配；分布内深层可删，L0 必要，末层残差可恒等。CARE、D/S 时间戳、LEDGE、按任务选 k 都是负结果或分析。效率只作支撑。no-KV 与三模型只删 L0 的 `eval_info.json` 已齐，主表 md 还没改。还在跑的是 LIBERO-plus 的 Robot Initial States 与 Camera Viewpoints（full 与 Fisher Drop-6，第 4.1 节）。不要占这四张卡，不要再开别的闭环。

权威数字只信这些文件，不要凭记忆改表：

- `functional_necessity/main_results/eval_pi05_libero_v044.md`
- `functional_necessity/main_results/eval_molmoact2_libero.md`
- `functional_necessity/main_results/eval_smolvla_libero.md`
- `functional_necessity/main_results/care_libero.md`（Stage-2 / CARE）
- `functional_necessity/main_results/ledge_libero.md`（LEDGE）
- `functional_necessity/main_results/eval_nfe_libero.md`（NFE）
- `functional_necessity/main_results/eval_pi05_vision_module_libero.md`（视觉剪层，不进经营点）
- `functional_necessity/main_results/efficiency_analysis.md`（效率。旧稿里的 `efficientcy_analysis.md` 拼写已废弃，不要再建那个文件名）
- `README.md`（对外摘要；和本文冲突时以 `eval_info.json` 与本文为准）
- `functional_necessity/main_results/eval_libero_plus.md`（LIBERO-plus 目前只写入了 Language；Init / Camera 还在跑，不进分布内 OP 表）
- `functional_necessity/results/stage1_recovery_free/last_layer_identity.md`（末层开环恒等）
- `functional_necessity/results/stage2_care_v2/composition/composition.md`（D/S 时间戳负结果，只打分）
- 旁支，**不要写进 FM 主结论**（已移出 `main_results/`）：`results/openvla_oft_libero/eval_openvla_oft_libero.md`、`results/cogact_libero/eval_db_cogact_libero.md`、`results/dtr_drop9_recovered/eval_dtr_drop9_recovered.md`、`results/drop6_extra3/eval_pi05_libero_v044_task_aware.md`、`results/dtr_drop9_recovered/eval_dtr_drop9_recovered_task_aware.md`。OFT 的 CosSim 站住、Fisher/Taylor 为 0 只作「读末层隐状态」的反例，可以进讨论。

本文件是唯一给新会话的源。

---



## 0. 新会话 60 秒

1. 冲 **2026-11 左右的 CVPR**，按发现型论文写，不按新剪枝算法写。工作名：**Which Layers Does a Fine-tuned Flow-Matching VLA Actually Need? A Recovery-free Study of Depth Redundancy**。不要用 efficient / speedup 当标题或摘要第一句。用户已知这是 borderline，不是稳中。
2. 主论点见第 1 节。Stage-1 的答案已经有了：**不要再扫 k、不要再跑九法、不要再把 CARE 或 LEDGE 做成方法。**
3. 主栈只有三条，全是 keep-KV、seed=1000、本机锚点当裁判：π0.5 **97.25**（400 集）/ **96.80**（2000 集 NFE=10）、MolmoAct2 **97.50**、SmolVLA **83.00**。OFT 与 DB-CogACT 是整层恒等，数字不能和这三条横比。
4. 推荐操作点：π0.5 Fisher Drop-6 **97.75**（400）/ **97.10**（2000，NFE=10）；Molmo Taylor Drop-16（不含 L0）**98.10**；SmolVLA Taylor Drop-4 **78.50**（−4.50，用户已接受）。SmolVLA 的经营点是 Drop-4 × **NFE=2**，不是 NFE=1。
5. 论文主数字优先用 2000 集，写「持平」，不写「+0.50」。400 集与 2000 集不当成同精度比较。分布内 OP 表不要放 Plus、RoboTwin、视觉 k。
6. 协议、公式、skip 算子都不要改。新实验单独成行。PPL = **action Δloss**，不是 token perplexity。高分 = 保留。选层只用校准 loss，禁止用闭环成功率回搜层或回搜 CARE 的权重。
7. 成功率协议保持 eager（`compile_model=false`）。不再跑 CUDA graph、`reduce-overhead`、逐层 profiler、12 GB 显存上限。效率表只用已有的存储参数、峰值显存、FLOPs，外加空闲时的配对 eager。不要直接打开 `compile_model=true`。

---



## 1. Idea 总览（2026-10-10 与用户对齐）



### 1.1 主论点和反驳对象

给微调后、training-free、零恢复的 flow-matching VLA 立一套层冗余标准。主问题是：哪些语言层在功能上是必要的，删层判据必须和 keep-KV 算子匹配。省下的首先是语言前缀的计算量、存储参数和显存。eager 下同 NFE 的墙钟变化小，因为 action expert 的启动开销占满了时间；这是结构事实，不把「大幅降低 inference time」写成贡献。

现有层冗余分析要么服务于「删完再训」（DTR、CLP），要么只用单一表征判据（EfficientVLA、MoLe 用余弦）。空白是：微调后、零恢复、逐层读 K/V 的 flow-matching VLA 上，还没有人系统比较过功能判据和表征判据。

- [DTR](https://arxiv.org/abs/2606.27755) 原文写 “drop-only regime is insufficient for VLA models”。[VLM2VLA](https://arxiv.org/abs/2606.31382) 写现有判据在无恢复时 “total performance collapse”（那篇是参数级剪枝）。
- 我们的反例，全部无恢复：
  - π0.5 Fisher Drop-6 `[1,13–17]`：2000 集 **97.10**，完整模型 NFE=10 **96.80**。
  - MolmoAct2 Taylor Drop-16 `[1–4,24–35]`：**98.10**，完整模型 **97.50**。
- 同一设定下表征判据全崩：π0.5 CosSim **0.75**、CKA **2.00**；Molmo CosSim Drop-8 约 **0.35**；SmolVLA CosSim **30.50**、CKA **22.75**。连续删尾同样不行：Last-6 **73.50**，Last-16 **58.25**。



### 1.2 结构解释

- π0.5、Molmo、SmolVLA 的 expert 只读骨干每层的 K/V，不读骨干末层输出。所以第 i 层的 Q/O/MLP 只能经由更深各层的 K/V 影响动作。
- 代码核对（2026-10-09）：Molmo 连续动作路径只用 `past_key_values`；depth gate（`_depth_gate_from_condition`）读的是各层 V，不读 `last_hidden_state`。这条可以写成已核对。
- 末层开环已测完（32 帧，keep-KV vs 完整动作，NFE=1 和 10，`results/stage1_recovery_free/last_layer_identity.md`，`EXIT:0`）。π0.5 L17、SmolVLA L15 的 max-abs 都是 **0**。Molmo L35 均值约 **1.6e-3 / 3.7e-3**，和 fused-QKV 改成 K/V 两行 GEMM 的 bf16 噪声同量级；没有再做「仍走 fused QKV、只改残差」的对照。末层检查只服务结构解释，效率表不加「去掉末层」这一行。
- 这能解释三件事：尾部为什么可删；CosSim/CKA 为什么选错；OFT 这种读末层隐状态的架构为什么结论相反（OFT 上 CosSim Drop-4 **86.60**，Fisher/Taylor 为 0）。
- π0.5 Fisher Drop-6 不保留 K/V 已结束（`results/eval_drop/k6_fisher_nokv/eval_info.json`，400 集，没有早停）。Spatial **34** / Object **36** / Goal **34** / Long **1**，平均 **26.25**。同一层集的 keep-KV 是 **97.75**。图能跑，成功率掉约 70 分。Long 不是记出来的 0。不要覆盖 `k6_fisher`。主表 md 还没写入这一行。
- 只删 L0 已结束。π0.5 **0.00**（0/400）。MolmoAct2 Spatial **0**，其余三套按 `stopped.json` 记 0，平均 **0.00**（`results/molmoact2_libero/l0_only/`，每任务 10 集，不是历史 2000 集）。SmolVLA Spatial **11.0** / Object **13.0** / Goal **4.0** / Long **7.0**，平均 **8.75**（`results/smolvla_libero/l0_only/`）。三模型都可以写 L0 必要；SmolVLA 是大掉点，不是整表 0。Molmo / Smol 的主表 md 还没写入这一行。



### 1.3 贡献（2026-10-10：只留一条）

1. **无恢复深度冗余的系统研究和结构解释。** 内容：功能判据与表征判据的对比、连续删尾失败、平台到悬崖、keep-KV 与「读末层隐状态」的算子对照。小模型冗余少，SmolVLA 作边界（Drop-4 为 78.50，相对 83.00 低 4.50）。L0 与 no-KV 的 json 已齐（第 1.2 节），是这条贡献的因果，不是新方法。写入主表 md 之前先再读 json。

CARE 和 LEDGE **不再是贡献**。

- CARE：三个经营点的层集不变。近邻平局上会翻到已有更强集合。Molmo GateProbe Drop-17 的 7 维是 19.80 反例。D/S 时间戳实验（第 3.13 节）没有证明「多采突变帧就能测得更准」。降为附录分析。不要再做随机权重、反向权重或留出模型。
- LEDGE：规则在 π0.5 与 Molmo 上拟合，只省约一档闭环，没有留出模型。SmolVLA 没有平台。保持分析，不升成方法，也不再开留出模型。

效率不是贡献。论文里用现有数字做一张支撑表：存储参数、峰值显存、FLOPs、同 NFE 的配对 eager 时延，再加一行少步组合（π0.5 Drop-6 × NFE=1 相对完整 NFE=10）。NFE 引用已有工作。这一行用来举例：层剪枝可以和无训练的少步方法叠在一起，不把倍数写成我们的加速。

建议论文顺序：问题 → 发现（功能判据 vs CosSim/CKA vs Last-k，三模型无恢复成功率）→ 结构解释（末层、L0、no-KV、OFT 反例）→ 效率支撑表 → limitation（Plus 语言、RoboTwin、Smol 无平台）。CARE / D/S / LEDGE 放附录。

主张范围是分布内深度冗余。LIBERO-plus 语言维 Fisher Drop-6 相对 full **−3.96**（89.20 → 85.24，Long −7.57）。作者在 RoboTwin 上看到 Drop-6 显著下降；本仓库只有 Drop-2 还接近满模型（69.63 vs 71.75），不要把那台机器的数字写进 LIBERO 表，但 limitation 不能藏。

### 1.4 禁止再写

- 「VLA 领域没有人做过层冗余分析」，以及「第一个系统性分析」「首次 training-free layer pruning」。DTR、CLP、EfficientVLA、MoLe、DySL、VLM2VLA 都做过。只能写：微调后、零恢复、逐层读 K/V 的 flow-matching VLA 上，功能判据与表征判据还没有被系统比较。
- 「超过 1/3」。π0.5 正好 6/18。Molmo 是 16/36（44%）。
- 「SmolVLA 维持成功率」。Drop-4 是 78.50 对 83.00。
- 「CosSim 对 VLA 不合适」。收窄到「逐层读 K/V 的 flow-matching VLA」。OFT 是反例边界。
- 「CARE 大幅提高稳定性」，或把 CARE / LEDGE 写成第二条、第三条方法贡献。CARE 是三次近邻翻转加一次被闸门拦住的失败，经营点层集不变。D/S 时间戳也是否定结果。LEDGE 只省一档左右（π0.5 约 4→3 次，Molmo 约 5→4 次），没有留出模型。也不要写「按任务难度决定删几层」：Fisher f(task) 43.00 vs shuffled 44.50，Task/Skill-aware 不是主方法。
- 「深度冗余可以迁移到 LIBERO-plus / RoboTwin」。Plus 语言维是 −3.96，不是分布内的持平。只测了 Language，没有测另外六维。
- 「当前任务下的删层数」若指单个 LIBERO 任务或技能。LEDGE 的粒度是模型 × 基准。
- 「简单任务删 1/3、困难任务删 1/6」。任务感知实验和技能留出实验都没有支撑。
- 「NFE=1 是我们的贡献」「层剪枝大幅降低 inference time」「达到 SOTA 加速」。同 NFE eager 的墙钟变化小，要按 Amdahl 写。MoLe 的约 1.4×、EfficientVLA 的 1.93×、AdaVLA 的 1.87× 不能当失败线：它们的宿主瓶颈、减步或量化与我们不同。
- 把 CUDA graph、`torch.compile`、`reduce-overhead` 的时延写进主表。成功率协议保持 eager。
- 标题或摘要第一句以 efficient / speedup 开头。
- 「+0.50% 提升」。在噪声带内，写「持平」。主数字用 2000 集。
- 「π0.5 的 drop-16」。是 Drop-6。
- 「Fisher 永远最好」。Molmo Drop-16 上 Taylor 98.10，Fisher/IGIA 72.75。
- 「DySL 是 training-free」。它要 adapter + controller + KD。CARE 只借用其轨迹不连续统计量做离线校准权重。
- 「CLP 是 ICPR 的 CNN CKA」。指的是 VLA 那篇微调前 CKA + 下游微调。
- 把 OFT / DB-CogACT 整层恒等的崩盘写成 FM keep-KV 的结论，或写成「因为 LoRA」。
- 0.25–0.75 点的 Avg. 差是方法差异。π0.5 / SmolVLA 上 1 集 = 0.25（400 集）或 0.05（2000 集）。

Stage-1 里仍然站得住、可以进主文的句子：

- Recoverability ≠ functional necessity。CLP/DTR 的 CKA/GateProbe 服务「删完再训」；零恢复时 CosSim/CKA 在 π0.5、Molmo 上接近 0。
- 判据必须匹配删层算子。keep-KV：被删 VLM 层仍写 K/V，残差不更新，action expert 不删。
- L0 是结构必要层，不是打分噪声。π0.5 只跳 `[0]` = **0.00**（0/400）。Taylor 原生 Drop-6 含 L0 = 0.00；去掉 L0 后 `[1,2,14–17]` = **97.25**。
- 排名 ≠ 切尾巴。π0.5 last-6 `[12–17]` = **73.50**，Fisher 同 k `[1,13–17]` = **97.75**。Molmo last-16 `[20–35]` = **58.25**，Taylor 同 k `[1–4,24–35]` = **98.10**。
- 不存在跨模型、跨 k 的单一冠军。中等 k 用 Fisher；悬崖边用 Taylor。
- 独立 PPL / Greedy-PPL 在 keep-KV FM 上不够。π0.5 Drop-9：Greedy-PPL 47.00，独立 PPL 43.00，Fisher 76.75。不要把 OFT 上「greedy 更好」搬进主文。



### 1.5 Related work（必须写对）


| 工作                                                                                                                                           | 实际是什么                                                                             | 相对我们                                                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **EfficientVLA** (NeurIPS 2025, [arXiv:2506.10100](https://arxiv.org/abs/2506.10100))                                                        | 已训练官方 CogACT（SIMPLER）上 training-free LLM 层剪枝；`1−cosine`；另有视觉 token 与 action cache | 最直接竞争者。不能写「首次」。他们的宿主没有公开 LIBERO 微调权重。反驳是：同一类 `1−cos` 在逐层读 K/V 的 FM-VLA 上选层会崩 |
| **CLP** ([arXiv:2606.20246](https://arxiv.org/abs/2606.20246))                                                                               | 微调前用相邻层 CKA 找 twin block，永久删层后再 downstream FT                                     | 删完再训                                                                         |
| **DTR** ([arXiv:2606.27755](https://arxiv.org/abs/2606.27755))                                                                               | GateProbe 选可恢复块，必须 recovery FT；原文称 drop-only 对 VLA 不够                             | 同 CLP 一侧。栈 B 用的是他们 Drop-9 recovered ckpt。我们的反例是无恢复仍持平                        |
| **MoLe-VLA** (AAAI 2026)                                                                                                                     | 训练期把 LLM 层当 expert；评 RLBench                                                      | 需要训练                                                                         |
| **DySL-VLA** ([arXiv:2602.22896](https://arxiv.org/abs/2602.22896))                                                                          | cosine 分 static/dynamic；跳层靠 adapter + controller + KD；LIBERO 宿主是 OpenVLA-OFT      | 不是 training-free。CARE 只借用其轨迹不连续统计量                                           |
| **AdaVLA** ([arXiv:2608.29208](https://arxiv.org/abs/2608.29208))                                                                            | 冻结 π0.5 上免训练自适应步数 + MLP 剪枝；A10 289.7→158.9 ms，Jetson 1.87×                        | 最接近的免训练组合。要同协议正面比（P1）                                                        |
| **Decoupled Early Exits** ([arXiv:2609.29382](https://arxiv.org/abs/2609.29382))                                                             | 训练出口模块，联合调骨干深度、expert 深度、步数；时延 −79.2%、FLOPs −31.8%；称步数是 free lever                | 数字接近。区别：我们不训练、不加参数、选非连续层                                                     |
| **Let It Be Simple** ([arXiv:2606.05737](https://arxiv.org/abs/2606.05737))、**Coda** ([arXiv:2609.21216](https://arxiv.org/abs/2609.21216))  | 少步 / 一步 VLA                                                                       | NFE 不能当贡献的依据                                                                 |
| **VLM2VLA** ([arXiv:2606.31382](https://arxiv.org/abs/2606.31382))                                                                           | 无恢复参数级剪枝，保留约 90% 性能；现有判据会塌                                                        | 参数级，不是层。可引用其「无恢复时判据会塌」                                                       |
| **Williams & Aletras, ACL'24**；**Ji et al., ICLR'25** ([arXiv:2410.17711](https://arxiv.org/abs/2410.17711))；**Bandari et al., EMNLP'24**    | LLM 剪枝里的校准数据研究                                                                    | CARE 的区别：在轨迹内按动作结构给时间步加权                                                     |
| **LIBERO-Plus** ([arXiv:2510.13626](https://arxiv.org/abs/2510.13626))、**LIBERO-PRO** ([arXiv:2510.03827](https://arxiv.org/abs/2510.03827)) | LIBERO 已饱和；模型对乱码指令输出不变                                                            | 语言维已测：Fisher Drop-6 −3.96（`eval_libero_plus.md`）。相机等另外六维没测，不补，除非用户明确要求 |
| **AtomicVLA / SkillNet / SkillsCrafter**                                                                                                     | 技能标注或 action-expert / LoRA MoE                                                    | 只做过负对照。没有证明不同技能需要不同 LLM 层                                                    |


---



## 2. 实验栈（绝对不要混）

绝对值不能横比。删层算子不同。


|        | A. π0.5 v044                           | D. MolmoAct2                                          | E. SmolVLA                                                  | B. recovered（旁支）                                  | C. OpenVLA-OFT（旁支）                                               | CogACT（旁支）                 |
| ------ | -------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------- | -------------------------- |
| 权重     | `lerobot/pi05_libero_finetuned_v044`   | `allenai/MolmoAct2-LIBERO-LeRobot`                    | `lerobot/smolvla_libero`                                    | `s1ghhh/VLADrop_pi05_LIBERO_Drop9Block_GateProbe` | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | `Dexmal/libero-db-cogact`  |
| 代码     | `scripts/pi05_libero/`                 | `scripts/molmoact2_libero/`                           | `scripts/smolvla_libero/`                                   | `dtr_drop9_recovered/`（权重、calib、`.venv` 已删）            | 已删，见 git `e6ac6e4`                                    | 已删，见 git `e6ac6e4`   |
| Python | lerobot                                | lerobot                                               | lerobot                                                     | serve：`.venv`（已删，需重建）；client：lerobot                      | `openvla-oft`                                                    | 见 git 历史                       |
| seed   | **1000**                               | **1000**                                              | **1000**                                                    | **7**                                             | **7**                                                            | **7**                      |
| 闭环     | 4×10×**10** = 400；另有 2000 集            | 4×10×**50** = 2000                                    | 4×10×**10** = 400                                           | 400                                               | 500/套                                                            | 2000                       |
| 跳层     | Appendix B：留 prefix K/V，丢 Q/attn/O/MLP | keep-KV：只写 K/V，残差不更新；expert 不删（2026-10-09 起不再跑 Q/attention） | 被删 VLM 层只写 K/V；expert 不删 | 同 A                                               | **整层恒等**                                                         | **整层恒等**（DiT 读最后隐状态）       |
| 表      | `eval_pi05_libero_v044.md`             | `eval_molmoact2_libero.md`                            | `eval_smolvla_libero.md`                                    | `results/dtr_drop9_recovered/eval_dtr_drop9_recovered.md`                     | `results/openvla_oft_libero/eval_openvla_oft_libero.md`                                     | `results/cogact_libero/eval_db_cogact_libero.md` |
| 本机锚点   | **97.25**（400）/ **96.80**（2000，NFE=10） | **97.50**                                             | **83.00**                                                   | **95.00**                                         | **95.95**                                                        | **92.95**                  |


Stage-1 主文只用 A / D / E。B 用来说明「再全局删已经没有廉价层」。C / CogACT 用来说明整层恒等不是 keep-KV，**不要扩展**。OFT 上 CosSim 能站住，是「读末层隐状态」这条边界，可以进讨论，不要进 FM 主表。

π0.5 的 2000 集协议是每任务 50 集（`--eval.n_episodes=50`），数字在 `eval_nfe_libero.md` §7 和 `eval_pi05_vision_module_libero.md` §7。Stage-1 的 400 集行不要改，不要覆盖旧目录。

选层只用校准 loss。**禁止**用闭环成功率回搜层或回搜 CARE 的权重。

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
- **Skip L0 only** `[0]`**：0.00**（0/400）。墙钟 3h 28m，因为几乎每集都跑满步数。
- Drop-3 九法闭环**暂停**，没有 `eval_info.json` 的目录不要补数。不要为了补全九法去跑。



### 3.2 栈 D：MolmoAct2（36 层，0–35，keep-KV）

目录：`results/molmoact2_libero/`。锚点 97.50。Hub 98.25 不当裁判。

- Drop-8 `[28–35]`：Taylor/Fisher/IGIA/GateProbe 同集 **97.75**。PPL 50.15。Magnitude 32.15。CosSim 约 0.35（Long 未跑完，其余记 0，无 `eval_info.json`）。CKA 不评。Hessian OOM，没有层号。
- Drop-12：Taylor=Fisher **97.75**。
- Drop-14 no-L0：Taylor=IGIA **98.00**；Fisher 97.45。
- Drop-15 no-L0：Fisher=IGIA **98.10**；Taylor 97.70。
- Drop-16 no-L0：Taylor **98.10** `[1–4,24–35]`（删 L24、保住 L5）；GateProbe 87.00；Fisher=IGIA **72.75** `[1–4,5,25–35]`（删 L5、保住 L24）；PPL 中途 Spatial 已崩，剩余记 0。
- Last-16 对照 `[20–35]`：**58.25**（Spatial 74.4 / Object 72.2 / Goal 46.4 / Long 40.0，1165/2000，`EXIT:0`）。相对锚点 **−39.25**，相对同 k Taylor **−39.85**。36 层上同样不是切尾巴。
- Drop-17 no-L0：Taylor **89.60**；GateProbe 82.65；Fisher=IGIA 70.60。
- 悬崖：Fisher 在 15→16 掉约 25 点；Taylor 在 16→17 才掉。



### 3.3 栈 E：SmolVLA（16 层，0–15，keep-KV）

目录：`results/smolvla_libero/`。锚点 83.00（332/400），不是论文 87.3。用户已定性：**Drop-4 有冗余，成功率略降可接受。**

Taylor 的 k=1…4 就是切尾巴：`[15]` 79.25、`[14,15]` 80.50、`[13–15]` 77.75、`[12–15]` **78.50**。Drop-2 高于 Drop-1、Drop-4 高于 Drop-3 都是几集噪声。

Drop-4 全九法：Taylor 78.50；PPL 74.75；IGIA 73.75；GateProbe 72.00；Fisher 69.25；Hessian 65.50；CosSim 30.50；CKA 22.75；Magnitude 4.75。

Drop-1 同层合并，不要拆开重跑：`[15]` 覆盖 Taylor/GateProbe/PPL；`[14]` 覆盖 IGIA/Fisher。Drop-3 Fisher=IGIA `[10,12,14]`，74.75。

L15 在 PPL/GateProbe 上分数为 0：keep-KV 下专家读的是 L15 的 K/V，残差更新不进 action loss。这是架构结果，不是 bug。也是第 1.2 节结构解释的已有证据。

### 3.4 效率（`efficiency_analysis.md`）

RTX 3090，`compile_model=false`，CUDA event warmup=5 / repeats=20，一条校准观测，`FlopCounterMode`。Stored params 不变；表里的 Param 是 **active**（只减去该栈 skip 真正不跑的权重）。真实存储参数和峰值显存见第 4.2 节第 4 条，不要和这里的 active params 混。

- 模块拆分是**未剪枝 baseline**（NFE=10）。π0.5 Action 占时间 74.2%、FLOPs 只占 7.7%。Molmo Action 占时间 89.5%。SmolVLA Vision 占 FLOPs 72.0%，Language 只占 9.6%。
- 共享剪枝表（NFE=10）：π0.5 Fisher-6 active 3489.08（−15.79%），521.19 ms（−3.70%），3711.6 G（−26.13%），SR 97.75。Molmo Taylor-16 active 4246.78（−21.97%），1158.76 ms（−1.03%），4043.3 G（−22.99%），SR 98.10。SmolVLA Taylor-4 active 416.86（−7.37%），485.10 ms（−1.64%），586.7 G（−1.96%），SR 78.50。
- JSON：`results/eval_drop/efficiency_{pi05_baseline_modules,fisher_k6,pi05_drop6_language_only}.json`，`results/molmoact2_libero/efficiency_taylor_drop16.json`，`results/smolvla_libero/efficiency_taylor_drop4.json`。
- 经营点（少步 × 剪枝，§4，`results/stage_nfe/sota_efficiency_*.json`）见 3.12。视觉 k 不进这张经营点表。



### 3.5 旁支（完成即可，不要扩展、不要重跑）

- 栈 B extra Drop-1：最便宜 L2 **87.00**；L0 = 0。extra Drop-3 最好也只有 **18.75**。没有还能再全局删的廉价 3 层。
- Task / Suite / Skill-aware：层会变，但没有可迁移的 skill→layer。原始输出在 `results/eval_task_extra/`、`results/eval_skill_extra/`、`results/drop6_extra3/`、`results/drop9_skill/`。引用前读 `eval_info.json`。Turn/Place 的 GO 判定是 False（STOP）。Fisher f(task) 43.00 vs shuffled 44.50。不要写成「任务难度决定最大 k」。
- OFT 整层恒等：含 L0 的梯度法大量是 0；去掉 L0 后 Fisher Drop-4 仍 15.00。Greedy-PPL Drop-3/4/8 = 94.05 / 91.80 / 68.80。CosSim / PPL Drop-4 **86.60**。这是「删策略主干 / 读末层隐状态」，不是 keep-KV。
- DB-CogACT Drop-8：Fisher no-L0 与 GateProbe 在 Spatial 开头就是 0，剩余按停止规则记 0。CosSim 没评。不要再排队，除非用户把 P2 的 CosSim Drop-4/8 明确排上。



### 3.6 校准协议（Stage-2 的对照基线）

- 32 帧，`lerobot/libero` 或 Molmo 对应集，**seed=42**，连续 flow-matching，每条样本冻结一对 `(noise, time)`。
- 九法公式在三条 FM 栈上一致，只换层 hook。Greedy-PPL 没有在 Molmo/SmolVLA 上打。
- 高分 = 保留。`importance_ranking`：按分数从高到低保留，Drop-k = 最低的 k 个。Molmo/π0.5 的 no-L0 是先从 `redundancy_order` 去掉 0 再取 k。
- 打分产物：π0.5 `results/ranks/ranks.json`；Molmo `results/molmoact2_libero/scores/`；SmolVLA `results/smolvla_libero/scores/`（`*.pt` 里有 `redundancy_order`）。



### 3.7 CARE（已完成，权威表 `care_libero.md`）

CARE = Calibration with Action-discontinuity REweighting。动作不连续（DySL 统计量，离线、加在示范动作上；H=10，均值归一化到 1）重打分数。它是稳健性检查 + 近邻并列时的决胜，不是新的主选层器。平局闸门是方法的一部分：相邻层分差小于中位相邻差的 10%、或 7 维与臂 6 维选层不一致（Jaccard<1）时，不采用加权新集。

操作点 Drop 集全部不变，不要重跑：

- π0.5 Fisher Drop-6 no-L0 仍 `[1,13–17]`，Jaccard 1.0。α=1 时与原始 64×8 归约逐位相同。
- Molmo Taylor Drop-16 no-L0 仍 `[1–4,24–35]`（7d 与臂 6d 都是），Jaccard 1.0。
- SmolVLA Taylor Drop-4 仍 `[12–15]`，完整 redundancy_order 也相同，Jaccard 1.0。

次优三条的 k 切口各对换一层，四条均 `n_episodes=400`、`EXIT:0`。不要覆盖历史行：

- Molmo Fisher Drop-16：`[1–4,5,25–35]` → `[1–4,24–35]`（L5 ↔ L24）。新集 = 已评 Taylor Drop-16（**98.10**），不要重跑 2000 集。
- π0.5 GateProbe Drop-6：`[2,4,14–17]` → `[2,13–17]`（L4 ↔ L13）。原集重评 **94.25**，加权新集 **97.00**。目录 `results/eval_drop/k6_gateprobe_{reval,action_discontinuity}/`。不要覆盖历史 95.00。
- SmolVLA Fisher Drop-4：`[10,11,12,14]` → `[10,12,13,14]`（L11 ↔ L13）。原集重评 **69.75**，加权新集 **71.25**。目录 `results/smolvla_libero/drop4_fisher_{reval,action_discontinuity}/`。不要覆盖历史 69.25。仍低于 Taylor 78.50。

悬崖切口，全部已结束，不要重开：

- π0.5 Drop-8：Fisher/Taylor 仍 `[1–3,13–17]`，Jaccard 1.0，不要重评 85.50。GateProbe 新集闭环 **88.00** vs 历史 85.75。Drop-7 GateProbe 新集 **94.00** vs 历史 93.25（噪声）。
- Molmo Drop-17：Fisher/Taylor Jaccard 1.0，不要重评 70.60 / 89.60。GateProbe 7 维 **19.80**（−62.85，反例，闸门拦下）；臂 6 维 **97.65**。两种维度不一致，不提升任一 CARE 集取代 Taylor Drop-16。不要覆盖 Stage-1 GateProbe 82.65。

CARE IGIA（不开闭环）：π0.5 Drop-6 Jaccard 1.0（97.00）；Molmo Drop-16 仍删 L5、保 L24（**不是** Taylor 98.10；L5=104.28 vs L24=131.32）；SmolVLA Drop-4 Jaccard 1.0（73.75）。SmolVLA GateProbe Drop-4 仍 `[7,13–15]`，Jaccard 1.0，不要重评 72.00。

CARE 到此停止。不要再扩定义、换窗口、做随机权重或反向权重。不能用闭环 SR 调 α 或 H。

### 3.8 LEDGE（`ledge_libero.md`）

漂移只决定测哪些 k，k 由区间内闭环读出。禁止用漂移直接给出 k，禁止跨模型或跨序套用 σ。

- π0.5 Fisher CARE：σ=0.60，区间 [3, 8]，闭环测得最大 k=7（96.00；k=8 为 85.50）。
- Molmo Taylor CARE：σ=0.079，区间 [9, 19]，闭环测得 k=16（98.10；k=17 为 89.60）。
- 相对「全序二分」，大约省一档（π0.5 4→3 次，Molmo 5→4 次）。
- SmolVLA 没有平台，前提不成立，LEDGE 不给它的最大删层数。
- 视觉 LEDGE 区间不可用，不设定视觉 k。见 `eval_pi05_vision_module_libero.md` §4。
- 没有在留出模型上验证区间。2026-10-10 决定：不升成主方法，也不再开留出模型。



### 3.9 NFE（`eval_nfe_libero.md`）

400 集，除非另行注明。NFE=1 是独立 Euler 调度（dt=−1/NFE），不是把 10 步轨迹提前截断。

- π0.5：完整模型 NFE=1 **97.00**；Drop-6 NFE=1 **96.50**。
- Molmo 400 集：完整模型 NFE=1 **98.25**；Drop-16 NFE=1 **99.25**。不能和 2000 集锚点当同精度比。
- Molmo 2000 集 NFE=1 的 `eval_info.json` 已在（2026-10-10 核对），还没确认是否写入 `eval_nfe_libero.md` §1。写入前再读 json，不要凭本文改表：完整模型 **97.80**（Spatial 97.4 / Object 99.4 / Goal 97.4 / Long 97.0）；Taylor Drop-16 **97.60**（97.8 / 98.6 / 96.8 / 97.2）。目录 `results/stage_nfe/molmoact2_{full,drop16_taylor}_nfe1_2000/`。论文用这两格替换 400 集的 Molmo NFE=1 行。
- SmolVLA：完整模型 NFE=2 **82.00**、NFE=1 **79.25**；Drop-4 NFE=2 **77.00**。经营点是 NFE=2。
- π0.5 Drop-8 NFE=5 **86.00**。减步救不回悬崖（Goal 仍在 74 附近）。
- 余弦（π0.5）：相邻步速度 **0.9988**，首末步 **0.9789**，独立 NFE=1 与 NFE=10 的动作 **0.9970**。图在 `main_results/figures/nfe_cosine_`*。
- 另一台机器上的 RoboTwin NFE=1 掉了约 15 分。目录 `another_server_results` **不进论文表**，只当警示：LIBERO 上的少步持平不要外推。



### 3.10 2000 集 π0.5（`eval_nfe_libero.md` §7，`EXIT:0`）


| 设置                      | 成功率              |
| ----------------------- | ---------------- |
| 完整模型 NFE=10             | 96.80（1936/2000） |
| Drop-6 NFE=10           | 97.10（1942/2000） |
| Drop-6 NFE=1            | 95.70（1914/2000） |
| Drop-6 NFE=1 复跑        | 96.25（1925/2000） |
| Drop-6 + 视觉 k=2 + NFE=1 | 96.50（1930/2000） |
| Drop-6 + 视觉 k=4 + NFE=1 | 95.20（1904/2000） |


相对各自 400 集，先跑的三行大约低 0.45–0.80，主要掉在 Long。Drop-6 NFE=10 vs 完整模型 NFE=10 是 97.10 vs 96.80。写持平。复跑相对先跑的 95.70 是 **+0.55**，当波动，不写成提升，也不覆盖 `pi05_k6_fisher_nfe1_2000/`。数字在 `eval_nfe_libero.md` §7.1。视觉 k=4 在 `eval_pi05_vision_module_libero.md` §8，Long 90.2，离开 k=2 的平台。

### 3.11 视觉剪层，Stage 5（`eval_pi05_vision_module_libero.md`）

做法：在语言 Fisher Drop-6 上整块恒等跳过 SigLIP 层，按 Fisher 7 维排序。不作为第五阶段，不进经营点效率表。

- k=2 `[20,18]`：NFE=10 **97.50**；NFE=1 400 集 **96.25** 和重复 **96.00**；NFE=1 2000 集 **96.50**。
- k=4 `[20,18,21,17]`：NFE=10 **95.75**；NFE=1 400 集 **96.00**；NFE=1 2000 集 **95.20**（1904/2000，Long 90.2）。离开 k=2 的平台，不进经营点。
- k≥8 离开平台。k=9 / k=10 为 90.00 / 86.25。
- Taylor 把 SigLIP L0 排进前列，含 L0 的闭环塌掉（k=4 平均 8.50 量级）。去掉 L0 后回到 95 附近。SigLIP L0 是关键层。
- 视觉 Drop-4 效率：视觉模块 56.84 → 48.58 ms。整段 chunk 在 NFE=10 下省约 6.8 ms（−1.3%），在 Drop-6 × NFE=1 下省 **8.8 ms**（−5.6%）和 97 G。不写入 `efficiency_analysis.md` 的经营点表。



### 3.12 效率核对（经营点，`efficiency_analysis.md` §4）

测量脚本 `scripts/stage_nfe/measure_sota_efficiency.py`，`compile_model=false`。Molmo 另关 `enable_inference_cuda_graph`。下面「相对完整 NFE=10」是表内数字；「同 NFE」是用同一张表的两行相减，论文里剪枝收益按同 NFE 报。


| 经营点                    | 时间       | vs NFE=10 完整 | FLOPs            | SR          |
| ---------------------- | -------- | ------------ | ---------------- | ----------- |
| π0.5 Drop-6 × NFE=1    | 155.9 ms | −70.3%       | 3361.6 G（−33.1%） | 96.50（400）  |
| Molmo Drop-16 × NFE=1  | 240.4 ms | −79.4%       | 3969.3 G（−24.4%） | 99.25†（400） |
| SmolVLA Drop-4 × NFE=2 | 181.8 ms | −67.3%       | 498.5 G（−16.7%）  | 77.00（400）  |


同 NFE 的剪枝收益（eager）：

- π0.5 NFE=1：155.9 vs 181.1 ms（**−13.9%**），3361.6 vs 4674.4 G（**−28.1%**）。
- Molmo NFE=1：240.4 vs 253.4 ms（**−5.1%**），3969.3 vs 5176.7 G（**−23.3%**）。
- SmolVLA NFE=2：181.8 vs 173.8 ms（**慢 4.6%**，噪声），FLOPs 498.5 vs 510.3（−2.3%）。交错 A/B 曾测到 135.24 vs 135.66 ms，符号不稳定。不要把 SmolVLA 的同 NFE 时延写成加速。

测量本身和表一致：语言模块省下的毫秒数能对上整段 chunk 的差值。整段百分比小，是因为 eager 下 expert 循环占满了时间。π0.5 语言时间大致跟上语言 FLOPs（约 −33%）。§4 的 Molmo 语言时间只约 −16%，因为当时被删层仍跑完整 attention。K/V-only 之后见 `efficiency_analysis.md` §5：语言 FLOPs −42.2%，语言时间 −20.0%，同 NFE=1 整段仍约 −5.7%。

分析结论（不在效率表的单元格里，新会话引用前要能从原始计时复现）：每个被删层的残余开销，π0.5 约 0.23 ms（完整层约 4.34 ms），Molmo 约 1.44 ms（完整层约 2.26 ms，约 64% 还在）。3090 BF16 **稠密**峰值约 71 TFLOP/s（142 是稀疏峰值，不要再用）。语言前缀约 51 TFLOP/s。expert 每步的带宽下限约 1.2–1.5 ms，eager 实测约 38 ms（π0.5）和约 102 ms（Molmo），九成以上是启动开销。

### 3.13 2026-10-10 新增（不要重跑已结束的）

**末层开环恒等。已结束。** `results/stage1_recovery_free/last_layer_identity.md`。keep-KV，32 帧，NFE=1 和 10。π0.5 L17、SmolVLA L15：max-abs = 0。Molmo L35：mean max-abs 1.621e-3（NFE=1）/ 3.719e-3（NFE=10），判定阈值 1e-4 下不是逐元素恒等。π0.5 末层 `k_proj` 在 full 与 drop 下各调用 64 次。no-KV 的开环 smoke：skip\[17\] L2 = 0.205，skip\[0\] L2 = 16.9。末层残差专家不用，末层 K/V 专家在用。

**LIBERO-plus Language。已结束，八个作业 `EXIT:0`。** 表在 `main_results/eval_libero_plus.md`。category `Language Instructions`，每变体 1 trial，`seed=1000`，keep-KV。full **89.20**（1367/1537），Fisher Drop-6 `[1,13–17]` **85.24**（1306/1537），Δ **−3.96**。套件：Spatial −4.10、Object −1.98、Goal −2.20、Long −7.57。不进分布内 OP 表。smoke `plus_smoke_full_lang_spatial10` 不进表。

**LIBERO-plus Robot Initial States 与 Camera Viewpoints。2026-10-11 用户批准后已开跑，见第 4.1 节。** 协议与 Language 相同。另外四维（Sensor Noise、Objects Layout、Light Conditions、Background Textures）没批准，不要自行开。

**D/S 时间戳。已结束，只打分，没有新闭环。** `results/stage2_care_v2/composition/composition.md`，`EXIT:0` 2026-10-10。同一 480 集：U 均匀、D 突变、S 平滑，等权，不用 α。Taylor 在三个经营点上不变。Molmo Fisher-16：D 映射到已评 Taylor-16 **98.10**，S 映射到已评 Fisher-16 **72.75**。方向和「突变帧更能找出冗余」相反。π0.5 Fisher-6 的 U/D 映射到已评 97.00 集合，S 映射到已评 97.75 集合。不要为 D/S 再开闭环。

**高 α GateProbe Drop-6。已结束，噪声。** drop `[2,3,14–17]`，400 集 **97.25**（389/400），`results/eval_drop/k6_gateprobe_v2_hialpha/`。不要覆盖 Stage-1 的 95.00 或 CARE 加权的 97.00。

**不保留 K/V、只删 L0。已结束，不要重跑。** 数字在第 1.2 节。目录是 `results/eval_drop/k6_fisher_nokv/`、`results/molmoact2_libero/l0_only/`、`results/smolvla_libero/l0_only/`。不要覆盖 `k6_fisher` 和 Molmo / Smol 的历史 eval。主表 md 尚未追加这些行。

---



## 4. 当前问题



### 4.1 正在跑（2026-10-11 09:59 UTC 启动，不要杀、不要并卡、不要覆盖目录）

no-KV 与两格 L0 已结束，数字在第 1.2 节。下面四路是唯一还在跑的闭环。每路按 Spatial → Object → Goal → Long 顺序跑，每变体 1 trial。启动时都在 Spatial。


| tmux | GPU | 设定 | 维度 | 变体数 | 目录前缀 |
| --- | --- | --- | --- | ---: | --- |
| `plus-full-init-g0` | 0 | full | Robot Initial States | 1550 | `plus_full_init_{spatial,object,goal,10}` |
| `plus-full-cam-g1` | 1 | full | Camera Viewpoints | 1599 | `plus_full_camera_{spatial,object,goal,10}` |
| `plus-d6-init-g2` | 2 | Fisher Drop-6 `[1,13–17]` | Robot Initial States | 1550 | `plus_fisher6_init_{spatial,object,goal,10}` |
| `plus-d6-cam-g3` | 3 | Fisher Drop-6 `[1,13–17]` | Camera Viewpoints | 1599 | `plus_fisher6_camera_{spatial,object,goal,10}` |


目录在 `results/eval_libero_plus/`。脚本：`scripts/pi05_libero/run_eval_plus_dim.sh`，单套件入口 `run_eval_plus_lang.sh`（第 5 个参数是 category）。已有 `eval_info.json` 的套件会跳过。某一套失败则该路停止，不要手动再开第二份。Language 的八个目录不要动。

### 4.2 已解决，不要再当缺口

1. 同 NFE eager 的墙钟变化小：π0.5 −13.9%，Molmo −5.1%（§4，改 K/V-only 之前）。这是结构事实，不再把大幅降时延当贡献。
2. K/V-only 已接上（2026-10-09）。数字在 `efficiency_analysis.md` §5。SmolVLA 动作与旧 skip 逐位相同。Molmo fused QKV 改成 K/V 两行 GEMM 后，校准帧上 NFE=1 动作最大差 7.8e-3。历史闭环不重跑。
3. 成功率协议是 eager。不要改成 compile。
4. 存储参数和峰值显存已在 `results/stage_nfe/deployment_*.json`：π0.5 allocated 约 −1.24 GB，Molmo 约 −2.92 GB，SmolVLA 约 −0.07 GB。论文报 allocated。
5. 末层开环恒等已做完（第 3.13 节）。Molmo depth gate 已核对：读各层 V，不读 `last_hidden_state`。
6. LIBERO-plus 语言维已做完。Robot Initial States 与 Camera Viewpoints 是用户在 2026-10-11 明确批准的，正在跑。其余四维不要自行开。
7. CARE / D/S / LEDGE / 按任务选 k 已定性为负结果或分析。不再补随机权重、反向权重、留出模型，也不要为 RoboTwin 做任务级 LEDGE。
8. Molmo NFE=1 的两格 2000 集、no-KV、Molmo/Smol 只删 L0 都已有 `eval_info.json`。写入对应 md 之前先核对 json。

### 4.3 仍开放，但不再为此开实验

1. 缺 π0.5 完整模型 NFE=1 的 2000 集。现有 97.00 是 400 集。论文少步行用已有 2000 集的 Drop-6×NFE=1（96.25 复跑）对完整模型 NFE=10（96.80），不要为了补这一格再排 6 小时。
2. no-KV 与三模型 L0 的 json 已齐，主表 md 还没改。写正文时从 json 抄，不要凭 README 或本文改已有的 keep-KV 行。
3. Molmo L35 的 1e-3 还没有 fused-QKV 对照。可以在正文里写成与 bf16 GEMM 噪声同量级，不要再开实验，除非用户要求。
4. RoboTwin：作者侧 Drop-6 显著下降；本仓库 Drop-2 为 69.63 vs 71.75。`another_server_results` 不进表。少步持平和深度冗余都不要外推到 LIBERO 以外。
5. 分布内冗余的主张到此为止。Plus −3.96 和 RoboTwin 掉点写 limitation。

---



## 5. 下一步计划

不要另起 seed、loss 或成功定义。新结果追加到对应 md，不要覆盖 Stage-1 的行。不再跑 CUDA graph、`reduce-overhead`、逐层 profiler、12 GB 显存上限。效率不再单独立项扩实验。**不要再开新的闭环或新的选层方法。**

### 现在做

1. **让第 4.1 节四路 Plus 跑完。** 不要杀，不要在同一张卡上再塞一个 LIBERO。四张卡都占着。
2. **Plus 两维 `EXIT:0` 之后写入 `eval_libero_plus.md`。** 追加 Init / Camera 的表，不要改 Language 的 89.20 / 85.24。没有 `eval_info.json` 不要编成功率。
3. **把已结束、但还没进主表 md 的因果行写进去。** no-KV 追加到 `eval_pi05_libero_v044.md`，不要改 Fisher keep-KV 的 97.75。L0 追加到 Molmo / Smol 的表，Molmo 写明 Spatial 为 0 后其余记 0。Molmo NFE=1 2000 集写入 `eval_nfe_libero.md` §1，替换 400 集那两行。写入前再读 json。
4. **然后写正文，不写新实验。** 顺序见第 1.3 节。根目录 `README.md` 已有对外摘要。CARE / D/S / LEDGE 放附录。

### 明确不要做

下面是 2026-10-10 起额外冻结的。再下面的历史禁止项仍然有效。

- 把 CARE、D/S、LEDGE 重新做成方法，或补随机权重、反向权重、留出模型、LEDGE 的 π0/GR00T。
- 再测 LIBERO-plus 其余四维（Sensor Noise、Objects Layout、Light Conditions、Background Textures）、LIBERO-PRO、语言 Drop-8、Molmo/Smol 的 no-KV Drop-k、末层 keep-KV 的 400 集闭环、π0.5 完整模型 NFE=1 的 2000 集、Molmo K/V-only Drop-16 × NFE=1 的新 2000 集。Init / Camera 是已经在跑的例外，不要再开第二份。
- 用闭环 SR 调任何权重。把 Plus 或视觉 k 写进分布内 OP 表。把 `another_server_results` 写进表。为 RoboTwin 做按任务预测最大 k 的 LEDGE。
- 占 GPU0–3，只要第 4.1 节的四个 tmux 还在。

### GPU

同一张 GPU 上不要同时跑两个 LIBERO 闭环。GPU0 历史上有 yany / sunxy。不要杀别人的进程。不要 `tmux kill-server`。第 4.1 节的四个 tmux 名不要复用，已结束的 `molmo-l0-only-g0`、`smol-l0-only-g1`、`pi05-k6-fisher-nokv-g3` 也不要拿来重跑。

### 历史禁止项（仍然有效）

- CUDA graph 捕获、`torch.compile`、`reduce-overhead`、逐层 profiler、12 GB 显存上限测试。已有 §6 里的 graph 和 `reduce-overhead` 数字不进主表，也不要重跑。
- 为了把同 NFE 墙钟做成 SOTA 加速而改成功率协议或再扫 k。
- 重跑任何已有 `eval_info.json` 且 EXIT:0 的闭环。包括 π0.5 Drop-6/7/8/9、last-6、Skip L0；Molmo Drop-8/12/14/15/16/17、last-16；SmolVLA Drop-1/2/3/4；CARE 已完成的四条 400 集和 Drop-7/8/17；NFE 已完成行；视觉已完成行；效率 JSON。
- 重跑 CARE 动作不连续的操作点打分，或为 Jaccard=1 的同一层集开闭环。不要重跑已变成 Taylor Drop-16 的 Molmo CARE Fisher。不要重跑 π0.5 CARE Taylor / Molmo CARE GateProbe / 三栈 CARE IGIA / SmolVLA CARE GateProbe（已有 `.pt`）。不要为 π0.5 Fisher/Taylor Drop-8、Molmo Fisher/Taylor Drop-17、CARE IGIA 的 97.00 / 72.75 / 73.75、或 SmolVLA GateProbe 72.00 开闭环。
- 覆盖 `k6_gateprobe` / `k6_fisher` / `k7_gateprobe_nol0` / `k8_gateprobe_nol0` / `drop4_fisher`，以及任何已有 `eval_info.json` 的旧目录。`k6_fisher_nokv`、`molmoact2_libero/l0_only`、`smolvla_libero/l0_only` 已结束，不要重跑，也不要把它们写进旧目录。第 4.1 节的 Plus 新目录不要写进 `plus_*_lang_*`。
- 为了「找悬崖」再跑 SmolVLA Drop-5/8，或 π0.5 last-8。
- 在 Molmo/SmolVLA 上补 Greedy-PPL，或把 π0.5 Drop-3 九法跑完。
- 把中途停掉的 CosSim/PPL（Molmo、CogACT）重跑。
- 开 OFT 全量微调、训 CogACT、做 OFT「只跳前缀」、做技能路由。
- 用闭环 SR 调 CARE 的 α 或 H。用漂移直接给出 k。跨模型或跨序套用 σ。把 LEDGE 写成按任务或按技能定 k。
- 把视觉 k、prefix trim、视觉 token 剪枝并进删层或写进经营点效率表。继续扩展视觉剪层。
- 跑语言 Drop-8。删 action expert。
- 用 `eval_s` 讨论加速。效率只认 `efficiency_analysis.md`。
- 直接打开 `compile_model=true` 来做效率表或成功率表。
- 把 `another_server_results` 写进论文表。
- 同一张 GPU 上同时跑两个 LIBERO 闭环。cuDNN 选择可能改 SR。别人的进程不要杀，测速也不要和别人的进程挤在同一张卡上。
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

`HF_ENDPOINT` 必须带 `https://`。漏协议时测量会卡在 huggingface.co。

- 栈 A / D / E：`/home/guorj/.miniforge3/envs/lerobot/bin/python`。
- OFT：`/home/guorj/.miniforge3/envs/openvla-oft/bin/python`，仓库 `/home/guorj/openvla-oft`。
- 栈 B：serve 用 `dtr_drop9_recovered/.venv`，client 用 lerobot。不要拷 `.venv`。
- 本机 4×RTX 3090 24GB。全量微调不在这台机器上做。
- 同一张卡两个 LIBERO 会互相影响成功率，禁止并卡。π0.5 / Molmo 加载后 1–3 分钟才出现在 `nvidia-smi` 里，这不是没挂上，不要因此再启动一份。
- GPU0 历史上有 yany（FITS）和 sunxy（ts2audio）。不要杀。新作业和测速先看 `nvidia-smi`。

栈 A 的 `eval_drop.py` **禁止** `from __future__ import annotations`，且 `compile_model=false`。栈 B 的 Gemma 补丁必须在：dropped 层在 `use_cache` 时仍 `past_key_value.update(K,V)`。

---



## 7. 踩过的坑

1. **三种跳层不要混。** π0.5、Molmo、SmolVLA 现在都是 Appendix B：被删层只写 K/V，残差不更新，expert 不删。OFT/CogACT 是整层恒等。`efficiency_analysis.md` §1–§4 是改之前的测量，Molmo/SmolVLA 当时仍跑 Q 和 attention。§5 才是 K/V-only。两套 active params 不能混用。
2. **同一 GPU 两个 LIBERO 闭环可能改 SR。** 不并卡。非 LIBERO 的小进程可以共存，但不要杀别人的进程。测速要干净的卡。
3. **L0。** 只跳 L0 闭环是 0。高 k 先从 redundancy_order 去掉 0。GateProbe 的 L0 本来就排在末尾，所以它的 no-L0 Drop-7/8 等于 native，不要重跑。Taylor no-L0 的 Drop-7/8 与 Fisher 同集，已经复用，不要重跑。SigLIP 的 L0 同样是关键层，Taylor 会把它排成冗余，删了视觉就崩。
4. **同层集合只评一次。** SmolVLA Drop-1 `[15]` 与 `[14]`、Drop-3 Fisher=IGIA、Molmo 多个 † 行，都是这个规则。
5. **SmolVLA 打分坑：** 图像必须 float 不能 uint8；action 要有 batch 维；noise 要 pad 到 `max_action_dim=32`。CosSim/CKA/GateProbe 不能 hook `DecoderLayer.forward`（这条栈不走它），要 hook `input_layernorm` 和 final RMSNorm。
6. **Molmo 加载：** `load_policy.py` 用本地 HF config 建图，权重来自 LeRobot 文件，避免下 20GB base shard。`num_flow_timesteps=1` 只用于打分脚本里的训练 loss；**效率与闭环的去噪步数是 10**（`flow_matching_num_steps`）。不要把打分用的 1 步当成推理步数。
7. **Hessian 24GB OOM、CKA 在 32/36 层上挤在 1e-6。** 没有分数就不要编层号。SmolVLA 的 Hessian 例外：16 层能跑完，Drop-4 已有 65.50。
8. **选层禁止看闭环 SR。** 停掉的作业（Molmo PPL Drop-16/17、CosSim、CogACT Drop-8）按用户当时的规则把剩余记 0，不要补跑，也不要写成完整 `eval_info.json`。
9. **墙钟随失败变长**是因为失败集跑满步数上限，不是挂死。成功的剪枝墙钟≈基线。
10. **不要** `tmux kill-server`**。** 新作业用新 tmux 名。调度脚本若用全局 `pgrep` 等「任意 run_drop」，会误等另一张卡。要按 GPU + 用户 + 命令行里的模型名等。tmux 名会被前缀匹配：`visk-g3` 匹配到了 `visk-g3b`。新作业用不会互为前缀的名字。
11. **1 集 = 0.25（400 集）或 0.05（2000 集）。** 不要把 SmolVLA Taylor Drop-2 高于 Drop-1 解释成「少删更差」。400 集标准误约 0.8 分，2000 集约 0.4 分，两档协议不要写成同一个 Δ。
12. **效率测量：** 模块时间之和可以略小于整段 `predict_action_chunk`（Molmo 差在 packing / depth-gate）。表 1 的占比用三模块之和做分母。π0.5 表 1 的 Language 78.6 ms 与表 3 的 78.10 ms 是两次 CUDA event，各自文件都对，不要强行合成一个数。经营点 §4 又是一轮重测（语言 79.6 ms），三套数字各归各的文件。
13. **换 GPU 会改点。** 锚点永远用本机数字，不用 Hub/论文数字。`another_server_results` 不进表。
14. **不要直接打开** `compile_model=true`**。** π0.5 会把 `sample_actions` 和 `forward` 用 `torch.compile(mode=max-autotune)` 包起来，运行时换的删层钩子可能进不去，Drop-6 会静默变成完整模型；这个开关还会把 FP32 矩阵乘改成 TF32。π0.5 推理还强制 eager attention。Molmo 的 `compile_model` 只编译 `action_expert.blocks`。`reduce-overhead` 会改动作，CUDA graph 不再作为论文主表，已有 §6 数字不要重跑、不要写进主结果。成功率协议继续 eager。
15. **3090 的 BF16 稠密峰值是约 71 TFLOP/s。** 142 是稀疏峰值。不要把语言前缀的利用率写成「峰值的三分之一」。稠密峰值下前缀大约七成。
16. **解析 tqdm 日志不能直接数** `50/50`**。** tqdm 会重印。在 `0/50` 处重置，每个任务只记第一次 `50/50`。按 CR 切分 “Stepping through eval batches” 那一行。
17. `HF_ENDPOINT` **必须是** `https://hf-mirror.com`**。** 测量卡在 huggingface.co 上时，杀掉后改走镜像重跑，不要空等。
18. **余弦探针的 batch 切片是** `(row, row+1)`**。** `slice_batch(raw, row, 1)` 是错的。
19. **Molmo 的 NFE=1 2000 集约 16–17 小时，不是 400 集墙钟乘 5。** 共享卡上大约每小时 2.7 个任务。π0.5 的 NFE=1 2000 集约 6–7.5 小时（已完成的 Drop-6 × NFE=1 是 6.34 h）。
20. **NFE=1 不是 10 步路径的提前截断。** 是独立调度，步长 dt=−1/NFE。论文里的余弦用的是独立调度之间的动作，不是把一条 10 步轨迹砍短。
21. **SmolVLA 同 NFE 的时延符号不稳定。** Drop-4 × NFE=2 可以比完整模型 NFE=2 更慢。FLOPs 约 −2%。不要写成加速。
22. **Molmo 加载后要等 1–3 分钟才出现在** `nvidia-smi`**。** 不要以为没挂上再启动一份。

---



## 8. 关键文件

```
Skill_Efficient_VLA/README.md
Skill_Efficient_VLA/HANDOFF.md
functional_necessity/
  main_results/eval_pi05_libero_v044.md
  main_results/eval_molmoact2_libero.md
  main_results/eval_smolvla_libero.md
  main_results/care_libero.md                 # Stage-2 / CARE
  main_results/ledge_libero.md                # LEDGE
  main_results/eval_nfe_libero.md             # NFE 闭环与余弦
  main_results/eval_pi05_vision_module_libero.md
  main_results/efficiency_analysis.md         # 效率。不要用 efficientcy_analysis.md
  main_results/eval_libero_plus.md            # Plus 语言维，limitation
  main_results/figures/nfe_cosine_*
  results/stage1_recovery_free/last_layer_identity.md
  results/stage2_care_v2/composition/composition.md   # D/S 负结果
  scripts/pi05_libero/run_eval_k6_fisher_nokv.sh
  scripts/pi05_libero/run_eval_plus_lang.sh
  scripts/pi05_libero/run_eval_plus_dim.sh
  scripts/molmoact2_libero/run_l0_only.sh
  scripts/smolvla_libero/run_l0_only.sh
  scripts/pi05_libero/          # 栈 A：eval_drop.py, layer_skip.py, measure_*.py
  scripts/molmoact2_libero/     # 栈 D：score.py, layer_skip.py, load_policy.py, measure_efficiency.py
  scripts/smolvla_libero/       # 栈 E：同上 + adapt_lerobot_official.py
  scripts/stage_nfe/            # NFE 闭环、经营点效率、视觉效率
  results/eval_drop/            # 栈 A 闭环与效率 JSON
  results/molmoact2_libero/     # 栈 D
  results/smolvla_libero/       # 栈 E；scores/*.pt, calib/calib_32_seed42.pt
  results/ranks/ranks.json      # 栈 A 排序
  results/stage2_action_weight/ # CARE 权重与加权打分
  results/stage3_ledge/         # LEDGE 曲线
  results/stage_nfe/            # 2000 集、NFE、视觉、sota_efficiency_*.json
```

旁支表和脚本仍在 `main_results/eval_{openvla_oft,db_cogact,dtr_drop9_recovered}*.md`，不要当成 CARE 或新主线的起点。

---



## 9. 新会话开机检查

1. 读第 0、1、4、5 节、根目录 `README.md`，以及三张 FM 的 `eval_*.md`、`eval_libero_plus.md`、`eval_nfe_libero.md`、`efficiency_analysis.md`、`last_layer_identity.md`。CARE / LEDGE 的 md 只作附录材料。不要凭记忆改数字。
2. `nvidia-smi` 和 `tmux ls`。先看第 4.1 节四个 Plus tmux 是否还在。不要抢这些卡，不要杀别人的作业。
3. Plus 两维结束后按第 5 节写入 `eval_libero_plus.md`。no-KV、L0、Molmo NFE=1 2000 集的 json 已在，写入主表 md 前再读 json。没有 `eval_info.json` 或 `stopped.json` 不要编成功率。
4. 不要重跑已有结果，不要 commit，不要 push，不要 `tmux kill-server`，不要并卡跑两个 LIBERO，不要再开新闭环。
5. 下一步是等第 4.1 节结束、把已齐的 json 写入 md，然后写正文。不是打开 `compile_model=true`，也不是再扫 k，也不是把 CARE / LEDGE 做成方法。
6. 简体中文。

