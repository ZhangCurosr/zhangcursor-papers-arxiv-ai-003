---
title: "What-Does-Post-Training-Change-in-Multilingual-Reasoning"
source: https://arxiv.org/pdf/2609.37104v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:58:15"
field: "多语言大语言模型推理"
keywords: ["multilingual reasoning", "post-training audit", "reinforcement learning", "language adherence", "delivery efficiency", "chain-of-thought"]
innovations: ["提出J=C∧L∧T交付指标揭示correct-but-undelivered缺口", "证明SFT准确性成本为通用现象而非多语言混合特有", "设计正确性门控语言奖励R_gate以同时恢复终止性与语言遵循"]
benchmarks: ["AIME-2026", "AMC-2023", "DAPO-Math-17K", "OpenR1"]
---

# 论文速读：What-Does-Post-Training-Change-in-Multilingual-Reasoning

## 一句话总结
本文对 Qwen3 系列模型在多语言数学推理上的后训练路径进行了阶段性审计，揭示了解题能力（correctness）与多语言交付（delivery）之间存在高达 94 个百分点的差距；多语言 SFT 修复了语言遵循但引入非终止循环，而带有正确性门控的语言奖励 RL 可同时恢复终止性与语言遵循，使非英语交付效率达到与英语相当的水平。

## 研究问题与动机
- **现有评估掩盖了"可用但不可达"的能力**：最终答案准确率会统计用英语写出可见推理的正确解答，无法反映用户请求语言下的完整交付，导致 multilingual reasoning gap 被低估。
- **语言成为访问屏障而非性能扰动**：相同内容在不同语言下占用 token 数差异显著（token tax），且非英语推理更易出现非终止循环，造成不平等访问成本。
- **不同失败模式未被统一追踪**：prior work 分别报告 accuracy、language adherence、efficiency、termination 等指标，但未在同一模型家族、同一协议下逐阶段刻画主导瓶颈的迁移路径。
- **RL 奖励设计的语言效应未被系统对比**：仅奖励 correctness 会诱导模型回到英语，加入 language term 又可能被错误回答利用，如何设计才兼顾两者尚无清晰结论。

## 核心贡献（创新点）
- **提出以交付为中心的评估框架 J = C ∧ L ∧ T**：将 correctness、language adherence、termination 三个分量联合报告，暴露出 released checkpoint 上最高 94 个百分点的"correct-but-undelivered"性能缺口。
- **诊断 SFT 带来的准确性损失是通用 cost 而非多语言混合特有**：通过 English-only SFT 和 single-language specialist 对照发现，accuracy decline 在所有 SFT 分支上一致出现，归因于 post-training 阶段的 prior capability 被 overwrite。
- **揭示 RL 阶段的主导瓶颈从 termination 转移到 accuracy，且 reward 设计决定交付语言**：仅 reward correctness 使模型回归英语（adherence 跌至 8.7%），而正确性门控的语言奖励（R_gate）以无 accuracy 代价恢复 termination 并维持 99.8% 语言遵循。
- **建立一条可复现的后训练构造路径**：从 English-pivoted released checkpoint → multilingual SFT → additive RL → correctness-gated RL，逐阶段消除瓶颈并将非英语交付效率提升至英语的 102%（adjusted）。

## 方法详解
- **评估协议**：在 70 道竞赛数学题（30 AIME-2026 + 40 AMC-2023）上，覆盖 UN 六官方语言（English/Chinese/Spanish/French/Arabic/Russian）及五个训练语料外语言；每道题 16 个 sample、temperature 0.7、24,576-token 响应预算。
- **三分量定义**：
  - **C（correctness）**：通过 `\boxed{}` 提取最终答案并调用 `math_verify`。
  - **L（language adherence）**：抽取可见 `<think>` 块去除数学表达式后，按句子切分并在最多 60 个片段上均匀抽样，使用 langid 判定目标语言占比过半则 L=1；辅以盲评 LLM judge 校验（Pearson r=97.9%）。
  - **T（termination）**：未触发响应 cap 且 zlib 压缩重复分不超过 0.85；区分 budget exhaustion 与 detected looping 两类失败。
  - **J = C ∧ L ∧ T** 为响应级联合成功事件，按 pass@k 无偏估计报告。
- **SFT 数据构造**：以 45K 条 English OpenR1 长 CoT 数学轨迹为基础，由本地部署的 Qwen3-14B 翻译为五目标语言（优于 Hunyuan-MT-7B），经 7 道 quality gate（结构完整性、数字保留、长度比、重复检测、脚本纯度/英语功能词比例等）过滤，得到 43,218（4B）/ 56,307（8B）条轨迹。
- **RL 奖励设计（三种 formulation）**：
  - `R_corr = C_tr`：仅奖励训练时正确性（control）。
  - `R_add = 0.7·C_tr + 0.3·ℓ_tr`：累加型，错误回答仍可获得最高 0.3 的语言分数。
  - `R_gate = C_tr · (0.7 + 0.3·ℓ_tr)`：门控型，错误回答奖励为零，语言 credit 必须由正确解答获取。
- **交付效率度量**：`joint_per_lk(ℓ) = 1000 × #J=1 / #adjusted_tokens`，按语言编码因子做归一化；以英语为基准计算相对效率 `Eff. vs. En`，100% 表示 parity。

## 实验与结果
- **Released checkpoint 的交付鸿沟**：Qwen3-4B 在非英语五语言上 C@16 = 90.6%，但 J@16 = 35.7%；Qwen3-8B 为 92.6% vs. 30.6%。Spanish/French/Arabic 在 4B 上语言遵循仅为 0.1%/0%/0%，8B 上完全为零。跨 10 种非英语语言（含训练语料外的 Icelandic/Swedish/Swahili/Tamil/Urdu），4B 为 84.3% vs. 17.9%，8B 为 89.0% vs. 15.4%，最大 gap 达 94 个百分点（Swedish on 8B）。
- **SFT 修复语言遵循但暴露终止瓶颈**：多语言 SFT 后 4B 的 L 从 39.8% 升至 99.4%，J@16 从 35.7% 升至 84.6%；8B 从 30.6% 升至 86.3%。然而非英语终止失败从 4.7% 升至 21.1%（4B），主要由 detected looping 驱动；同期 C@16 下降约 5–6 个百分点，且 English-only SFT 与 single-language specialist 均呈现相似的准确性退化，说明该 cost 非多语言混合特有。
- **RL 分臂对比揭示奖励设计的关键作用**：correctness-only RL 虽将非英语终止失败降至 0.3%，但 L 跌至 8.7%、J@16 仅 15.4%，模型回归英语；additive RL 将非英语终止失败降至 0.5%、L 保持 99.8%，但错误轨迹仍获 63.1% 的奖励质量；correctness-gated RL 在保持 C@1 ≈ 60.1% 的同时实现 L = 99.8%、J@16 = 81.7%。
- **交付效率达英语 parity**：经 SFT 后非英语相对效率从 47%（4B）/ 27%（8B）升至 73%/78%；gated RL 后进一步提升至 102%（adjusted，4B）。 conditioning on delivered responses 时 raw-token 效率在 gated endpoint 达到 100% parity。
- **问题语言比指令更强**：即便经过 SFT + gated RL，以英语题干附目标语言指令仍几乎无法 eliciting 目标语言推理（Chinese/Spanish/French/Arabic 均为 0–2.5%），仅 Russian 在 SFT 后达 94.2%、但在 gated RL 回落至 9.2%。

## 相关工作脉络
- **英语中心表征与可见推理**：Wendler et al. (2024)、Schut et al. (2025) 揭示内在表征的 English-pivot；Tam et al. (2025) 指出可见推理链默认倾向高资源语言。本文在同一模型链路上量化了该现象对交付的直接影响（gap 最高 94 pp）。
- **翻译壁垒假说**：Bafna et al. (2025) 提出多语言生成受隐式翻译失败制约；本文通过 SFT + RL 的阶段性干预证明，问题不仅在于 output-side 渲染，更在于 reasoning trace 本身的语言归属。
- **输入/输出两端翻译缺失**：Kan et al. (2026) 强调非英语 prompt 未能进入 English-dominant 推理链；Qi et al. (2025) 发现要求目标语言推理会降低准确率。本文通过 specialist vs. multilingual SFT 对照，将该 tradeoff 归因为 general SFT cost 而非 mixing 特有。
- **Long CoT 跨语言研究**：Barua et al. (2026)、Gurgurov et al. (2026) 对比 English-pivoted 与 target-language reasoning traces；本文在此基础上提供了一站式 stage-by-stage 的瓶颈迁移图谱。
- **多语言 RL 与一致性奖励**：Zhan et al. (2026)、Fan et al. (2026) 引入 consistency-enhanced objectives；Elhady et al. (2026) 使用 cross-lingual self-consistency 无监督奖励。本文对比的 R_gate 以显式语言项 + 正确性门控实现了类似目标但更简明的实现。
- **评估基准**：PolyMath（Wan et al.，2025）、MTM-Bench（Zhan et al.，2026）分别报告 per-language accuracy 和语义/语言分解；本文的 J = C ∧ L ∧ T 进一步纳入 termination 与 delivery efficiency，补齐端到端可用性度量。

## 局限性与未来方向
- 结论局限于 Qwen3 家族与竞赛数学任务，released checkpoint 已历经充分数学 post-training，干预目标是 delivery 而非 general reasoning capability 的提升。
- 每个训练运行仅使用单一 seed，评估仅 70 道翻译题；English-only 与 specialist SFT 在数据量与训练配置上存在差异，比较属诊断性而非完全 matched causal。
- 翻译可能改变数学问题的自然表达与难度，尽管 Pearson 相关（0.819–0.988，均值 0.907）表明难度结构被较好保留。
- gated RL 在 response cap 与 batch size 上与 additive 阶段不同，91 个百分点的 adherence gap 无法完全排除超参贡献。
- 未来方向：探索不依赖翻译的端到端多语言推理训练；在更多模型家族与任务域验证瓶颈迁移规律；改进错误回答的语言 credit 收集问题（additive phase 中 44%–82% 奖励流向错误轨迹）。

## 研究启发与可借鉴点
- **分阶段瓶颈审计框架**：以 J = C ∧ L ∧ T 为联合指标，逐训练阶段记录主导失败模式的迁移，可作为多语言/多能力后训练的通用诊断模板。
- **门控奖励设计的通用性**：`R_gate = C · (α + β·ℓ)` 在避免错误轨迹"搭便车"方面表现优异，可迁移至其他需要同时优化任务性能与约束条件（如安全、格式、语言）的 RL 场景。
- **SFT accuracy cost 的对照诊断方法**：通过 English-only SFT 与 single-language specialist 分离"多语言混合"与"SFT 本身"的贡献，为后续工作提供可控归因范式。
- **交付效率的多会计惯例**：adjusted vs. raw tokens、all-attempt vs. J-only、token yield 四维度量体系，为跨语言系统的 cost-effectiveness 评估提供精细化工具。
- **可结合本团队方向**：若团队关注低资源语言建模或多语言 agent 系统，本研究的路径（SFT 修复遵循 → RL 修复终止 → gated reward 锁定交付）可直接复用于长程推理 agent 的多语言部署流程。

## 关键术语表
- **Delivery（交付）**：响应同时满足正确性、目标语言遵循与终止性的联合状态，以 J = C ∧ L ∧ T 衡量，比单一准确率更能反映多语言可用性。
- **Language adherence（语言遵循）**：可见推理 trace 中被判定为目标语言的比例，本文以 langid 过半片段判定，并以盲评 LLM judge 交叉验证。
- **Termination failure（终止失败）**：响应因达到 token budget 或陷入重复循环而无法完成，分为 budget exhaustion 与 detected looping 两类。
- **Correctness-gated reward（正确性门控奖励）**：`R_gate = C_tr · (0.7 + 0.3·ℓ_tr)`，仅在答案正确时发放语言 bonus，防止错误轨迹窃取语言奖励。
- **Delivery efficiency（交付效率）**：每 1,000 个生成 token 所产生的联合成功响应数，按语言编码因子调整后可与英语 parity 比较。
- **OpenR1**：Hugging Face 维护的 DeepSeek-R1 完整开源复现项目，本文以其 45K 条长 CoT 数学轨迹作为 SFT 语料来源。
- **DAPO-Math-17K**：基于 DAPO 开源 RL 系统的数学提示集，本文以其 16,754 条本地化提示进行 RL 训练。
- **Pass@k（pass@k 估计）**：在无偏意义下估计 k 次采样中至少一次成功的概率，本文用于 C@16 与 J@16 的计算。

## 可复现要素
- **数据集**：70 道竞赛数学题（30 AIME-2026 + 40 AMC-2023）；SFT 语料 43,218（4B）/ 56,307（8B）条 OpenR1 轨迹（经 Qwen3-14B 翻译）；RL 语料 16,754 条 DAPO-Math-17K 提示。论文未明确声明外部数据集公开链接，但补充材料包含完整 corpus statistics 与 quality audit。
- **代码/权重**：所有 generated traces 与 per-sample measurements 随 accompanying artifact 发布；analysis scripts 为 parameter-free entry points 并附 machine-readable outputs 与 run logs。模型权重为 Qwen3 开源权重（released checkpoints）。具体代码仓库与 artifact 链接见论文 supplementary/B.5 与 accompanying artifact。
- **关键超参**：SFT 学习率 2×10⁻⁵（cosine schedule）、bfloat16、batch 16（multilingual）/ 4（controls）、A100-SXM4-40GB × 16/4；RL 使用 clipped PPO + GAE（γ=λ=1）、actor LR 1×10⁻⁶、critic LR 1×10⁻⁵、clip ratio 0.2、rollout per prompt 8、train batch 32/64（gated）、max steps 1,500（evaluated at 700/200/1,000）。完整配置见 Appendix B Tables S6/S7。
