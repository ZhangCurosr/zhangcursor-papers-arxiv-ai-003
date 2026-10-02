---
title: "RLTL-DR-Self-improvement-by-Internalizing-Self-generated-Fee"
source: https://arxiv.org/pdf/2609.37633v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:18:59"
field: "大语言模型强化学习与自我改进"
keywords: ["RLVR", "self-improvement", "reinforcement learning", "insight internalization", "GRPO", "agent", "tool calling"]
innovations: ["顺序自生成TL;DR洞察+任务到洞察的内部化SFT损失打破Pass@128=0学习停滞", "发现仅用(task,insight)元组做SFT可恢复大部分RL性能形成SFTL;DR紧凑范式"]
benchmarks: ["Appworld", "Synthetic-API", "Leetcode"]
---

# 论文速读：RLTL;DR: Self-improvement by Internalizing Self-generated Feedback

## 一句话总结
论文提出 RLTL;DR，通过在极难任务（Pass@128=0）上让模型失败后生成一行 TL;DR 式洞察并顺序插入后续 rollout，再用 SFT 损失将"任务→洞察"映射内部化，从而打破 GRPO 的学习停滞屏障，使 Qwen 3.5 9B Thinking 在工具调用与编码任务上 Pass@1 从 0–1% 提升至 12–13%。

## 研究问题与动机
- **RLVR 在极端困难任务上的信号枯竭**：标准 GRPO 依赖多 rollouts 间的相对优势，当所有 rollout 均失败时优势坍塌（advantage collapse），梯度为零，训练无法推进。
- **前沿自我改进场景缺乏教师信号**：在 agentic coding 等前沿任务上，不存在更强的 teacher 模型或 golden solution 用于蒸馏，策略必须从自身尝试中自主提取可迁移经验。
- **已有探索引导方法的缺陷**：SGE 等方法仅注入"上次失败"前缀而不反思 unit test 错误，甚至会误导后续 rollout；RLTF-SD 虽做自蒸馏，但目标是完整 rollout 而非高层洞察。
- **训练期利用洞察 ≠ 测试期无洞察可用**：即使顺序 rollout 期间能找到解，测试时单步无洞察仍接近原始性能，需一种将洞察知识压缩进权重的机制。

## 核心贡献（创新点）
1. **顺序自生成 TL;DR 洞察的 rollout 策略**：将 GRPO 的并行 i.i.d. rollout 改为顺序条件 rollout，每次失败后让策略自己生成一行洞察并插入后续尝试，突破探索屏障。与 SGE/RLTF-SD 的本质区别在于洞察是高层抽象规则而非完整诊断或尝试历史。
2. **task→insight 内部化机制（L_SFT）**：通过翻转反向传播掩码，在已存在的上下文上对 insight tokens 施加 SFT 损失，使模型学会仅从任务描述预测洞察，实现知识跨格式泛化。与 RLTF-FM 训练洞察生成器本身的本质区别在于直接内化映射而非蒸馏生成过程。
3. **发现 SFTL;DR 紧凑训练范式**：完全去掉 GRPO 损失、仅用 (task, insight) 元组（去重后仅 4k 条）做 SFT，可恢复 RLTL;DR 几乎全部性能，揭示"在此类任务上，记住这类事情"的高效训练形式。
4. **系统性的去混淆评估与洞察分析**：提出 macro-averaged Pass@1（无洞察上下文）、insight advantage、insight reliance 等指标，证明内部化确实泛化而非过拟合。

## 方法详解
- **顺序 rollout 与洞察插入**：对每个任务，策略首先生成 trajectory τ₁；若 verifier 失败，策略基于完整 rollout + 失败 unit test 输出 JSON（含 summary、feedback、wrong_step_id、corrected_code、TL;DR hint），仅提取最后一句 one-sentence 洞察 f₁。当当前 batch 运行平均成功率 ≤50% 时，将之前所有洞察 {fᵢ} 以 user messages 形式插入下一轮上下文：h̃ₜ = (g, f₁, …, f_I, …)，生成 aₜ ~ π_θ(·|h̃ₜ)。最多允许 16 条洞察。
- **L_GRPO**：标准 GRPO 损失，但不除以标准差（借鉴 Dr. GRPO），并加入 positive-ratio filtering（确保 75% rollouts 有正奖励）以稳定训练。
- **L_SFT 洞察内部化**：对每条已插入上下文的洞察，编辑 backprop mask 使其参与反向传播，训练 π_θ(f₁,…,f_I | g)，即仅用任务描述预测洞察 tokens（约 17 tokens/洞察）。总损失 L = L_GRPO + λ·L_SFT，默认 λ = 0.5。由于洞察已在上下文 h̃ₜ 中，L_SFT 不增加额外 forward 计算。
- **SFTL;DR 简化版本**：仅保留 L_SFT 项，不回传任何 rollout token；进一步去重为 4k 条唯一 (g, f) 元组逐个训练。

## 实验与结果
- **数据集**：Appworld（34 个 Pass@128=0 任务）、Proprietary Synthetic-API/SAPI（458 个 frontier-difficult 任务）、Leetcode（123 个 frontier-difficult 任务）。
- **基线**：GRPO（调优后）、SGE（Strategy-guided Exploration）、RLTF-SD（Self-Distillation 版本）。
- **主要结果（Table 1，heldout eval，无洞察上下文）**：
  - SAPI：GRPO 57.8% → RLTL;DR 91.1%（+33.3pp）
  - Appworld：GRPO 33.7% → RLTL;DR 61.7%（+28.0pp）
  - Leetcode：GRPO 55.1% → RLTL;DR 49.1%（下降，疑似过拟合）
- **Frontier 子集结果（Table 2，SAPI 458+184 任务，基线 Pass@1=4%）**：
  - RLTL;DR：Train 21.5%，Eval 18.9%
  - 仅 L_SFT（无 GRPO）：Train 20.0%，Eval 17.0%
  - SFT on 100 full rollouts：Eval 20.8%
  - SFTL;DR（去重 4k 元组）：Train 19.1%，Eval 16.9%（仅 68k backward tokens vs. SFT 217k）
- **洞察消融（Table 3）**：TL;DR 格式最优（Eval 19.2%）；更详细的 diagnostic paragraph + corrected code 反而下降；移除 failed unit tests 信息导致 Pass@1 暴跌至 3.4%；GLM 5.2 teacher 生成洞察小幅提升（Eval 20.5%）。

## 相关工作脉络
- **RLVR 优势坍塌问题**：Lambert et al. (2024) Tulu 3、Shao et al. (2024) GRPO、Yu et al. (2026) DAPO 均依赖并行 i.i.d. rollouts，在 Pass@K=0 时信号枯竭；本文通过顺序洞察打破该瓶颈。
- **特权信息引导探索**：Agrawal et al. (2026) Of-context GRPO、Zhang et al. (2025/2026) 注入 gold solution prefix、Chen et al. (2026b) 混合 guided/unguided rollouts；本文仅用失败 unit tests 作为唯一外部信号，且洞察为自生成高层规则。
- **SGE (Szot et al., 2026)**：仅注入失败尝试摘要而不反思具体错误，本文证明这会导致 contextual drag 并误导策略。
- **RLTF-SD (Song et al., 2026)**：对完整 rollout 做 self-distillation；本文改为仅对 insight tokens 做 SFT，证明高层抽象比完整示例更有效内部化。
- **Context distillation / 内部化**：Askell et al. (2021)、Snell et al. (2022)、Wang et al. (2026)；本文将 context 压缩为一行洞察并通过语义平滑泛化。
- **ECHO (Shrivastava et al., 2026)、Lu et al. (2026a)**：类似发现 backprop 通过 user message 中的 tokens 可提升策略能力，本文为此提供更系统的 ablation 与简化范式。

## 局限性与未来方向
- **Leetcode 上过拟合**：在编程推理任务上 RLTL;DR 未泛化（Eval 49.1% < GRPO 55.1%），可能因数学证明类任务缺乏可迁移的通用洞察规则。
- **对预训练光滑性的依赖**：(task, insight) 内部化可能是一种涌现能力，要求模型已有足够大的预训练基础；小模型或未充分预训练的 adapter 可能无效。
- **洞察质量门槛**：若任务本身极难到连生成有用洞察都需要与求解同等能力（如自动化科学研究），顺序 rollout 策略可能失效。
- **未测试不可训练的前沿模型场景**：本文聚焦可训练策略；对于 frozen frontier model，作者建议未来对比 insight database + retrieval 方案。
- **未来方向**：联邦学习场景下的洞察共享、人类书写反馈的吸收、超长 rollout 摘要的压缩训练。

## 研究启发与可借鉴点
- **紧凑训练范式 (task, insight) → SFT**：将复杂 rollout 经验压缩为一行高层规则再做 SFT，仅需 4k 元组即可恢复大部分性能，计算开销降低一个数量级，适合资源受限场景。
- **反向传播掩码编辑技巧**：无需修改 forward pass，仅通过编辑 backprop mask 即可对 context 中的 user message tokens 施加监督信号，实现零额外计算的成本内部化。
- **洞察详细度的 trade-off 设计**：过于具体的 diagnostic/corrected code 会引入 episode-bound literals（如 access_token、product_id），阻碍泛化；一行 procedural rule 更易内部化。这对设计 self-reflection 系统有直接指导意义。
- **与 RLTF-SD 结合**：将 L_SFT 添加到 RLTF-SD 的梯度掩码中可进一步提升性能（Figure 13），提示自蒸馏与洞察内部化可互补。
- **去混淆评估框架**：macro-averaged Pass@1、insight advantage、insight reliance 等指标体系可直接复用于其他含 context bootstrap 的 RL 方法比较。

## 关键术语表
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：通过可验证的 binary reward（如 unit test 通过/失败）对 LLM 策略进行强化学习的训练范式。
- **Advantage collapse / Learning cliff**：当一组 rollouts 全部成功或全部失败时，GRPO 的相对优势为零，导致梯度消失、训练停滞。
- **TL;DR insight**：一行（约 17 tokens）高层程序性规则，总结失败原因并指导下次尝试，如"Remember to paginate search results."
- **Insight internalization**：通过 L_SFT 将"任务→洞察"映射训练进模型权重，使模型在无洞察上下文的测试时也能隐式应用该知识。
- **Deconfounded evaluation**：训练曲线与评测均仅在无洞察上下文的 rollout 上计算 Pass@1，排除 context bootstrap 的直接贡献。
- **Insight reliance**：Xia et al. (2026) 提出的指标，衡量成功 trajectory 对 insight 的依赖程度，低依赖意味着知识已内化。
- **SFTL;DR**：RLTL;DR 的简化版本，仅保留 L_SFT 项，在 (task, insight) 元组上做 SFT，无需回传 rollout tokens。
- **Positive-ratio filtering**：训练稳定技巧，过滤 GRPO 批次确保 75% rollouts 具有正优势，防止熵爆炸。

## 可复现要素
- **数据集**：Appworld（开源）、Leetcode dataset（开源，Xia et al. 2025）；Synthetic-API (SAPI) 为 Apple 私有数据集，未公开。
- **代码/权重**：论文未明确声明开源，仅提供 Appendix B 的详细实现描述与 prompt 模板。
- **关键超参**：λ = 0.5（L_SFT 权重）、success rate ≤50% 触发洞察插入、max 16 insights、learning rate 3×10⁻⁶（Appworld/SAPI）或 6×10⁻⁶（Leetcode）、minibatch size 4（SAPI）/1（Leetcode）、gradient accumulation 8/4。
- **硬件**：8×B200 GPUs，单次训练 1–4 天。
- **模型**：Qwen 3.5 9B Thinking（Qwen Team, 2026）。
