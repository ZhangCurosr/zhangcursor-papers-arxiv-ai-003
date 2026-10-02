---
title: "RLTL-DR-Self-improvement-by-Internalizing-Self-generated-Fee"
source: https://arxiv.org/pdf/2609.37633v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:19:05"
field: "大模型强化学习与自我改进"
keywords: ["RLVR", "Self-improvement", "Insight Internalization", "GRPO", "Tool-calling Agents", "SFT-Like Training"]
innovations: ["顺序自生成洞察+内部化SFT突破Pass@128=0学习壁垒", "仅用(task, insight)元组SFT可复现完整RL性能"]
benchmarks: ["Appworld", "Synthetic-API", "Leetcode"]
---

# 论文速读：RLTL;DR: Self-improvement by Internalizing Self-generated Feedback

## 一句话总结
论文提出了 **RLTL;DR** 方法，通过让模型在失败后生成高层 TL;DR 洞察并顺序迭代尝试，结合对洞察的 SFT 反传实现“任务→洞察”映射的内部化，从而突破了极低成功率（Pass@128=0）自改进场景下的 RL 学习瓶颈；且进一步发现仅用精简的 (任务, 洞察) 元组进行 SFT（SFTL;DR）即可复现绝大部分性能。

## 研究问题与动机
1. **RLVR 在极端困难任务上的信号饥荒**：当任务极难导致 Agent 在 128 次并行尝试中成功率接近 0 时，标准 GRPO 因所有 rollouts 获得相同负奖励导致优势函数坍塌（advantage collapse），学习信号为零，训练陷入停滞。
2. **缺乏教师模型与外部引导**：在自我改进（self-improvement）的前沿场景中，不存在更强的教师模型或金标准解可供蒸馏，Agent 必须依赖自身尝试探索并从失败中提取经验。
3. **现有引导探索方法的局限性**：既有方法多依赖外部特权信息（如完整参考答案、强教师提示）或在测试时移除引导后性能无法保持，且往往需要复杂的检索系统或重要性采样校正，难以迁移至无外部资源的纯自生成场景。
4. **上下文引导与测试时去耦的挑战**：即便在训练中利用失败经验引导探索成功，如何让模型在测试时（无上下文洞察、单次尝试）内化并泛化这些高层知识，是提升泛化能力的关键难题。

## 核心贡献（创新点）
1. **顺序自生成洞察的 sequential rollout 机制**：将传统并行 i.i.d. 采样改为顺序采样，每次失败后让策略自身基于验证器报错生成一条 TL;DR 级洞察作为后续尝试的上下文条件，从而在原本无学习信号的“绝境”中找到正向梯度。
2. **洞察内部化的 L_SFT 目标**：通过在已插入洞察的 rollouts 上激活反向传播掩码，训练模型直接从任务描述 $g$ 预测历史洞察 $\{f_i\}$，将“在此类任务中需记住此类事项”的知识直接固化到权重中，实现测试时无洞察单步推理的性能保持。
3. **SFTL;DR 范式发现**：证明完整的 GRPO  rollout 反传并非必需，仅使用去重的 (任务, 单句洞察) 元组进行 SFT 即可逼近 RLTL;DR 的全部性能，提出了一种计算代价极低、高度压缩的自改进训练范式。
4. **系统性消融与工程细节**：详细分析了洞察内容的抽象程度（TL;DR vs 详细诊断）、生成者（学生自生成 vs 教师模型）、上下文数量及损失权重对最终性能的影响，揭示了“高度抽象、去 session-specific 细节”是内化与泛化的关键。

## 方法详解
- **Sequential Self-generated Insights**：对每个困难任务，策略首先生成第一次 rollout $\tau_1$；若验证器判定失败，策略在独立上下文中基于完整 rollout 历史与 unit test 报错生成 JSON 格式的洞察（包含摘要、反馈、错误步骤、修正代码及最终的单句 TL;DR hint）。当该任务当前批次的平均成功率 $\le 50\%$ 时，将历史洞察 $\{f_1, \dots, f_I\}$ 以 user message 形式插入后续 rollout 的上下文 $\tilde{h}_t$ 中，形成顺序条件采样。
- **Insight Internalization (L_SFT)**：在包含洞察上下文的第 $k$ 次 rollout 反向传播阶段，除了正常的 GRPO 策略梯度外，额外开启对洞察 token 的反传掩码，施加监督损失 $\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{i=1}^{I} \log \pi_\theta(f_i | g, f_{<i})$，训练模型在看到任务 $g$ 时能直接输出对应的洞察内容。最终训练目标为 $\mathcal{L} = \mathcal{L}_{\text{GRPO}} + \lambda \mathcal{L}_{\text{SFT}}$，默认 $\lambda = 0.5$。
- **训练稳定性优化**：采用 DAPO 风格的 token-level policy gradients、移除优势函数中的标准差归一化、引入 positive-ratio filtering（确保 75% rollout 具有正优势）以缓解熵爆炸。
- **SFTL;DR 简化版**：固定收集到的 rollout 数据，完全移除 GRPO 损失，仅对去重后的 (任务 $g$, 洞察 $f$) 元组执行 next-token prediction SFT，无需在 rollout 生成步骤上进行前向/反向计算。

## 实验与结果
- **数据集**：Synthetic-API (SAPI，私有工具调用数据集，筛选出 Pass@128=0 的 458 个前沿困难任务)、Appworld（开源多步骤工具调用基准）、Leetcode（编程基准）。
- **基线**：标准 GRPO、SGE（Strategy-guided Exploration）、RLTF-SD（Self-distillation 变体）。
- **主要结果**：
  - 在 SAPI 前沿困难集上，标准 GRPO 保持在 0%-1% Pass@1；RLTL;DR 在无洞察测试条件下达到 **12-13% Pass@1**，显著突破学习瓶颈。
  - 在 Appworld test-challenge 上，RLTL;DR 达到 **61.7% Pass@1**（对比 GRPO 的 33.7%）。
  - Leetcode 上出现一定过拟合，但仍优于 GRPO。
  - **消融结论**：移除 $\mathcal{L}_{\text{SFT}}$（$\lambda=0$）性能骤降；仅保留 $\mathcal{L}_{\text{SFT}}$ 去除 GRPO 损失仍可恢复至 **17.0% Pass@1**（接近 RLTL;DR 的 18.9%）。
  - **SFTL;DR 结果**：仅用 4k 去重 (task, insight) 元组训练，在 SAPI eval 上达到 **16.9% Pass@1**，与完整 RLTL;DR 性能几乎一致，但反向传播 token 数从 12M 降至 68k。

## 相关工作脉络
1. **RLVR 与 GRPO 学习壁垒**：Lambert et al. (2024), Shao et al. (2024) 等提出的标准 RLVR 范式在零成功样本下面临优势坍塌，本文通过顺序自我反馈机制缓解该问题。
2. **特权信息引导探索**：如 RLTF-SD (Song et al., 2026) 利用 rollout 级别的 self-distillation，SGE (Szot et al., 2026) 利用失败摘要；本文与之不同在于仅提取高层抽象洞察且通过 SFT 直接内部化而非仅依赖上下文引导。
3. **Context Distillation / 上下文内部化**：Askell et al. (2021), Snell et al. (2022) 提出将上下文知识蒸馏至权重；本文将其应用于工具调用与代码生成的 agent 自我改进场景，且强调洞察格式的压缩性。
4. **Semantic Smoothness 与跨格式泛化**：Nakkiran et al. (2026), Cook et al. (2026) 发现模型可将一种格式的知识迁移至另一种格式；本文利用此现象实现从“任务描述→抽象洞察”到“任务描述→完整代码 rollout”的隐式能力迁移。
5. **Insight Database 检索增强**：Zhang et al. (2026a), Tang et al. (2026) 等构建外部洞察库并通过检索注入；本文主张通过 SFT 将洞察直接内化至模型参数，省去检索系统的复杂性。

## 局限性与未来方向
1. **领域适用性边界**：对于高度依赖特定技巧且缺乏通用规则的领域（如部分复杂数学证明、过于具体的 Leetcode 难题），(task, insight) 内部化可能无法有效泛化，甚至出现过拟合迹象。
2. **对预训练平滑性的依赖**：该方法依赖于模型具备足够大的预训练语义平滑性以支持跨格式知识迁移；对于小规模或预训练不足的模型可能失效。
3. **洞察生成质量依赖特权信息**：当前实验假设洞察生成器可访问验证器报错（privileged information）；若移除该信息，学生模型自生成洞察质量急剧下降，未来需探索无特权信息下的洞察生成。
4. **长 rollout 效率**：虽然 SFTL;DR 减少了更新计算，但洞察生成仍需完整 rollout 的前向计算，总 token 开销仍然巨大，未来可探索更高效的 rollout 采样或仅使用部分关键步骤。

## 研究启发与可借鉴点
1. **从并行到顺序的反馈闭环设计**：在极稀疏奖励场景下，将 i.i.d. 并行采样转为顺序依赖上一轮失败反馈的串行尝试，可有效打破零信号僵局，该思路可迁移至其他探索困难的 agent 任务。
2. **高层抽象信息的内部化范式**：证明只需训练模型预测“一句话建议”而非完整 rollout，即可实现知识内化，为后续研究提供了极低成本的正弦监督信号设计参考。
3. **去 session-specific 细节的洞察提取**：实验表明包含具体 ID、token 的详细诊断会阻碍泛化，而抽象的操作原则（如“记得分页”、“需确认权限”）更易被模型内部化，提示未来在构建 self-feedback 时应有意识地过滤实例级噪声。
4. **L_SFT 与 L_GRPO 的解耦分析**：通过消融分离出“探索信号”与“内部化信号”的贡献，揭示了即使无 GRPO 梯度也能通过高质量 SFT 目标实现显著性能提升，为混合训练策略设计提供依据。

## 关键术语表
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：基于可验证奖励的强化学习范式，利用单元测试或最终状态检查提供二元反馈，常见于代码生成与 agent 任务。
- **TL;DR Insight**：Too Long; Didn't Read 的缩写，指从长时间失败 rollout 中提炼出的单句高层改进建议，用于指导后续尝试。
- **Advantage Collapse**：在 GRPO 中当组内所有 rollout 获得相同奖励时，标准化后的优势值为零，导致策略梯度消失的现象。
- **Insight Internalization**：通过 SFT 损失使模型学会从任务描述直接生成对应洞察的能力，从而在测试时无需上下文即可隐式应用历史经验。
- **Deconfounded Evaluation**：在训练曲线与评估中排除上下文洞察带来的性能 Inflate，仅统计无洞察条件下的 Pass@1，确保指标公平可比。
- **Positive-ratio Filtering**：GRPO 训练中的稳定技巧，筛选出至少 75% 正奖励的 batch 进行更新，防止熵爆炸与策略坍塌。

## 可复现要素
- **数据集**：Appworld（公开）、Leetcode dataset（公开）、Synthetic-API (SAPI)（私有，论文未开源）。
- **代码/权重**：论文未明确开源代码仓库，但提供了详细的 Appendix B 复现指南（包括 prompt 格式、上下文构造逻辑、掩码设置示例）。
- **关键超参**：$\lambda = 0.5$（SFT 损失权重）；success rate threshold = 50%（决定是否插入洞察）；学习率 $3 \cdot 10^{-6}$ (Appworld/SAPI) 或 $6 \cdot 10^{-6}$ (Leetcode)；minibatch size 4 (SAPI) 或 1 (Leetcode)；gradient accumulation 8 或 4。
