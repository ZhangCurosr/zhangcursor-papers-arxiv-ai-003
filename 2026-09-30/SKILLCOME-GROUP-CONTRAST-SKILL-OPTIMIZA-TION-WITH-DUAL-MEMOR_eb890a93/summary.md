---
title: "SKILLCOME-GROUP-CONTRAST-SKILL-OPTIMIZA-TION-WITH-DUAL-MEMOR"
source: https://arxiv.org/pdf/2609.37128v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:21:22"
field: "LLM技能演化与自我改进"
keywords: ["skill evolution", "group contrast", "dual memory", "LLM agents", "non-parametric optimization", "trajectory analysis"]
innovations: ["组rollout与同题成败对比优化，提供可靠优化信号", "双记忆系统跨步骤积累支持证据，防止单步噪声主导", "证据导向的技能修订与优先级排序机制"]
benchmarks: ["SpreadsheetBench", "SearchQA", "LiveMathematicianBench", "ALFWorld", "OfficeQA", "DocVQA", "HotpotQA", "Omni-MATH"]
---

# 论文速读：SKILLCOME-GROUP-CONTRAST-SKILL-OPTIMIZA-TION-WITH-DUAL-MEMOR

## 一句话总结
SkillCome 是一种基于组对比优化与双记忆系统的技能演化方法，通过为每个问题生成多条轨迹并进行成功/失败对比分析，结合跨步骤积累的历史证据，实现对 LLM 智能体技能的高效迭代优化。

## 研究问题与动机
- 现有技能演化方法通常为每个问题只生成一条轨迹，难以从单一轨迹中判断哪些行为导致了失败、哪些行为促成了成功，优化信号不可靠。
- 现有方法仅基于当前局部批次进行轨迹分析，有价值的技能模式可能分散在多步训练中，单步证据不足，易受噪声干扰。
- 相比参数优化已发展出的成熟训练目标（如 DPO、GRPO）和优化器状态（如 Adam、AdamW），技能演化缺乏可靠的优化信号构造机制和状态记忆机制。

## 核心贡献（创新点）
- **组 rollout 与组内对比优化**：每个问题生成 n 条轨迹形成同一技能的对比组，通过比对成功与失败轨迹识别关键行为分歧，提供比单轨迹反思更可靠的优化信号。与 SkillOpt 等方法相比，本质区别在于从"独立反思每条轨迹"转向"同问题成败对比分析"，直接锁定因果差异。
- **双记忆系统（Failure Memory + Contrast Memory）**：分别积累跨步骤的失败模式和同组对比中发现的可推广行为模式，支持集大小反映支撑问题的数量而非轨迹数量。与已有方法相比，本质区别在于记忆不仅存储历史结论，还持续关联新证据并作为排名优先级的依据，防止单步噪声主导优化方向。
- **系统实验验证与效率优势**：在六个基准上对五个不同规模模型进行广泛评估，平均较无技能基线提升达19.55%，相对第二名 SkillOpt 提升5.69分；同时因轨迹去重机制，优化器 token 消耗较 SkillOpt 减少约50%~64%。

## 方法详解
- **组 Rollout（Grouped Rollout）**：在第 t 步优化中，从训练集采样 m 个问题，每个问题在相同技能 $s^t$ 下独立生成 n 条轨迹 $\tau_{ij}^t \sim M_{\text{tar}}(\cdot|x_i^t, s^t)$，按成功率 $r_{ij}^t = R(x_i^t, \tau_{ij}^t)$ 分类，形成成功池 $\mathcal{T}^{t,+}$、失败池 $\mathcal{T}^{t,-}$ 和混合组子集 $\mathcal{G}_{\text{mix}}^t$。
- **双记忆系统**：失败记忆 $\mathcal{M}_{\text{f}}$ 存储传统反思发现的失败模式；对比记忆 $\mathcal{M}_{\text{c}}$ 存储同组成败对比发现的模式。每个模式由自然语言描述 $p_k$ 和支撑组集合 $u_k$ 构成，支持组越多代表泛化证据越强。
- **传统失败反思与成功反思**：分别利用失败池和成功池，结合对应记忆，调用 $M_{\text{opt}}$ 提出候选技能编辑。
- **组对比反思**：对每个混合组 $G_i^t \in \mathcal{G}_{\text{mix}}^t$，调用 $M_{\text{opt}}$ 比较组内成功与失败轨迹，定位最早的行为分歧点，提取可推广假设，并更新对比记忆。
- **证据导向的技能修订**：合并三路编辑（$\mathcal{P}_{\text{f}}^t \cup \mathcal{P}_{\text{s}}^t \cup \mathcal{P}_{\text{c}}^t$），按对比编辑>失败编辑>成功编辑的优先级排序，考虑支持组数量，选择最多 $L^t$ 条编辑应用为新候选技能 $s_{\text{cand}}^t$，仅在验证分数提升时接受。

## 实验与结果
- **数据集**：SpreadsheetBench、SearchQA、LiveMathematicianBench、ALFWorld、OfficeQA、DocVQA（共6个基准），另有 HotpotQA 和 Omni-MATH 用于跨数据集迁移实验。
- **模型**：Qwen3.8-Flash-Next、DeepSeek-V4-Pro-0813、GLM-5.3（自评优化）、DeepSeek-V4-Flash、Qwen3.6-35B-A3B（强模型引导弱模型）。
- **基线**：No skill、Init skill、Trace2Skill、SkillOpt、SkillOpt-Lite。
- **主要结果**：SkillCome 在全部26个模型-基准对上取得最高分。最强提升为 DeepSeek-V4-Flash 在 LiveMath 上从31.60%→74.29%（+42.69点）；OfficeQA 从5.41%→50.51%（+45.10点）。平均较无技能提升约19.55~27.23点，较第二名 SkillOpt 平均提升5.69分。
- **消融**：移除双记忆后性能在所有对上都下降，但 SkillCome w/o Memory 仍优于 SkillOpt，证明两组分为协同互补。

## 相关工作脉络
- **SkillOpt**：结构化地将轨迹转化为有界技能编辑，通过编辑聚合和验证接受机制工作；本文相比之处的定位差异在于引入组对比和跨步记忆，聚焦更可靠的优化信号构造。
- **SkillOpt-Lite**：将演化过程委托给智能体自动执行；本文保留人工结构化流程但增强了信号质量。
- **Trace2Skill**：并行从执行轨迹中提取局部经验并整合为可迁移程序指导；本文与之区别在于不仅提取局部经验，还通过组对比直接对比同题成败以定位因果差异。
- **OEO / Evoskill**：分别委托演化过程给智能体或用于多智能体系统技能发现；本文方法更强调组级信号和记忆累积。
- **SKILLOS / Coevoskills**：分别探索技能策展和协同验证演化；本文聚焦组对比机制而非协同或策展架构。
- **Prompt优化方法（PEP、AutoPrompt等）**：关注静态提示改进；本文聚焦动态技能演化循环中的信号构造问题。

## 局限性与未来方向
- 混合组比例在24.88%~40%之间，意味着约60%~75%的组无对比价值，资源利用仍有提升空间。
- 从零技能演化时平均仅提升约12.54点，弱于从初始技能演化（约15~27点），说明初始技能对演化起点和过程均有重要影响。
- 未充分讨论长周期演化中记忆膨胀的管理策略和记忆检索效率问题。
- 未来可扩展至更多任务类型，研究动态调整每组轨迹数量 n 的策略，以及将双记忆思想与强化学习结合进行更细粒度的技能优化。

## 研究启发与可借鉴点
- **组内对比信号构造**可将组对比思想迁移到 RLHF/DPO 训练中，利用同问题多响应的成败对比替代纯偏好排序，提供更细粒度的优化信号。
- **双记忆系统的设计**可用于其他需要跨步积累的优化场景（如提示词自动演化、工具使用策略学习），通过"支持集计数"量化证据强度值得借鉴。
- **轨迹去重降低优化器开销**的策略（相同输入的去重轨迹合并后再分析）可作为通用技巧应用于各类基于轨迹的学习方法中。
- **跨模型/跨数据集迁移实验设计**为后续工作的泛化性验证提供了可复用的评测范式。

## 关键术语表
- **Group Rollout（组 rollout）**：在同一技能下为每个问题独立生成多条轨迹并分组，形成可供对比分析的数据结构。
- **Group Contrast Reflection（组对比反思）**：比较同一组内成功与失败轨迹，定位最早行为分歧，提取可推广的技能编辑建议。
- **Dual Memory System（双记忆系统）**：由失败记忆和对比记忆组成的存储系统，分别积累跨步骤的失败模式和组对比中发现的模式。
- **Mixed Group（混合组）**：同时包含成功和失败轨迹的组，是组对比反思的主要输入来源。
- **Meta-option（元选项）**：某些数学题中出现的"另一选项正确但更强结论可证"类选项，SkillCome 能学习针对性答题策略。
- **SkillOpt-Lite**：将技能演化过程委托给智能体自动执行的简化方法，本文作为主要对比基线之一。
- **Codex CLI Harness**：基于代码执行环境的智能体评估框架，用于验证 SkillCome 在真实工作流中的适用性。
- **Optimizer Token Efficiency**：SkillCome 通过组内轨迹去重，优化器 token 消耗较 SkillOpt 减少约50%~64%。

## 可复现要素
- **数据集**：训练和评估数据集详见附录 B.1，数据分割遵循 SkillOpt-Lite 的设定；数据本身为公开基准（SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMathematicianBench、ALFWorld、HotpotQA、OlympiadBench、Omni-MATH）。
- **代码**：论文在 Reproducibility Statement 中声明代码在补充材料中提供。
- **关键超参**：每组轨迹数 n=8；最大响应长度16384 tokens；优化轮次4个 epoch 或10个 batch；验证接受策略（仅当 $V(s_{\text{cand}}) > V(s^t)$ 时接受）。
