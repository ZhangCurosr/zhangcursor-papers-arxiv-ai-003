---
title: "SynCo-Data-Synthesis-Co-Training-for-Self-Evolving-LLMs-via"
source: https://arxiv.org/pdf/2610.11345v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:28:26"
field: "大语言模型强化学习"
keywords: ["多智能体强化学习", "自我进化", "数据合成", "课程学习", "GRPO", "数学推理"]
innovations: ["提出SynCo将数据合成与推理训练统一为多智能体RL联合优化", "引入基于Rollout结果的Teachability可教性门控奖励", "构建了智能体能力与训练数据分布的闭环协同进化机制"]
benchmarks: ["GSM8K", "MATH-500", "AIME24", "AIME25", "Minerva"]
---

# 论文速读：SynCo-Data-Synthesis-Co-Training-for-Self-Evolving-LLMs-via

## 一句话总结
本文提出了 SYNCO，一种基于多智能体强化学习的自我进化大模型框架，通过联合优化 Synthesizer（生成训练任务）与 Reasoner（解决任务），实现智能体能力与其训练数据的动态协同演化，显著提升了数学推理能力。

## 研究问题与动机
1. **数据分布与能力脱节**：现有方法依赖静态或离线更新的数据，导致随着智能体能力提升，已生成的数据不再具有信息量（梯度消失），无法支持持续的自我改进。
2. **合成与优化分离**：现有自适应数据生成通常将生成策略作为辅助流程，未能将其作为可学习组件与推理策略进行真正的联合优化（Joint Optimization）。
3. **反馈闭环缺失**：缺乏将 Reasoner 的实际交互结果实时反馈给 Synthesizer 以动态调整生成难度的机制。

## 核心贡献（创新点）
1. **提出多智能体强化学习协同训练框架**：将数据合成与任务学习统一为耦合的多智能体 RL 过程，而非串行的两个阶段。
2. **构建闭环自我进化机制**：通过 Synthesizer 感知 Reasoner 能力状态生成针对性任务，并利用 Rollout 结果反向监督 Synthesizer，形成持续迭代的进化循环。
3. **引入“可教性”（Teachability）奖励信号**：设计基于 Reasoner 成功率的门控奖励函数，确保生成任务能有效产生组内相对优势（Group-Relative Advantage）。

## 方法详解
1. **能力条件化数据合成（Capability-Conditioned Synthesis）**：
   - **上下文构建**：Synthesizer 维护交互记忆 $\mathcal{M}_t$（包括成功/失败记录、拒绝原因），并据此生成包含题目、元数据（目标能力、难度）和参考答案的结构化任务卡片。
   - **验证过滤**：通过规则检查验证答案一致性，排除重复或格式错误的无效任务。

2. **结果驱动的多智能体协同训练（Interaction-Grounded Co-Training）**：
   - **Reasoner 优化**：针对每个生成任务进行 $K=4$ 次独立 Rollout，通过检查器判定答案正确性，使用 GRPO 算法根据组内归一化优势更新策略。
   - **Synthesizer 优化**：奖励函数设计为 $r_{S,t} = G_q(r^{quality}) \cdot G_r(r^{reliability}) \cdot T(\mu_t)$，其中 $T(\mu_t)$ 是**可教性函数**，在 Reasoner 正确率接近 50% 时达到峰值，抑制过于简单或过难的样本。

3. **闭环演化（Closed-Loop Evolution）**：
   - 每步结束后更新内存 $\mathcal{M}_{t+1}$ 与上下文 $c_{t+1}$，使 Synthesizer 基于最新的能力边界生成下一阶段的任务。

## 实验与结果
- **基准**：Qwen3-8B 作为骨干模型，测试 GSM8K, MATH-500, AIME24, Minerva 等8个数学推理基准。
- **最强结果**：SYNCO 平均准确率 **78.72%**，较 Vanilla (72.36%) 提升 **+6.36 pp**，较 Frozen Synth 基线提升 +2.87 pp。
- **增益归因**：增量主要来源于对初始模型未能解决的困难题目（Learn 子集），证明其具备扩展新能力而非仅巩固已有知识。
- **效率**：与冻结的 Synthesizer 相比，同步优化带来的边际时间开销几乎为零（<1%）。

## 相关工作脉络
1. **Self-Instruct/WizardLM**：基于种子示例的指令合成，但通常用于静态微调，非持续在线演化。
2. **R-Zero/Socratic-Zero**：建立了生成者与求解者的交替迭代，但本文实现了在同一 Batch 内的联合更新，耦合更紧密。
3. **SwS/CoEvolve**：关注弱点发现或不确定性引导，本文通过 RL 策略直接优化生成质量分布。
4. **MetaMathQA/OpenMathInstruct**：大规模离线静态数据集，无法响应推理器动态能力的变化。
5. **GRPO (Group Relative Policy Optimization)**：本文利用组内归一化优势作为 Reasoner 和 Synthesizer 的学习信号基础。

## 局限性与未来方向
- **领域局限**：目前实验主要集中在数学推理，尚未验证在通用 agent 工具调用或长文本生成中的效果。
- **可教性信号的退化**：随着推理器能力提升，大部分生成任务可能变得过于简单（全对），导致有效的组内信号（Gradient-bearing）逐渐减少。
- **超参敏感性**：奖励函数中的可教性阈值及 Q = 4 的设置可能需要针对不同的任务类型进行调整。

## 研究启发与可借鉴点
1. **任务难度动态化**：对于持续训练的系统，必须建立“生成者-学习者”的在线交互机制，单纯增加数据量或静态筛选不如自适应课程重要。
2. **可教性/梯度富集分析**：借鉴其对合成数据的诊断分析（统计不同信号与组内梯度的相关性），可以量化生成数据的实际训练价值。
3. **门控奖励设计**：用软门控函数（Soft Gating）结合静态质量信号与动态交互结果，避免探索到无效或过于简单的区域。

## 关键术语表
- **Synthesizer / Reasoner**：本文定义的两个独立参数化的智能体，前者负责生成题目，后者负责解题。
- **Teachability (可教性)**：衡量一个生成任务对当前 Reasoner 的多样本 Rollout 结果产生梯度信号的能力。
- **Gradient-bearing tasks**：在 Reasoner 的 K 个 Rollout 中产生混合结果（部分对部分错）的任务，能提供最有效的组内对比学习信号。
- **Frozen Synthesizer**：仅动态生成任务但不更新生成策略的基线模型，用于隔离 Synthesizer 参数更新的收益。
- **Group-Relative Policy Optimization (GRPO)**：通过组内标准化优势来评估轨迹价值的强化学习算法。

## 可复现要素
- **代码/权重**：已开源 (GitHub: https://github.com/weiyang930/SynCo.git, HuggingFace 数据集已发布)。
- **骨干模型**：Qwen3-8B / Qwen3-4B。
- **训练框架**：UnityMAS-O (基于 vLLM)。
- **关键超参**：K=4 rollouts, actor LR=1e-6, 温度参数 Synthesizer=1.0 / Reasoner=0.9。
