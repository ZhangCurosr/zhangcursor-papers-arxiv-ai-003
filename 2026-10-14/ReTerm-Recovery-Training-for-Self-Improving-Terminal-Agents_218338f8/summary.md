---
title: "ReTerm-Recovery-Training-for-Self-Improving-Terminal-Agents"
source: https://arxiv.org/pdf/2610.18484v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:01:07"
field: "终端智能体与自我恢复训练"
keywords: ["terminal agents", "self-recovery", "reinforcement learning", "GRPO", "SFT", "agent training"]
innovations: ["构建ReTerm-10K恢复轨迹数据集，从基座模型自身失败状态生成高质量恢复演示", "提出ReTerm-GRPO在线恢复强化学习算法，联合优化原始任务与恢复任务", "设计恢复任务队列机制，跨训练步保留近期失败任务供恢复训练"]
benchmarks: ["Terminal-Bench 2.0", "SWE-bench Verified", "OpenThoughts-TBLite", "AIME 2024/2025", "SpreadsheetBench Verified"]
---

# 论文速读：ReTerm-Recovery-Training-for-Self-Improving-Terminal-Agents

## 一句话总结
本文提出 ReTerm，一种通过自我恢复训练来改进的终端智能体。该方法构建了专门的恢复轨迹数据集 ReTerm-10K 用于 SFT 冷启动，并设计了 ReTerm-GRPO 在线强化学习算法，使智能体能够从自身执行失败的状态中学习恢复能力，最终在 Terminal-Bench 2.0 上取得 34.8% Pass@1 的显著提升。

## 研究问题与动机
- **核心问题**：终端智能体在长时间执行中频繁遭遇错误（命令报错、实现缺陷等），且难以从错误状态恢复，导致任务最终失败。
- **现有方法不足一**：TermiGen 等通过人工注入错误生成恢复样本，但这些人工构造的错误与智能体自身自然产生的错误存在分布不匹配。
- **现有方法不足二**：TermiGen 仅依赖 SFT，未探索强化学习对恢复能力的进一步挖掘潜力。
- **现有方法不足三**：标准 GRPO 训练虽然能提升整体任务完成能力，但并未显式针对"从失败状态恢复"这一行为进行优化。

## 核心贡献（创新点）
- **构建 ReTerm-10K 恢复轨迹数据集**：设计了三阶段数据管道（失败状态采集→教师模型恢复生成→四步质量过滤），为 SFT 冷启动提供高质量恢复演示；与常规 SFT 数据教智能体从零开始解题不同，本数据集显式教授如何从错误状态继续执行。
- **提出 ReTerm-GRPO 在线恢复强化学习算法**：从智能体当前策略自身生成的失败状态出发，分支出多个独立恢复尝试，并通过验证器奖励提供组内相对学习信号；与标准 GRPO 仅在原始任务上优化不同，ReTerm-GRPO 联合优化原始任务与恢复任务，使恢复能力与任务解决能力协同演化。
- **引入恢复任务队列机制**：维护跨训练步的恢复任务队列，新构建的失败任务插入队首优先采样，超出容量时丢弃最旧任务；这一设计减少了对额外 rollouts 的依赖，同时保证恢复训练聚焦于近期策略产生的失败。
- **在多个基准上验证有效性**：基于 Qwen3.5-9B 的 ReTerm-9B 在 Terminal-Bench 2.0 上达到 34.8% Pass@1（+14.6 分），SWE-bench Verified 达到 58.4%（+13.2 分），且在 AIME 2025、SpreadsheetBench Verified 等域外基准上同样显著提升。

## 方法详解
**ReTerm-10K 数据构建（三阶段）**：
1. **失败状态采集**：用基座模型（Qwen3.5-9B）在 TMax-15K 任务上采样多次执行，保存失败尝试的交互历史与环境快照（VM 暂停→快照→恢复验证）。
2. **恢复轨迹生成**：恢复失败快照，用教师模型（DeepSeek-V4-Pro）基于原始任务指令和学生的失败历史生成恢复续写，附带结构化恢复工作流（审计继承状态→重建任务契约→诊断失败原因→处理继承状态）。
3. **四步质量过滤**：① 验证器过滤（保留通过验证的恢复）；② 格式过滤（去除畸形消息/工具调用）；③ 合规性过滤（排除违反任务约束的行动）；④ 效率过滤（剔除无意义循环）。最终获得约 10K 高质量轨迹。

**SFT 冷启动损失函数**：
$$\mathcal{L}_{\mathrm{SFT}} = -\mathbb{E}_{(x, h_f, r, y) \sim \mathcal{D}_{\mathrm{ReTerm}}} \left[ \sum_{t \in \mathcal{A}(y)} \log \pi_\theta(y_t | x, h_f, r, y_{<t}) \right]$$
其中 $x$ 为任务指令、$h_f$ 为学生失败交互历史、$r$ 为恢复提示、$y$ 为教师恢复轨迹，损失仅计算教师生成 token 位置。

**ReTerm-GRPO 算法设计**：
- **原始任务 rollouts**：从初始状态 $s_0$ 采样 $G$ 个独立 rollout，记录最终环境状态 $s_i^O$ 和验证器奖励 $R_i^O$。
- **恢复任务构造**：若一组中存在失败 rollout，保留其失败状态 $(x, h_f, s_f)$ 作为恢复任务，存入恢复任务队列。
- **恢复任务 rollouts**：从队列采样恢复任务，附加恢复提示 $r$，生成 $G$ 个独立恢复 rollout，记录奖励 $R_i^R$。
- **组内相对优势归一化**：对原始任务和恢复任务分别计算 $\hat{A}_{g,i}^c = \frac{R_{g,i}^c - \mathrm{mean}(R_{g,j}^c)}{\mathrm{std}(R_{g,j}^c) + \delta}$。
- **联合优化目标**：
$$\mathcal{I}_{\mathrm{ReTerm}}(\theta) = \frac{1}{M_O + M_R} \left( \sum_{g=1}^{M_O} \mathcal{I}_g^O(\theta) + \sum_{g=1}^{M_R} \mathcal{I}_g^R(\theta) \right)$$
每步更新采样 8 个原始任务组和 2 个恢复任务组，每组 16 rollouts，全局 batch size 为 160。

## 实验与结果
**数据集与基线**：
- 基座模型：Qwen3.5-9B
- 训练数据：TMax-15K（从中筛选），ReTerm-10K（6K 用于 SFT，4K 任务实例用于 RL）
- 基线模型：TMax-9B、Socratic-SWE、TermiGen-32B、OpenThinkerAgent-32B 等

**主要结果（Terminal-Bench 2.0 / SWE-bench Verified / OpenThoughts-TBLite）**：
- ReTerm-9B：34.8% / 58.4% / 60.6%
- 相对于 Qwen3.5-9B 基座的提升：+14.6 / +13.2 / +16.4 分
- 超过同规模 TMax-9B（Terminal-Bench: 27.2%）及 TermiGen-32B（Terminal-Bench: 19.3%）
- 超过部分 32B 专用模型（如 OpenThinkerAgent-32B 在 SWE-bench Verified 上为 54.0%）

**消融实验（Terminal-Bench 2.0）**：
- Base: 20.2% → SFT: 25.8% → GRPO: 28.0% → ReTerm-GRPO: 34.8%
- ReTerm-GRPO 比标准 GRPO 额外提升 6.8 分

**域外泛化**：
- AIME 2025: +20.0 分（73.3% → 93.3%）
- SpreadsheetBench Verified: +11.0 分（37.8% → 48.8%）
- Aider Polyglot: +3.1 分；AIME 2024: +10.0 分

**跨 Agent Harness 泛化**（Terminal-Bench 2.0）：
- 四种 harness 下均保持稳定提升：+14.6（ReAct）/ +10.1（Terminus-2）/ +6.7（mini-SWE-agent）/ +12.4（OpenHands）

**错误恢复分析**（五类错误）：
- ReTerm-GRPO 在所有五类错误（Planning / Tool Use / Implementation / Verification / Resource Management）上的恢复率均高于 Base 和标准 GRPO
- 但除 Tool Use Error 外，其余四类错误恢复率均 ≤32%，说明恢复能力仍有较大提升空间

## 相关工作脉络
- **TermiGen (Zhu et al., 2026)**：通过主动注入人工错误生成恢复轨迹并做 SFT；本文差异在于从智能体自身执行失败中生成恢复数据，并通过 RL 持续演化。
- **TMax (Ivison et al., 2026)**：大规模合成终端任务 + 过滤不稳定 token 更新的 RL 算法；本文补充其未涉及的恢复能力训练。
- **ECHO (Shrivastava et al., 2026)**：引入辅助观测预测目标增强终端环境理解；本文关注从失败恢复而非环境建模。
- **Self-Correction via RL (Kumar et al., 2025; Zhang et al., 2026)**：在静态推理任务上训练自我修正；本文将此思路扩展到交互式终端环境中，需处理错误行动对环境状态的污染。
- **GRPO (Guo et al., 2025)**：标准组相对优势强化学习；本文扩展其在终端 agent 恢复场景的应用，通过恢复任务队列实现持续学习。
- **SWE-agent (Yang et al., 2024) / OpenHands (Wang et al., 2025)**：通用软件工程师智能体框架；本文聚焦终端交互中的恢复能力，与这些系统可形成互补。

## 局限性与未来方向
- **恢复成功率仍有上限**：除 Tool Use Error 外，其他四类错误的恢复率均 ≤32%，说明智能体仍难以应对大多数复杂错误场景。
- **数据集规模有限**：仅构建了 10K 高质量恢复轨迹，对于复杂任务领域的覆盖可能不足。
- **依赖更强教师模型**：数据生成阶段使用 DeepSeek-V4-Pro，增加了数据构建成本；如何减少对教师模型的依赖值得研究。
- **评估基准局限**：主要实验集中在 Terminal-Bench 2.0 和 SWE-bench Verified，更多长时交互场景（如真实开发工作流）尚未充分验证。
- **未来方向**：探索更高效的恢复数据合成方法（如蒸馏或自动化错误生成）、扩展多轮恢复能力、结合世界模型预测来提前避免错误。

## 研究启发与可借鉴点
- **失败状态保真采集**：通过 VM 快照-恢复机制精确捕获失败环境状态，而非简单截断轨迹，这一设计可迁移到任何需要"从中间状态继续"的训练场景。
- **恢复任务队列机制**：将历史失败任务纳入跨步采样，既保证训练数据的时效性又减少额外 rollout 开销，可作为在线 RL agent 训练的通用组件。
- **结构化恢复工作流提示**：在数据生成阶段引导教师模型按"审计→诊断→规划→修复→验证"四步操作，这种结构化恢复范式可作为 prompt 设计模板复用。
- **SFT + RL 双阶段恢复训练**：先用高质量恢复演示做冷启动 SFT，再用在线 RL 让恢复能力随策略演化，这种两阶段设计平衡了训练稳定性与适应性。
- **组内相对优势在恢复场景的适配**：将 GRPO 的组相对归一化同时应用于原始任务和恢复任务，并联合优化，可推广到其他需要多任务协同训练的 agent 场景。

## 关键术语表
- **ReTerm**：本文提出的自我改进终端智能体训练框架，包含数据构建管道和在线恢复 RL 算法。
- **ReTerm-10K**：基于 TMax-15K 构建的高质量恢复轨迹数据集，共约 1 万条经过四步过滤的成功恢复示例。
- **ReTerm-GRPO**：面向终端智能体恢复能力的在线强化学习算法，从策略自身失败状态分支多个恢复尝试并进行组内相对优化。
- **恢复任务队列**：跨训练步存储恢复任务的 FIFO 队列，新任务插入队首优先采样，超容时丢弃最旧任务。
- **组内相对优势**：同一训练组内各 rollout 验证器奖励的归一化差值，用于 GRPO 风格的优势估计。
- **Pass@1**：单次尝试即完成任务的成功率，本文主要评测指标。
- **Terminal-Bench 2.0**：包含多种复杂终端交互任务的评测基准，用于评估智能体的长程执行与恢复能力。
- **SWE-bench Verified**：基于 GitHub 真实仓库问题的软件工程评测基准，通过人类验证的测试套件评估修复正确性。

## 可复现要素
- **数据集**：ReTerm-10K 已开源（https://github.com/twothsnd/ReTerm）；训练任务来源为 TMax-15K。
- **代码**：已开源（https://github.com/twothsnd/ReTerm）。
- **模型权重**：ReTerm-9B 已开源。
- **关键超参**：SFT 学习率 5×10⁻⁶，batch size 16；RL 学习率 5×10⁻⁷，每步 8 个原始任务组 + 2 个恢复任务组，每组 16 rollouts，全局 batch size 160；最大生成长度 8192 tokens，上下文窗口 65k，最多 32 轮交互；推理时 temperature=1.0，top-p=1.0，最大生成长度 16k，256k token 预算。
- **硬件**：16 块 NVIDIA A100（80GB）。
- **基座模型**：Qwen3.5-9B；教师模型：DeepSeek-V4-Pro；过滤评估模型：DeepSeek-V4.1-Flash。
