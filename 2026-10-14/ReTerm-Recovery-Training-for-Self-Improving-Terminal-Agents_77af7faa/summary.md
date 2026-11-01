---
title: "ReTerm-Recovery-Training-for-Self-Improving-Terminal-Agents"
source: https://arxiv.org/pdf/2610.18484v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:45:19"
field: "Agent训练与自我改进"
keywords: ["terminal agents", "self-recovery", "reinforcement learning", "error recovery", "GRPO", "post-training"]
innovations: ["提出ReTerm-10K数据集，从智能体自身失败状态生成高质量恢复轨迹用于SFT冷启动", "提出ReTerm-GRPO在线强化学习算法，从演化策略的失败状态分支恢复尝试并联合优化", "设计恢复任务队列跨越训练步维持失败样本供给，使恢复能力与任务解决能力共演化"]
benchmarks: ["Terminal-Bench 2.0", "SWE-bench Verified", "OpenThoughts-TBLite", "AIME 2025", "SpreadsheetBench Verified"]
---

# 论文速读：ReTerm-Recovery-Training-for-Self-Improving-Terminal-Agents

## 一句话总结
论文提出了ReTerm，一种自改进的终端智能体，通过从自身执行失败中学习恢复能力来进行后训练。该方法构建了ReTerm-10K高质量恢复轨迹数据集支持SFT冷启动，并提出ReTerm-GRPO在线强化学习算法，使恢复能力与任务解决能力同步进化，在Terminal-Bench 2.0上以9B模型规模取得34.8% Pass@1，超越多数32B专用智能体。

## 研究问题与动机
1. **终端智能体缺乏错误恢复能力**：终端智能体在长时间执行中频繁遭遇命令错误、实现缺陷等问题，常误诊根本原因并继续错误方向，最终导致任务失败。
2. **现有方法存在分布不匹配**：TermiGen等通过人工注入错误生成恢复样本，但与智能体自然产生的错误存在分布差异；且仅依赖SFT，未探索强化学习的潜力。
3. **恢复训练与任务执行脱节**：已有工作主要关注从零开始完成任务，未显式教智能体如何从 faulty 状态诊断问题并修复。

## 核心贡献（创新点）
1. **ReTerm-10K数据集**：通过基座模型执行收集失败状态，由更强教师模型从失败状态恢复生成轨迹，经四步过滤（验证器、格式、合规性、效率）保留约10K高质量恢复演示，用于SFT冷启动。
2. **ReTerm-GRPO在线强化学习算法**：从智能体自身执行产生的失败状态分支出多个独立恢复尝试，通过验证器结果提供组相对学习信号，使自恢复能力随策略演化自适应，并通过恢复任务队列跨越训练步骤维持恢复样本供给。
3. **跨基准强泛化性能**：ReTerm-9B在Terminal-Bench 2.0（34.8%）、SWE-bench Verified（58.4%）、OpenThoughts-TBLite（60.6%）上均显著优于基座模型，且在AIME 2025（+20.0）、SpreadsheetBench Verified（+11.0）等跨域任务及不同Agent harness下保持提升。
4. **与已有工作的本质区别**：ReTerm-10K直接基于智能体自然失败状态而非人工注入错误生成恢复数据，消除分布不匹配；ReTerm-GRPO通过在线RL使恢复能力与任务解决能力共演化，而非仅依赖静态SFT数据。

## 方法详解
**阶段一：Recovery数据集构建（ReTerm-10K）**
- **失败状态收集**：使用基座模型（Qwen3.5-9B）在TMax-15K任务上采样多次尝试，在虚拟机容器中执行并暂停快照，保留交互历史和失败状态的快照。
- **恢复轨迹生成**：恢复快照后，由教师模型（DeepSeek-V4-Pro）基于原始任务指令和失败历史生成恢复续写，需提供恢复工作流指导（审计继承状态→重建任务契约→诊断失败→处理继承状态）。
- **四步质量过滤**：①验证器过滤（仅保留通过任务验证器的恢复）；②格式过滤（移除畸形消息/工具调用）；③合规性过滤（排除违反任务约束的动作）；④效率过滤（移除无意义循环动作），后两步使用DeepSeek-V4.1-Flash辅助判定。

**阶段二：SFT冷启动**
$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}\left[\sum_{t \in A(y)} \log \pi_\theta(y_t \mid x, h_f, r, y_{<t})\right]$$
输入为任务指令x、学生失败历史$h_f$、恢复提示r，仅对教师生成的token计算loss。

**阶段三：ReTerm-GRPO在线强化学习**
- **失败状态保留与分支**：从策略生成的G个原始rollout中采样至少一个失败rollout，保留交互历史$h_f$和最终环境状态$s_f$，构成恢复任务$(x, h_f, s_f)$。
- **恢复任务队列**：容量为K的队列存储跨训练步的恢复任务，新任务插入队首优先采样，溢出时移除最老任务，确保恢复样本与当前策略的失败分布匹配。
- **组相对优势计算**：对原始任务和恢复任务的G个rollout分别计算组内归一化优势：
$$\hat{A}_{g,i}^c = \frac{R_{g,i}^c - \text{mean}(R_{g,j}^c)}{\text{std}(R_{g,j}^c) + \delta}$$
- **联合优化目标**：
$$\mathcal{I}_{\text{ReTerm}}(\theta) = \frac{1}{M_O + M_R}\left(\sum_{g=1}^{M_O} \mathcal{I}_g^O(\theta) + \sum_{g=1}^{M_R} \mathcal{I}_g^R(\theta)\right)$$
每步采样8组原始任务+2组恢复任务，各含16 rollouts，全局batch size=160。

## 实验与结果
- **训练设置**：16×NVIDIA A100 (80GB)，Qwen3.5-9B基座，ReTerm-10K划分为6K轨迹用于SFT、4K任务用于RL，SFT学习率$5\times10^{-6}$，GRPO学习率$5\times10^{-7}$，最大生成长度8192 tokens，上下文窗口65k，最多32交互轮次。
- **核心基准**：Terminal-Bench 2.0上ReTerm-9B达34.8% Pass@1（基座20.2%，+14.6分）；SWE-bench Verified上58.4%（基座45.2%，+13.2分）；OpenThoughts-TBLite上60.6%（基座44.2%，+16.4分）。
- **跨域泛化**：AIME 2025达93.3%（+20.0分），SpreadsheetBench Verified达48.8%（+11.0分），Aider Polyglot达20.4%（+3.1分）。
- **跨Harness泛化**：在Terminus-2（+10.1）、mini-SWE-agent（+6.7）、OpenHands（+12.4）上均保持稳定提升。
- **消融结果**：Base→SFT→GRPO→ReTerm-GRPO在Terminal-Bench 2.0上依次达到20.2%→25.8%→28.0%→34.8%。
- **错误类型分析**：ReTerm-GRPO在五种错误类型（Planning/Tool Use/Implementation/Verification/Resource Management）上均优于基座和标准GRPO，但除Tool Use Error外其余类别恢复率≤32%。

## 相关工作脉络
1. **TMax (Ivison et al., 2026)**：通过大规模合成终端任务和RL训练提升执行能力，但未关注错误恢复；本文聚焦恢复能力，两者互补。
2. **ECHO (Shrivastava et al., 2026)**：引入辅助观察预测目标增强环境理解；本文侧重从失败状态学习的恢复策略。
3. **TermiGen (Zhu et al., 2026)**：通过主动注入人工错误生成恢复轨迹并SFT训练，存在分布不匹配且未探索RL；本文直接使用智能体自身失败状态+在线RL。
4. **Self-Refine/Self-Correction (Madaan et al., 2023; Shinn et al., 2023)**：在静态文本生成任务中通过迭代修订提升输出质量；本文将自修正扩展到交互式终端环境，处理环境状态被错误动作改变的特殊挑战。
5. **Swe-Agent (Yang et al., 2024) / OpenHands (Wang et al., 2025)**：开源软件工程Agent框架；本文在SWE-bench Verified上超越Socratic-SWE（同为9B级），展示小模型通过恢复训练可达更强性能。
6. **GRPO (Guo et al., 2025)**：组相对策略优化算法；本文将其扩展至恢复场景，引入失败状态保留、恢复任务队列和双任务联合优化。

## 局限性与未来方向
1. **恢复成功率仍有提升空间**：除工具使用错误外，其余四类错误恢复率≤32%，表明智能体在复杂规划错误、实现错误、验证错误和资源管理错误上的恢复能力仍有限。
2. **依赖教师模型质量**：ReTerm-10K的数据生成依赖DeepSeek-V4-Pro等强教师模型，教师能力直接影响数据集质量。
3. **恢复任务队列容量限制**：队列容量K需权衡，过大可能引入过时失败分布，过小则样本不足。
4. **未来方向**：可扩展至更多错误类型、探索更大规模恢复数据集、结合人类反馈优化恢复策略、研究多轮恢复机制。

## 研究启发与可借鉴点
1. **失败状态复用范式**：从执行中间状态而非从头开始训练恢复能力的思路，可迁移至其他交互式Agent（如GUI Agent、机器人控制）的错误恢复训练。
2. **恢复任务队列设计**：跨训练步存储和优先采样近期失败任务的机制，可有效缓解在线RL中失败样本稀疏问题，适用于任何需要错误恢复的序列决策任务。
3. **四步质量过滤 pipeline**：验证器+格式+合规性+效率的多维过滤策略，尤其是使用LLM辅助判定合规性和效率，可为其他数据集构建提供通用模板。
4. **双任务联合优化**：原始任务与恢复任务共享优化目标但分别计算组相对优势的设计，保证恢复训练不破坏主任务性能，可用于多目标Agent训练。

## 关键术语表
**ReTerm-10K**：从基座模型失败状态收集并由教师模型恢复生成的约10K高质量恢复轨迹数据集，用于SFT冷启动。
**ReTerm-GRPO**：在线强化学习算法，从策略自身生成的失败状态分支多个独立恢复尝试，通过组相对优势联合优化原始任务与恢复任务。
**恢复任务队列**：容量为K的FIFO队列，跨训练步存储恢复任务，新任务优先采样，保持恢复样本与当前策略失败分布匹配。
**组相对优势**：对同组内G个rollout的验证器奖励进行均值-方差归一化，消除组间尺度差异，提供更稳定的学习信号。
**失败状态快照**：在虚拟机中暂停执行并保存的当前环境状态，用于后续从错误点恢复而非重置环境。
**Pass@1**：单次尝试完成任务的成功率，衡量智能体的一次性成功率。
**Terminal-Bench 2.0**：包含真实复杂终端任务的标准化评测基准，评估Agent的长程交互能力。
**SWE-bench Verified**：在GitHub真实仓库上验证的软件工程任务基准，评估Agent修复实际代码问题的能力。

## 可复现要素
- **数据集**：ReTerm-10K已公开（https://github.com/twothsnd/ReTerm），基于TMax-15K任务构建。
- **代码**：已开源（https://github.com/twothsnd/ReTerm）。
- **模型权重**：ReTerm-9B已开源。
- **关键超参**：SFT学习率$5\times10^{-6}$，batch size=16；RL学习率$5\times10^{-7}$，每步8组原始任务+2组恢复任务，每组16 rollouts，全局batch=160；最大生成长度8192 tokens，上下文窗口65k，最多32轮交互；温度1.0，top-p 1.0。
- **硬件**：16×NVIDIA A100 (80GB)。
