---
title: "Unlocking-Token-Rewards-via-Training-Free-Reward-Attribution"
source: https://arxiv.org/pdf/2610.18704v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:36:38"
field: "大语言模型推理强化学习"
keywords: ["Token Reward Attribution", "Process Reward Model", "Training-Free RL", "Chain-of-Thought Reasoning", "Reward Shaping", "GRPO", "Credit Assignment"]
innovations: ["提出训练免费的P2T方法将过程奖励高效分解为token级奖励", "基于一阶泰勒近似将O(N)复杂度降至O(1)实现单次反向传播归因", "在40个model-benchmark组合上稳定超越结果奖励与现有token奖励方法"]
benchmarks: ["AIME24", "MathVista", "MinervaMath", "MATH-500", "AMC23", "OlympiadBench"]
---

# 论文速读：Unlocking-Token-Rewards-via-Training-Free-Reward-Attribution

## 一句话总结
本文提出 P2T（Process-to-Token）方法，通过一阶泰勒近似将现有过程奖励模型（PRM）的粗粒度过程奖励高效分解为 token 级奖励，实现完全无需训练的精细信用分配，显著提升 LLM 在数学推理与多模态任务上的表现。

## 研究问题与动机
1. 现有 PRM 虽提供步骤级奖励，但每个推理步骤仍含数百至数千个 token，奖励分配依然粗放。
2. 已有 token 级奖励方法陷入两难：训练专用 token 奖励模型成本高昂且伪标签噪声大；使用启发式代理指标（如 token 熵）缺乏与真实奖励的语义对齐。
3. 获取大规模人工标注以训练真正细粒度的 token 级奖励模型在现实中不可行，亟需一种低成本、有 principled 依据且鲁棒的 token 级奖励估计方法。

## 核心贡献（创新点）
1. **提出 P2T 算法，实现完全训练免费的 token 级奖励转换**，弥合粗粒度过程奖励与细粒度 token 级监督之间的差距。
2. **设计基于梯度的高效近似求解器**，将 token 影响计算的复杂度从 O(N) 降至 O(1)，仅需单次前向+反向传播即可对所有 token 归因。
3. **实证验证 P2T 在多种模型族、推理范式与模态下均稳定超越结果奖励与现有 token 级奖励方法**，并显著提升训练效率（用 3-4 倍更少的训练步数达到同等性能）。

## 方法详解
1. **Token 归因分数定义**：Token $y_i$ 的归因分数 $\mathcal{T}_i = R(E) - R(E_{i \to \emptyset})$，其中 $E_{i \to \emptyset}$ 表示将第 $i$ 个 token 的 embedding 替换为空标记（null token）embedding 后的序列，$R(\cdot)$ 为可微 PRM 输出的过程奖励。$\mathcal{T}_i > 0$ 表示该 token 对奖励有正向贡献，$\mathcal{T}_i < 0$ 表示负向贡献。

2. **一阶泰勒近似**：为避免 N 次前向传播，采用一阶泰勒展开近似：$\mathcal{T}_i \approx \nabla_{\mathbf{e}_i} R(E)^\top (\mathbf{e}_i - \mathbf{e}_\emptyset)$，其中 $\nabla_{\mathbf{e}_i} R(E)$ 为 reward 对 token embedding 的梯度，所有 token 的梯度可通过单次反向传播同时获得。

3. **Token 级奖励重分配**：对每个推理步骤 $s$，将归因分数取绝对值后归一化作为权重 $w_i = \frac{|\mathcal{T}_i|}{\sum_{j \in \mathcal{T}_s} |\mathcal{T}_j|}$，最终 token 级奖励为 $\mathcal{R}_i^{\text{P2T}} = \mathcal{R}_s + \eta \cdot w_i \cdot \mathcal{R}_s$，其中 $\mathcal{R}_s$ 为步骤级过程奖励作为 baseline，$\eta$ 控制重分配强度（默认 0.5）。baseline 的设计可限制归因误差引入的噪声扰动。

4. **与 GRPO 集成**：将 token 级奖励融入 GRPO 的 advantage 计算：$\widetilde{A}_{n,i} = \hat{A}_n + \alpha \cdot \mathcal{R}_{n,i}^{\text{P2T}}$，其中 $\hat{A}_n$ 为基于 outcome reward 计算的 group-level advantage，$\alpha$ 为平衡权重（默认 1.0）。

## 实验与结果
1. **数据集与模型**：多模态模型（Qwen2.5-VL-7B-Instruct）使用 VerMulti-65K 和 Vision-R1-10K 训练，评估 MathVista、MathVerse、MathVision、MMMU、MMStar；纯文本模型（Qwen2.5-Math-7B、LLaMA3.2-3B-Instruct、DeepSeek-R1-Distill-Qwen-1.5B、Qwen3-1.7B）使用 MATH/DeepScaleR-40K 训练，评估 AIME24/AIME25、AMC23、MATH-500、MinervaMath、OlympiadBench、GaoKao2023en。

2. **PRM 选择**：文本模型使用 ReasonFlux-PRM-7B，多模态模型使用 VisualPRM。

3. **核心结果**：GRPO+P2T 在全部 40 个 model-benchmark 组合上均稳定超越 GRPO。具体地：Qwen2.5-VL-7B-Instruct 在 MathVista 上 +4.9%；Qwen2.5-Math-7B 在 AIME24 上 +11.5%（40.3 vs 28.8）；DeepSeek-R1-Distill-Qwen-1.5B 在 MinervaMath 上 +14.2%（41.4 vs 27.2）。平均提升幅度达 +5.8%（Qwen2.5-Math-7B）。

4. **超越现有 token 级奖励**：在 Eurus-2-7B-SFT 上，GRPO+P2T 平均 Pass@1 为 40.7，超越 GRPO（33.5）、PRIME（36.0）和 SPRO（38.4）。

5. **训练效率**：Qwen2.5-Math-1.5B 实验中，GRPO+P2T 以 3-4 倍更少的训练步数达到 GRPO 的最佳测试精度。

6. **突破收敛瓶颈**：对已通过 outcome RL 训练收敛的 Vision-R1-7B 和 DeepScaleR-1.5B-Preview 继续用 GRPO+P2T 训练，进一步提升了 AIME24 等基准成绩（如 DeepScaleR-1.5B-Preview 在 AIME24 上 +6.8%）。

## 相关工作脉络
1. **Outcome Reward RL（DeepSeek-R1、Gemini、OpenRLHF 等）**：使用最终答案正确性作为二元奖励，reward 稀疏且无法区分同一轨迹中各 token 的贡献。
2. **Process Reward Model（Math-Shepherd、qwen2.5-math-PRM、VisualPRM 等）**：提供步骤级奖励，粒度优于 outcome reward 但仍覆盖整个推理步骤内的所有 token。
3. **PRIME [6]**：在线训练隐式过程奖励模型并从其推导 token 级奖励，依赖奖励模型质量且需额外优化。
4. **SPRO [17]**：基于 policy 与 reference model 的 log-probability 差分配 token 奖励，未直接评估推理步骤的正确性。
5. **OREAL [7] / TVM [65]**：训练专用 token 级奖励模型，需大量标注或额外网络结构，成本高昂。
6. **CAPO [64] / T-SPMO [62]**：通过启发式规则或采样估计分配 token 奖励，缺乏基于 PRM 的可微归因原则。

## 局限性与未来方向
1. P2T 性能依赖所使用 PRM 的质量，若 PRM 在目标域上不可靠则效果受限。
2. 当前方法仅使用归因分数的绝对值进行重分配，未充分利用 sign 信息（正/负贡献方向），作者指出这是未来方向。

## 研究启发与可借鉴点
1. **一阶泰勒近似用于信用分配**：将复杂的多步影响分解转化为梯度与 embedding 差的内积，这一思路可迁移到其他需要细粒度归因的场景（如注意力机制分析、多步决策系统的 credit assignment）。
2. **Baseline 保留设计**：在重分配中保留原始 coarse reward 作为 baseline（$\mathcal{R}_s$），可有效抑制归因噪声导致的训练不稳定，这一设计模式值得在其它 reward shaping 任务中借鉴。
3. **已收敛模型的再优化验证**：对已通过 outcome RL 充分训练的模型继续使用 P2T 进一步提升，证明了方法的补充优化能力，可作为评估新 reward 方法的有力实验设计。
4. **Null Token 选择敏感性分析**：论文系统比较了 pad token、EOS token、zero embedding、vocabulary mean 等多种 null token 选择，发现保持"中性且非人工构造"是关键，这一分析范式可为其他归因方法提供参考。

## 关键术语表
**Process Reward Model (PRM)**：评估推理轨迹中每个步骤质量的过程级奖励模型，提供比 outcome reward 更细粒度的监督信号。
**Token Attribution Score**：衡量单个 token 对整体过程奖励边际贡献的分数，定义为该 token 被替换为空标记后奖励的变化量。
**Null Token**：用作归因参考的非信息性 token，论文选用 PRM 词汇表中的 padding token，因其在推理响应中几乎不出现且携带极少内容含义。
**First-order Taylor Approximation**：一阶泰勒展开近似，将 token 影响计算简化为 reward 梯度与 embedding 差的内积，使复杂度从 O(N) 降至 O(1)。
**Process-to-Token (P2T)**：本文提出的方法，将过程级奖励通过归因权重重分配为 token 级奖励。
**GRPO**：Group Relative Policy Optimization，一种无需 value function 的 RL 算法，通过组内 reward 相对差异估计 advantage。

## 可复现要素
- **数据集**：VerMulti-65K、Vision-R1-10K（多模态）；MATH、DeepScaleR-40K、Eurus-2-RL-Data（纯文本）；数学基准包括 AIME24/AIME25、AMC23、MATH-500、MinervaMath、OlympiadBench、GaoKao2023en；多模态基准包括 MathVista、MathVerse、MathVision、MMMU、MMStar。
- **代码**：已开源于 https://github.com/stonewst/P2T
- **PRM**：ReasonFlux-PRM-7B（文本）、VisualPRM（多模态）
- **关键超参**：$\eta = 0.5$（重分配强度）、$\alpha = 1.0$（token reward 权重）、学习率 $1 \times 10^{-6}$、KL 系数 0.001、batch size 256（文本）/ 128（多模态）
- **训练步数**：文本模型 500-1000 steps，多模态模型 1 epoch
