---
title: "Unlocking-Token-Rewards-via-Training-Free-Reward-Attribution"
source: https://arxiv.org/pdf/2610.18704v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:36:27"
field: "大模型推理强化学习"
keywords: ["Token-level Reward", "Process Reward Model", "Reinforcement Fine-Tuning", "Training-free Attribution", "GRPO", "Reasoning LLM"]
innovations: ["训练无关的一阶梯度近似将PRM过程级奖励分配为token级奖励，复杂度从O(N)降至O(1)", "提出magnitude-based归一化再分配策略，在GRPO中无缝集成token reward", "无需额外训练奖励模型即可显著提升推理模型准确率与训练效率"]
benchmarks: ["AIME24", "AIME25", "AMC23", "MATH-500", "MinervaMath", "MathVista", "OlympiadBench", "GaoKao2023en"]
---

# 论文速读：Unlocking-Token-Rewards-via-Training-Free-Reward-Attribution

## 一句话总结
本文提出 **P2T（Process-to-Token）** 方法，通过**训练无关**的梯度近似，将已有的过程奖励模型（PRM）的粗粒度步骤级奖励高效分配为细粒度 token 级奖励，从而在不额外训练任何奖励模型的前提下，显著提升大模型推理能力的 RL 训练效率与最终性能。

## 研究问题与动机
- **核心问题**：现有 RL 训练中的监督信号（结局奖励 Outcome Reward 或过程奖励 Process Reward）粒度过粗，无法实现精确的 token 级信用分配（credit assignment）。
- **现有 token 级方法的局限**：
  1. 训练专用 token 级奖励模型成本高，且依赖噪声较大的伪标签，可靠性存疑；
  2. 使用启发式代理指标（如 token entropy）计算成本低，但缺乏语义对齐，无法反映 token 对最终奖励的真实贡献；
  3. 直接获取大规模人工标注来训练真正细粒度的 token 级奖励模型在经济上不可行。
- **关键开放问题**：如何设计一种既成本低、又 principled、且鲁棒的 token 级奖励估计方法？

## 核心贡献（创新点）
1. **训练无关的 P2T 算法**：提出一种新算法，将过程级奖励完全无训练地转换为 token 级奖励，有效弥合了粗粒度与细粒度监督之间的差距。
2. **高效一阶梯度近似**：通过一阶泰勒展开将 token 影响力估计复杂度从 O(N)（需 N 次 PRM 前向）降至 O(1)（只需一次前向+一次反向），使方法可扩展至大规模 RL 训练。
3. **广泛的实证提升**：在多个数学推理基准上，P2T  consistently 优于结局奖励 RL、PRIME、SPRO 等基线，并在 DeepSeek-R1-Distill-Qwen-1.5B 上取得最高 +14.2% 提升（MinervaMath），同时使训练效率提升 3–4 倍。

## 方法详解
### 1. Token Attribution Score 定义
Token $y_i$ 的影响力定义为：将其嵌入 $\mathbf{e}_i$ 替换为空 token 嵌入 $\mathbf{e}_{\emptyset}$ 后，整体奖励的变化量：
$$\mathcal{T}_i = R(E) - R(E_{i \to \emptyset})$$
- $\mathcal{T}_i > 0$：该 token 对奖励有正向贡献；
- $\mathcal{T}_i < 0$：该 token 拉低了奖励；
- $\mathcal{T}_i \approx 0$：影响力可忽略。

空 token 选择 PRM 词表中的 padding token（如 `</s>`），因其在推理过程中几乎不出现，语义上最"中性"。

### 2. 高效梯度近似（核心）
利用一阶泰勒展开近似：
$$\mathcal{T}_i \approx \nabla_{\mathbf{e}_i} R(E)^\top (\mathbf{e}_i - \mathbf{e}_{\emptyset})$$
- 所有 token 的梯度在一次反向传播中同时获得，复杂度从 O(N) 降至 O(1)。

### 3. 过程到 Token 的奖励分配
对第 s 步的 token 集合 $\mathcal{T}_s$，归一化影响力后重新分配过程奖励 $\mathcal{R}_s$：
$$\mathcal{R}_i^{\text{P2T}} = \mathcal{R}_s + \eta \cdot w_i \cdot \mathcal{R}_s, \quad w_i = \frac{|\mathcal{T}_i|}{\sum_{j \in \mathcal{T}_s} |\mathcal{T}_j|}$$
- $\mathcal{R}_s$ 作为 baseline 保证稳定性；
- $\eta$ 控制 redistribution 强度，默认 $\eta=0.5$；
- 实验表明**基于 magnitude 的分配优于带符号版本**（Eq.4 的 signed 变体）。

### 4. 与 GRPO 的集成
将 token 级奖励融入 GRPO advantage 计算：
$$\widetilde{A}_{n,i} = \hat{A}_n + \alpha \cdot \mathcal{R}_{n,i}^{\text{P2T}}$$
- $\alpha=1.0$ 为默认值，PRM 不可靠时可减小或 schedule 递增。

## 实验与结果
### 数据集与模型
- **多模态**：Qwen2.5-VL-7B-Instruct，训练数据 VerMulti-65K + Vision-R1-10K，评测 MathVista、MathVerse、MathVision、MMMU、MMStar。
- **纯文本**：Qwen2.5-Math-7B、LLaMA3.2-3B-Instruct、DeepSeek-R1-Distill-Qwen-1.5B、Qwen3-1.7B，训练数据 MATH / DeepScaleR-40K / Eurus-2-RL-Data，评测 AIME24/25、AMC23、MATH-500、MinervaMath、OlympiadBench、GaoKao2023en。
- **PRM**：多模态用 VisualPRM，文本用 ReasonFlux-PRM-7B。

### 主要结果
| 模型 | 基准 | GRPO | GRPO+P2T | 提升 |
|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | MathVista | — | — | **+4.9%** |
| Qwen2.5-Math-7B | AIME24 | 28.8 | 40.3 | **+11.5%** |
| DeepSeek-R1-Distill-Qwen-1.5B | MinervaMath | 27.2 | 41.4 | **+14.2%** |
| Eurus-2-7B-SFT (vs PRIME/SPRO) | Avg (6项) | 33.5 | 40.7 | +7.2% / +4.7% / +2.3% |

- **收敛速度**：Qwen2.5-Math-1.5B 上，P2T 以 **3–4× 更少训练步数**达到 GRPO 的最优准确率。
- **突破上限**：对已用 outcome-based RL 训练至收敛的 Vision-R1-7B 和 DeepScaleR-1.5B-Preview 继续用 GRPO+P2T 微调，分别在 AIME24 和 MinervaMath 上提升 +6.8% 和 +12.1%。
- **40 个 model-benchmark 组合无一例外全面优于 GRPO**。

### Ablation 关键发现
- **η 鲁棒**：η ∈ [0.25, 1.0] 精度波动仅 ±0.6%，无需精细调参。
- **Magnitude 优于 Signed**：53.8 vs 52.6（Table 3c）。
- **PRM 质量影响显著**：ReasonFlux-PRM-7B（53.8）> Qwen2.5-Math-PRM-7B（52.4）> Skywork-PRM（50.0）；视觉 PRM 同理。
- **空 token 选择**：Padding token（53.8）> Vocab Mean（52.5）> Zero Embedding（51.4）> EOS Token（50.3）。

## 相关工作脉络
1. **DeepSeek-R1 / GRPO（outcome reward）**：仅依赖最终答案正确性，reward 稀疏，是本工作的基础对比基线；P2T 在其之上引入细粒度 token 级信号，无需修改算法框架。
2. **Process Reward Model（PRM）系列**：如 Math-Shepherd、qwen2.5-math-PRM、VisualPRM、ReasonFlux-PRM 等提供步骤级奖励，但仍处于 step 粒度；P2T 直接复用这些现成 PRM，无需额外训练。
3. **PRIME（implicit process reward）**：在线联合训练隐式奖励模型并推导 token 级 reward；P2T 的优势在于**零训练成本**，直接利用 off-the-shelf PRM。
4. **SPRO（self-guided process reward optimization）**：从策略模型与参考模型的 log-prob ratio 推导 token reward，反映的是偏离程度而非推理步骤正确性；P2T 的 reward 源自明确训练用于判断步骤正确性的 PRM，语义更直接。
5. **CAPO / TVM / OREAL**：均为 token 级奖励方法，或依赖预训练 LLM 识别错误步骤（CAPO），或需训练 token-level 奖励头（TVM、OREAL）；P2T 完全训练无关。
6. **Shapley-based credit assignment（SCAR 等）**：理论严谨但计算开销极大；P2T 以 one-pass 梯度近似实现了实用级别的细粒度分配。

## 局限性与未来方向
- **依赖 PRM 质量**：若所用 PRM 在目标领域/格式上不可靠，P2T 效果受限；不同 PRM 间存在最大 3.8% 的性能差距。
- **未充分利用符号信息**：当前采用 magnitude-based 分配，signed variant（Eq.4）实验表现更差，正负项不平衡问题待解决。
- **未来方向**：
  1. 探索更有效地利用 token attribution score 的符号信息；
  2. 将方法扩展至非推理类任务或更广泛的多模态场景。

## 研究启发与可借鉴点
1. **训练无关的细粒度奖励提取范式**：P2T 的"复用现成粗粒度奖励模型 + 梯度近似分配"思路可迁移到其他需要 token/子序列级监督信号的场景（如代码生成、多轮对话），避免高昂的标注和训练成本。
2. **一阶梯度近似在 reward attribution 中的应用**：Taylor 近似将 O(N) 降为 O(1) 的技巧具有通用性，可推广至其他基于 influence function 的模型可解释性或训练分析任务。
3. **奖励 baseline 设计**：P2T 将过程奖励 $\mathcal{R}_s$ 作为 baseline 再叠加归一化 influence 的设计，有效平衡了细粒度信号与训练稳定性，这一"baseline + delta"结构值得在其他 reward shaping 方法中借鉴。
4. **可与本团队方向结合**：若团队涉及推理模型 RLHF/RFT，可直接集成 P2T 模块（仅需调用 PRM 做一次 forward-backward）；同时可尝试将 signed attribution 的改进版应用于当前 Pipeline 以提升难样本表现。

## 关键术语表
- **P2T（Process-to-Token）**：本文提出的训练无关方法，将过程级奖励通过梯度近似分配为 token 级奖励。
- **PRM（Process Reward Model）**：对推理轨迹中每个步骤输出打分的过程奖励模型，是 P2T 的输入来源。
- **Token Attribution Score**：衡量单个 token 对整体奖励边际影响的分数，定义为零替换前后的奖励差值。
- **Outcome Reward**：仅基于最终答案正确性给出的二元奖励，粒度最粗。
- **GRPO（Group Relative Policy Optimization）**：无价值函数的 RL 算法，通过组内奖励均值/标准差估计 advantage，本文作为基线优化器。
- **Null Token / Padding Token**：用于替换目标 token 的空占位 token，作为 attribution 计算的对照组嵌入。
- **Credit Assignment**：将轨迹级/步骤级奖励合理分配给具体 token 的任务，是 P2T 要解决的核心问题。
- **Taylor Approximation**：一阶泰勒展开，本文用于高效近似 token 影响力，避免 N 次前向传播。

## 可复现要素
- **数据集**：VerMulti-65K、Vision-R1-10K、MATH、DeepScaleR-40K、Eurus-2-RL-Data；评测集 MathVista、AIME24/25、AMC23、MATH-500、MinervaMath、OlympiadBench、GaoKao2023en 等（均为公开数据集）。
- **代码**：已开源，GitHub: https://github.com/stonewst/P2T
- **权重**：PRM（VisualPRM、ReasonFlux-PRM-7B、Qwen2.5-Math-PRM-7B、Skywork-PRM 等）均基于已有开源模型；主模型为 Qwen/LLaMA 系列开源权重。
- **关键超参**：η=0.5（ redistribution 强度）、α=1.0（token reward 权重）、KL coef=0.001、lr=1e-6、weight decay=0.01、temperature=1.0（采样）、top-p=1.0、global batch size=128（多模态）/256（文本）、采样 8 条 response/题。
