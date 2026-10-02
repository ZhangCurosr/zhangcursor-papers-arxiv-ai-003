---
title: "The-Hidden-Ratio-in-Adam-Stable-Structure-Compression-and-Si"
source: https://arxiv.org/pdf/2609.35392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:14:24"
field: "优化器理论与高效训练"
keywords: ["Adam optimizer", "tied-beta", "model compression", "low-precision training", "sign-based methods", "optimization dynamics"]
innovations: ["发现Adam中变换比率y_t的稳定重尾分布", "推导y_t的标量递归并实现4-bit固定码本压缩", "揭示Adam与Signum的联系并给出学习率迁移规则"]
benchmarks: ["FineWeb pretraining", "UltraChat SFT", "UltraFeedback RLHF"]
---

# 论文速读：The-Hidden-Ratio-in-Adam-Stable-Structure-Compression, and-Sign-Dynamics

## 一句话总结
本文在 tied-β  regime（β₁=β₂=β）下发现 Adam 优化器中存在一个变换比率 y_t，其具有稳定且重尾的分布特性；基于该稳定性，可将 y_t 用固定 4-bit 码本压缩表示而几乎不损失性能，并揭示了 Adam 与 Signum 之间的联系。

## 研究问题与动机
- Adam 作为现代深度神经网络的默认优化器，其自适应行为因一阶矩 EMA m_t 和二阶矩 EMA v_t 的复杂交互而难以解释。
- 近期大规模训练（如 OLMo、SmolLM3）越来越多地采用相近的衰减率（如 (0.9, 0.95)）， motivted 研究 tied-β regime（β₁=β₂）下的 Adam 结构简化。
- 现有方法缺乏对 Adam 自适应行为的清晰不变量刻画，原始矩量 m_t 和 v_t 的幅度随梯度尺度变化，难以直接用于低精度表示。
- 需要找到一种更稳定的内在状态量，既能揭示 Adam 的理论结构，又能支持高效的低精度实现。

## 核心贡献（创新点）
1. **发现变换比率 y_t 的稳定分布**：在 tied-β  regime 下，通过提取 β 相关尺度后定义的 y_t = (1-β)/β · (v_t - m_t²)/m_t² 在多种任务、模型规模和学习阶段下表现出集中的主体与持久的重右尾分布。与已有工作仅从经验调参角度分析 Adam 不同，本文给出了 y_t 的统计结构分析并证明其与 shifted inverse-square 参考分布的二阶匹配。

2. **推导 y_t 的标量递归**：提出 y_t 的一维递归公式 y_t = βx_t²y_{t-1} + (1-x_t)²（其中 x_t = m_{t-1}/m_t），实现了将 Adam 从 (m_t, v_t) 参数化重参数化为 (m_t, y_t)，使得二阶状态变为可压缩的单标量。这一递归形式是本工作的核心推导，区别于仅修改原始矩状态的现有量化方法。

3. **固定 4-bit 码本的压缩实现**：首次证明对变换后的 y_t 使用固定 4-bit 码本（16 级）进行量化，无需辅助块级或张量级缩放因子，即可在预训练、SFT、RLHF 全流程中保持与全精度 Adam 相当的性能。相比现有低精度优化器方法通常需要学习量化器或额外归一化，本方法更为简洁。

4. **揭示 Adam 与 Signum 的联系并给出学习率迁移规则**：将 Adam 解读为 sign-momentum 方法受 y_t 衰减调制，当用常数 y_const ≈ 3 替代 y_t 时退化为 Signum 类方法，并给出基于期望衰减匹配的学习率转换公式。这为 Signum 等 sign-based 方法的超参调优提供了 Adam-native 的标定方式。

## 方法详解
- **变换比率定义**：在 tied-β regime（β₁=β₂=β）下，定义 r_t = (v_t - m_t²)/m_t² ≥ 0，进一步定义 y_t = (1-β)/β · r_t，使得 Adam 更新方向写为 sign(m_t) / √(1 + β/(1-β) · y_t)。
- **递归推导**：由 m_t = βm_{t-1} + (1-β)g_t 解出 g_t = m_t/(1-β) · (1 - βx_t)，其中 x_t = m_{t-1}/m_t。代入 v_t 的递归并结合 v_t = m_t²(1 + β/(1-β)·y_t)，得到 y_t 的标量递归：y_t = βx_t²y_{t-1} + (1-x_t)²。
- **低精度量化策略**：对 y_t 施加 cut64 截断（min(y_t, 64)）后，使用固定 FP4-like 非均匀码本 C = {0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64} 进行最近邻投影：ỹ_t = argmin_{q∈C}|y_t - q|，然后用 ỹ_t 重建自适应分母。
- **Signum-like 极限**：将 y_t 替换为常数 y_const，得到 Δθ_t ≈ -η_adam · (1 + β/(1-β)·y_const)^{-1/2} · sign(m_t)，其中有效学习率 η_sign = η_adam · (1 + β/(1-β)·y_const)^{-1/2}。通过匹配期望衰减 E[(1 + β/(1-β)·y_t)^{-1/2}] = (1 + β/(1-β)·y_const)^{-1/2} 确定 y_const，理论值约 3.1~3.4。

## 实验与结果
- **数据集**：FineWeb（预训练）、UltraChat（SFT）、UltraFeedback（RLHF/ReMax）。
- **模型规模**：Llama-style 20M、Pythia-style 160M、Pythia-1B、Llama-3.2-1B。
- **基线**：全精度 tied-β Adam、Signum。
- **主要结果**：
  - 预训练：TR-Adam (FP32-m_t, 4-bit y_t) 与 Adam 的验证损失差异 Δ 在 -0.016 ~ +0.016 之间；在 Pythia-1B β=0.92 时优于 Adam（Δ = -0.019）。
  - SFT：TR-Adam 在所有配置下均优于或持平 Adam，最佳 Δ = -0.011（Pythia-1B, β=0.95）。
  - RLHF（ReMax）：TR-Adam 显著优于 Adam，β=0.90 时奖励提升 +0.036。
  - 学习率敏感性：TR-Adam 在 β ∈ {0.90, 0.92, 0.95, 0.97} 的学习率扫描中与 Adam 的最优学习率区间一致。
- **最强结果**：RLHF 阶段 TR-Adam (FP32-m_t, 4-bit y_t) at β=0.90 达到 0.825 vs Adam 0.788，提升 +0.036；SFT 阶段最佳 Δ = -0.011。
- **Signum 对比**：常数 y_const ∈ {3, 5} 的代理在保留 Adam 学习率尺度方面优于纯 Signum，后者随 β 增大需显著降低学习率。

## 相关工作脉络
- **Adam（Kingma & Ba, 2015）**：本文的分析起点，关注其 m_t 和 v_t 的 EMA 交互机制。本文通过 tied-β 重构揭示其内在稳定结构。
- **Signum（SignSGD with momentum, Bernstein et al., 2018; Sun et al., 2023）**：保持一阶矩 EMA 但仅用 sign 方向。本文证明 Signum 是 Adam 的常数 y_t 极限情形，并给出学习率迁移规则。
- **Adam 结构分析（Orvieto & Gower, 2025, NeurIPS）**：将 Adam 分解为 sign-momentum 与噪声-信号比的乘积。本文在此基础上进一步发现 y_t 的稳定性并实现压缩。
- **Tied-β 梯度尺度不变性（Fernández-Hernández et al., 2026）**：证明 β₁=β₂ 时 Adam 具有一阶梯度尺度不变性。本文将其作为理论前提并导出更精细的比率结构。
- **Mini-batch 噪声与 β 选择（Cattaneo & Shigida, 2026）**：分析 mini-batch 噪声对 Adam 隐式偏差的影响，支持较大 batch 下 β₁ 接近 β₂ 的经验。本文从分布稳定性角度补充其理论基础。
- **低精度训练（FP8-LM, Fishman et al., 2025）**：探索 FP8 训练 LLM。本文证明即使 m_t 也用 FP8、y_t 仅 4-bit 仍可保持性能，为低精度优化器状态表示提供新思路。

## 局限性与未来方向
- 实验仅限于 Transformer 语言模型，最大 1B 参数；未验证更大规模或非 Transformer 架构（如 Vision Transformer、扩散模型）。
- 仅研究了 tied-β regime，β₁ ≠ β₂ 的通用场景下的比率行为未分析。
- 理论分析基于局部高斯窗口假设（Assumption 1），实际训练中的非平稳性和非高斯梯度噪声可能偏离该模型。
- 常数 y_const 的匹配基于期望衰减，实际波动较大的 y_t 分布下可能需要更精细的自适应策略。
- 未来方向：扩展到更大规模模型（如百亿参数）、非 Transformer 架构、以及 β₁ ≠ β₂ 的更一般情形；研究自适应而非固定的码本设计。

## 研究启发与可借鉴点
- **状态重参数化思路**：将 coupled 原始状态 (m_t, v_t) 转换为更稳定的内在状态 (m_t, y_t)，这一思路可迁移到其他自适应优化器（如 AdaGrad、RMSprop）的低精度表示设计。
- **固定码本量化策略**：基于目标变量的稳定分布设计非均匀固定码本（而非逐块/逐张量动态缩放），可实现无额外元数据的压缩，适合硬件部署。
- **优化器间的理论桥接**：通过常数近似建立 Adam 与 Signum 之间的显式联系，并提供学习率迁移公式，此类"极限情形分析"方法可用于揭示不同优化器族的内在关联。
- **诊断性实验设计**：先用保留原始递归的诊断实验（仅替换 y_t 的表示）验证稳定性，再发展完整递归实现，这种渐进式验证策略值得借鉴。
- **与团队方向结合机会**：若团队关注低资源训练或高效部署，可将此 4-bit 比率量化方案集成到现有的低精度训练流水线中；也可探索将此稳定比率思想应用于强化学习中的策略优化器。

## 关键术语表
- **Tied-β regime**：指 Adam 中 β₁=β₂=β 的特殊设置，此时 v_t - m_t² 恒非负，具有梯度尺度不变性等良好结构性质。
- **Transformed ratio y_t**：定义 y_t = (1-β)/β · (v_t - m_t²)/m_t²，提取了 Adam 自适应分母中的稳定比率成分。
- **Shifted inverse-square reference**：参考分布 Y_c = 2/(Z+c)²（Z~N(0,1)），用于逼近 y_t 的尾部行为，其生存函数在前两阶依赖 c²=E[A²]。
- **TR-Adam**：本文提出的基于变换比率递归的低精度 Adam 变体，用 4-bit 量化存储 y_t。
- **Signum**：SignSGD with momentum，维护一阶矩 EMA 但更新方向仅用 sign(m_t)，步长固定。
- **Cut64**：对 y_t 施加 min(y_t, 64) 截断的上尾截断策略。
- **Pooled-shift distribution P̂_A**：将多个局部坐标窗口的 shift a_i 扁平化为经验分布，用于理论上的二阶匹配分析。
- **Bias correction**：Adam 中对 m_t 和 v_t 的零初始化偏差校正，分别为 m̂_t = m_t/(1-β¹) 和 v̂_t = v_t/(1-β²)。

## 可复现要素
- **数据集**：FineWeb（预训练）、UltraChat（SFT）、UltraFeedback（RLHF），均为公开数据集。
- **代码**：匿名化代码包作为补充材料提供的压缩存档（论文 NeurlPS Checklist Q5 回答 Yes）。
- **模型**：Llama-style 20M（自定义 plainLM 实现）、Pythia-style 160M、Pythia-1B、Llama-3.2-1B（官方架构）。
- **关键超参**：β ∈ {0.90, 0.92, 0.95, 0.97}；预训练 LR 根据 sweep 选择；SFT LR = 1×10⁻⁵；RLHF actor LR = 1×10⁻⁶；weight decay：短训练 0，长训练 0.1，RLHF actor 0.01。
- **硬件**：1B 规模实验使用 8×H200 GPU，较小规模使用 16×RTX 4090 GPU。
- **实现框架**：plainLM（预训练）、VERL（SFT 和 RLHF）。
