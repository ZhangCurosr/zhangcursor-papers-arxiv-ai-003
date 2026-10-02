---
title: "The-Hidden-Ratio-in-Adam-Stable-Structure-Compression-and-Si"
source: https://arxiv.org/pdf/2609.35392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:14:31"
field: "优化器理论与高效训练"
keywords: ["Adam optimizer", "tied-beta regime", "model compression", "quantization", "sign-based optimization", "transformer training"]
innovations: ["发现Adam变换比率y_t的稳定重尾分布并提出递归表征", "提出TR-Adam用固定4-bit码本压缩二阶状态无需辅助缩放", "建立Adam与Signum间的量化学习率转移规则"]
benchmarks: ["FineWeb预训练(Llama 20M/Pythia 160M/1B)", "UltraChat SFT(Pythia-1B/Llama-3.2-1B)", "UltraFeedback RLHF(Llama-3.2-1B)"]
---

# 论文速读：The-Hidden-Ratio-in-Adam-Stable-Structure-Compression, and-Sign-Dynamics

## 一句话总结
论文在 Adam 的 tied-β  regime（β₁=β₂=β）下，发现了一个变换后比率 y_t，其具有稳定且重尾的分布；基于此提出 TR-Adam，用固定 4-bit codebook 替代原始二阶矩状态，在语言模型预训练、SFT 和 RLHF 中保持与全精度 Adam 相当的性能，同时揭示了 Adam 与 Signum 之间的联系。

## 研究问题与动机
- Adam 作为深度学习默认优化器，其自适应行为因一阶/二阶矩 EMA 的复杂交互而难以理解。
- 现有大规模训练常用接近的衰减率（如 (0.9, 0.95)），tied-β regime 在结构和稳定性上更为简洁，且能保证 v_t - m_t² ≥ 0（方差似残差）。
- 原始矩量级（m_t, v_t）随梯度尺度变化剧烈，缺乏跨任务/模型的稳定表征，难以指导高效压缩。
- 需要一种能隔离随机波动与确定性缩放的变量，用于分析 Adam 的本质自适应结构。

## 核心贡献（创新点）
- **发现稳定变换比率 y_t**：在 tied-β regime 下定义 y_t = (1-β)/β · (v_t - m_t²)/m_t²，证明其在不同任务、模型规模和训练阶段具有稳定的重尾分布，区别于原始矩量的剧烈波动。
- **推导 y_t 的标量递归**：得到递推式 y_t = β x_t² y_{t-1} + (1-x_t)²（其中 x_t = m_{t-1}/m_t），使 Adam 可用 (m_t, y_t) 参数化替代 (m_t, v_t)，实现二阶状态的压缩存储。
- **提出 4-bit 固定码本量化方案**：基于 y_t 的稳定分布，仅需 16 个量化级别（如 FP4-like 码本 {0.25, 0.5, …, 64}）即可在无需辅助缩放的条件下，使 TR-Adam 在预训练/SFT/RLHF 中几乎匹配全精度 Adam。
- **揭示 Adam 与 Signum 的定量联系**：将 Adam 解读为带随机衰减的符号动量方法，通过 y_t 的期望衰减推导学习率转移规则，并证明常数近似 y_const ≈ 3 可将 Adam 退化为 Signum 类方法，且保留 Adam 的 learning-rate 尺度。

## 方法详解
- **tied-β regime**：令 β₁=β₂=β，此时 v_t - m_t² ≥ 0 对所有梯度序列成立，具有方差似解释；定义 r_t = (v_t - m_t²)/m_t² ≥ 0，则自适应因子可写为 m_t/√v_t = sign(m_t)/√(1+r_t)。
- **变换比率 y_t**：引入 y_t = (1-β)/β · r_t，使 Adam 更新写为 Δθ_t ∝ sign(m_t)/√(1 + β/(1-β) · y_t)。该变换分离了 β-依赖的确定性尺度与随机波动。
- **统计结构**：在局部高斯窗口假设下，y_t ≈ 2/(Z+a)² · (1 + η_t/2)，其中 Z~N(0,1)，A 为局部 shift 的池化分布。定理证明 y_t 的生存函数可由固定参考 Y_c = 2/(Z+c)² 二阶匹配（c² = E[A²]）。
- **递归推导**：由 m_t = β m_{t-1} + (1-β)g_t 解出 g_t，代入 v_t 递归得到 y_t = β x_t² y_{t-1} + (1-x_t)²，其中 x_t = m_{t-1}/m_t。该递推仅需一维标量状态。
- **量化策略**：截断长尾（cut64: min(y_t, 64)）后使用固定 16 级码本 Q(y_t) = argmin_{q∈C} |y_t - q|，无需 per-block 或 per-tensor 缩放因子。一阶矩 m_t 可保留 FP32 或降至 FP8。
- **Signum 极限**：用常数 y_const 替代 y_t，得到 Δθ_t ≈ -η_adam · (1 + β/(1-β)y_const)^(-1/2) · sign(m_t)。通过期望衰减匹配可得 y_const ≈ 3.2（见表1），使 Signum 类的 learning rate 与 Adam 尺度对齐。

## 实验与结果
- **数据集与模型**：FineWeb 预训练（Llama-style 20M、Pythia-style 160M、Pythia-1B）、UltraChat SFT（Pythia-1B、Llama-3.2-1B）、UltraFeedback RLHF（Llama-3.2-1B + Skywork-Reward-V2）。
- **基线**：全精度 tied-β Adam，以及 Signum。
- **主要结果（Table 2-4）**：
  - 预训练（Llama 20M, β=0.95）：Adam=3.672，TR-Adam(FP32-m_t, 4-bit y_t)=3.672，Δ=0.000。
  - 预训练（Pythia 1B, β=0.92）：Adam=2.996，TR-Adam=2.977，Δ=-0.019（TR-Adam 更优）。
  - SFT（Llama-3.2-1B, β=0.95）：Adam=1.082，TR-Adam=1.073，Δ=-0.009。
  - RLHF ReMax 奖励（β=0.90）：Adam=0.788，TR-Adam=0.825，Δ=+0.036（提升显著）。
- **学习率敏感性**：LR sweep（图7）显示 TR-Adam 在 β∈{0.90, 0.92, 0.95, 0.97} 下与 Adam 的最优 LR 区域高度对齐，退化阈值相近。
- **最强结果**：在 RLHF 任务中 TR-Adam(FP32-m_t, 4-bit y_t) 以 β=0.90 达到奖励 0.825，较 Adam 提升 +0.036；预训练中部分配置实现零差距甚至轻微超越。

## 相关工作脉络
- **Adam (Kingma & Ba, 2015)**：本文在 tied-β regime 下重新解释其自适应分母，提出 y_t 变换而非直接用原始 v_t。
- **Signum/SignSGD with momentum (Bernstein et al., 2018; Sun et al., 2023)**：本文揭示 Adam 可视为带随机衰减的 Signum，且常数近似 y_const≈3 时退化为 Signum 类方法，提供学习率转移规则。
- **Adam 结构分析 (Orvieto & Gower, 2025)**：前者从信噪比角度解释 Adam 分母；本文进一步提炼出稳定分布的 y_t 变量，支持量化与压缩。
- **tied-β 结构性质 (Fernández-Hernández et al., 2026)**：证明 β₁=β₂ 时 Adam 具有一阶梯度尺度不变性；本文在此基础上推导 y_t 递归与分布特征。
- **低精度训练 (Fishman et al., 2025; Chitsaz et al., 2024)**：FP8 训练相关工作聚焦于权重/激活量化；本文首次证明二阶矩派生状态 y_t 可用固定 4-bit codebook 压缩且无需辅助缩放。
- **非平稳优化 (Dohare et al., 2023; Ellis et al., 2024)**：指出 β₁ 与 β₂ 失配可能导致更新不稳定；本文从分布稳定性的角度为 tied-β 提供新的理论依据。

## 局限性与未来方向
- 实验仅限 Transformer 语言模型，规模最大 1B 参数；更大模型及非 Transformer 架构（如卷积网络、多模态）尚待验证。
- 理论分析基于局部高斯窗口假设，实际训练梯度的非高斯、非平稳特性未充分刻画。
- tied-β 虽在结构上更简洁，但是否在所有场景（小 batch、特定 RL 任务）下均优于经典 (0.9, 0.999) 配置仍需更多实证。
- 4-bit 固定码本对分布漂移的鲁棒性仅在小尺度实验中检验，长期预训练的累积误差未知。
- 未来方向包括：扩展至更大模型/多模态任务、探索自适应码本或在线校准机制、将 y_t 视角推广至其他自适应优化器（如 AdaGrad、RMSProp）。

## 研究启发与可借鉴点
- **稳定量提炼用于量化**：寻找原始状态中分布更稳定的变换变量（如本文的 y_t），可大幅降低量化代价，此思路可迁移至其他优化器状态压缩。
- **无辅助缩放的固定码本**：TR-Adam 仅需 16 级预定义码本即达全精度性能，避免 per-block/per-tensor 缩放因子的额外开销与实现复杂度，工程上更易部署。
- **优化器间学习率转移规则**：通过期望衰减匹配推导 η_sign ≈ η_adam · (1 + β/(1-β)·y_const)^(-1/2)，为 Signum 类方法与 Adam 之间的超参调优提供理论起点，减少网格搜索成本。
- **递归替代耦合状态**：将 (m_t, v_t) 耦合递归转化为 (m_t, y_t) 标量递归，减少状态维度且保留自适应信息，类似降维思想可应用于其他 EMA 类优化器。
- **后训练中的正则化效应**：TR-Adam 在 SFT/RLHF 中常略优于 Adam，粗粒度 y_t 可能起到 mild smoothing/regularization 作用，值得探索其与训练稳定性的关联。

## 关键术语表
- **tied-β regime**：Adam 中令 β₁=β₂=β 的特殊配置，此时 v_t - m_t² ≥ 0，具有一阶梯度尺度不变性与更简洁的结构。
- **变换比率 y_t**：y_t = (1-β)/β · (v_t - m_t²)/m_t²，分离 β-尺度后的自适应衰减变量，分布稳定且可递归更新。
- **TR-Adam**：本文提出的优化器变体，用递归更新的 y_t 替代 v_t 作为二阶状态，并采用固定 4-bit codebook 量化。
- **FP4-like 固定码本**：16 级非均匀量化码本 C={0.25, 0.5, …, 64}，针对 y_t 的集中 bulk 设计，无需额外缩放元数据。
- **Signum**：SignSGD with momentum，仅保留动量方向 sign(m_t) 并以固定步长更新，无自适应分母。
- **Shifted inverse-square reference**：Y_c = 2/(Z+c)²（Z~N(0,1)），用于二阶近似 y_t 分布的理论参照，其生存函数与 Empirical pooled-shift 分布二阶匹配。
- **Pooled-shift distribution**：将各坐标/窗口的局部 shift a_i 视为样本的混合分布，用于形式化 y_t 的分布稳定性分析。
- **Learning-rate transfer rule**：通过期望衰减匹配 η_sign ≈ η_adam · E[(1 + β/(1-β)y_t)^(-1/2)] 推导两优化器学习率之间的对应关系。

## 可复现要素
- **数据集**：FineWeb（预训练）、UltraChat（SFT）、UltraFeedback（RLHF）；均为公开数据集。
- **代码/权重**：论文附录声明提供匿名化代码包（compressed archive），随补充材料发布；模型从头初始化训练，未使用预训练权重。
- **关键超参**：β∈{0.90, 0.92, 0.95, 0.97}；4-bit 码本 C 固定；m_t 存储精度 FP32 或 FP8；weight decay 在短跑 SFT 实验中为 0，长程预训练为 0.1；SFT LR=1e-5，RLHF actor LR=1e-6。
- **硬件**：1B 模型实验使用 8×H200 GPU，小规模实验使用 16×RTX 4090 GPU。
