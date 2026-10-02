---
title: "STEPQUANT-WHEN-AND-WHERE-ERRORS-MATTER-IN-DELTA-RULE-RECURRE"
source: https://arxiv.org/pdf/2609.38169v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:24:56"
field: "高效 LLM 推理与量化"
keywords: ["recurrent state quantization", "post-training quantization", "gated delta networks", "linear attention", "mixed-precision quantization", "SGLang serving"]
innovations: ["提出时空双维误差分析框架，揭示 Delta-rule 循环状态量化误差的时间累积与空间非均匀性", "设计 Lifetime-aware Bit Allocation 实现基于记忆寿命的混合精度位分配", "提出 Key-Row-Aware Dual-axis Fitting 将 readout impact 嵌入双轴尺度拟合"]
benchmarks: ["Live-CodeBench v6", "AIME 2026", "MATH-500", "GPQA Diamond", "HMMT Feb 2026", "MMLU", "ARC-C"]
---

# 论文速读：STEPQUANT-WHEN-AND-WHERE-ERRORS-MATTER-IN-DELTA-RULE-RECURRE

## 一句话总结
本文提出 STEPQuant，一种针对 Delta-rule 循环注意力模型（GDN/KDA）的时空后训练量化框架，通过分析量化误差的时间累积特性与空间分布结构，在名义 6-bit 预算下实现接近 FP32 精度的循环状态压缩，并在 SGLang 中集成实现最高 68.7% 的总内存节省。

## 研究问题与动机
- **循环状态成为并发服务瓶颈**：线性注意力用固定大小状态替代 KV cache，但每个并发请求需独立持久状态，FP32 状态下 Qwen3.8-27B 的 state pool 在 70 并发时已超过 BF16 权重内存。
- **均匀量化导致严重精度退化**：直接 INT4/INT6 量化使长生成任务平均准确率暴跌至 12–45%，均匀量化假设所有状态单元同等重要，忽略了误差传播机制。
- **量化误差具有时空非均匀性**：误差沿解码步骤递归累积（时间维度），且在 key rows 和 value columns 两个轴上均存在显著量级差异（空间维度），现有单轴量化方法无法同时适配两种结构。

## 核心贡献（创新点）
- **时空误差分解分析**：从理论上证明 Delta-rule 下量化误差的传播公式 $E_t = A_t E_{t-1} + \varepsilon_t$，并实证 gate half-life 与累积误差呈强正相关（Spearman $\rho \approx 0.80$），这是现有工作未系统研究的误差动力学机制。
- **Lifetime-aware Bit Allocation（时间维度）**：基于 mean log retention $\ell_u$ 计算 lifetime weight $L_u = \sum_{j=0}^{H-1} \exp(2j\ell_u)$，在固定比特预算下以动态规划/Lagrangian 分配精度，将高风险单元保留为 FP16 sparse pivots（仅占 1.39% heads）。
- **Key-Row-Aware Dual-axis Fitting（空间维度）**：定义 row-impact score $\omega_i = \mathbb{E}[g_{t,i}^2]$ 衡量 key row 对输出误差的贡献，采用行尺度 $r_i = m_i^{1/2}w_i^{-1/2}$ 和列尺度 $c_j$ 的二维分离拟合，区别于仅单轴缩放的 SmoothQuant/Q-Mamba。
- **端到端 Serving 集成**：将量化框架嵌入 SGLang 0.5.12，通过 fused CUDA kernel 实现 tilewise 重建+Delta更新+readout 融合，写回阶段在独立 stream 上重叠执行，6-bit 配置实现 5.03× 状态压缩与 2.91× 加速状态更新。

## 方法详解
**时间维度——生命周期感知位分配：**
- 将量化单元定义为：Qwen 为整个 head（$n_u = d_k \cdot d_v$），KDA 为 key row（$n_u = d_v$）。
- 估算每个单元的 mean log retention $\ell_u = \mathbb{E}[\log r_{t,u}]$，其中 $r_{t,u}$ 为 gate retention（GDN 为标量 $\alpha_t$，KDA 为通道级 $d_{t,i}$）。
- 计算 lifetime weight $L_u = \sum_{j=0}^{H-1} \exp(2j\ell_u)$，反映量化误差在多步解码中的累积衰减。
- 优化目标：$\min_{b_u} \sum_u L_u d_u(b_u)$，受限于 $\sum_u n_u b_u \leq \bar{b} \sum_u n_u$，其中 $d_u(b)$ 为候选精度 $b$ 下的重建失真（Qwen 用 row-impact 加权 MSE，KDA 用未加权 MSE）。
- 最困难单元保留为 FP16 pivots（Qwen 选 32 heads，KDA 选 512 rows），不参与共享列尺度拟合。

**空间维度——Key-Row 感知双轴拟合：**
- Row factor：$r_i = m_i^{1/2} w_i^{-1/2}$，其中 $m_i = \frac{1}{d_v}\sum_j |X_{ij}|$ 为行幅度，$w_i$ 为 normalized $\omega_i^{1/8}$（Qwen）或 $\omega_i^{1/4}$（KDA），高 impact 行获得更精细量化分辨率。
- Column fitting：$\min_{\{c_j>0\}} \sum_{i,j} w_i^2 (X_{ij} - r_i c_j z_{ij})^2$，在高 impact 行上施加更大惩罚。
- 重建公式：$\hat{X}_{ij} = r_i c_j z_{ij}$，每头存储一个 FP16 row vector + 一个 FP16 column vector + 整数 codes。

**部署优化：**
- Prefill 阶段 unpack → native chunk kernel → pack；decode 阶段直接 on compressed pages 执行 fused kernel。
- 写回（scale fitting + packed writeback）在独立 CUDA stream 上与后续 layer 计算重叠。

## 实验与结果
**基准设置：**
- 模型：Qwen3.8-27B（GDN，48 recurrent layers × 48 heads）、Kimi-Linear-48B-A3B-Instruct（KDA，20 layers × 32 heads）。
- 硬件：4× NVIDIA A800，BF16 权重 + 4-bit AWQ 权重两种设置。
- 校准数据：32 段 WikiText-2（每段 2048 tokens）。
- 长生成任务（7 个）：Live-CodeBench v6、EvalPlus、AIME 2026、MATH-500、HMMT Feb 2026、GPQA Diamond、IFBench。
- 短生成任务（6 个）：MMLU、ARC-C、OpenBookQA、HellaSwag、WinoGrande、LAMBADA。

**主要结果（BF16 权重）：**
| 配置 | Qwen Avg（长生成） | Kimi Avg（长生成） | Qwen Avg（短生成） | Kimi Avg（短生成） |
|---|---|---|---|---|
| FP32 | 80.60% | 61.52% | 87.78% | 68.36% |
| INT8 | 71.86% | 56.02% | 86.25% | 67.80% |
| INT6 | 45.04% | 45.70% | 82.49% | 61.35% |
| INT4 | 12.73% | 21.63% | 65.74% | 42.24% |
| **STEPQuant@6bit** | **80.59%** | **61.47%** | **87.54%** | **68.82%** |
| **STEPQuant@4bit** | **80.51%** | **58.52%** | **87.63%** | **68.11%** |

- **最强结果**：STEPQuant@6bit 在 Qwen 上达到 80.59%（vs FP32 80.60%，差距 0.01pt），在 Kimi 上达到 61.47%（vs FP32 61.52%，差距 0.05pt）；4-bit 配置远优于 uniform INT8/INT6/INT4。
- **与 W4 AWQ 兼容**：STEPQuant@6bit + W4 权重，Qwen 平均 79.27%（vs FP32 79.32%，差距 0.05pt），Kimi 58.62%（vs 58.95%，差距 0.33pt）。
- **效率**：STEPQuant@6bit 在 batch=512 时将 Qwen 总内存从 419.73 GiB 降至 131.18 GiB（-68.7%），循环状态压缩 5.03×（-80.1%），状态更新加速 2.91×；decode throughput 提升 20.53%（6,040→7,280 tokens/s）。
- **Generation Length**：均匀量化导致"过度思考"（如 Kimi INT4 在 AIME 上平均生成 63.4K tokens 而准确率近 0），STEPQuant 保持与 FP32 相近的输出长度。

**Ablation（Qwen，4-bit）：**
- Spatial only：73.95% vs FP32 84.61%
- Temporal only：12.87%（无 pivots 时仅 6.17%）
- STEPQuant 完整：84.72%（接近 FP32）

## 相关工作脉络
- **Quamba/Quamba2（Chiang et al., 2025）**：针对 Mamba 类 SSM 的权重+激活量化，Quamba2 将 cached recurrent states 量化到 8-bit；本文针对 GDN/KDA  hybrid 架构，支持低至 4-bit 且精度损失更小。
- **Q-Mamba（Tianqi et al., 2025）**：提出 Dual-axis State Quantization（DSQ）用于 Mamba state-cache，但缺少时间维度误差分析；本文 spatial only（使用 DSQ 适配版）在 4-bit 下仅 73.95%，而完整 STEPQuant 达 84.72%，说明 temporal 组件对累积误差补偿的关键作用。
- **DAMP（Zhang et al., 2026，并行工作）**：用 decay-based persistence 选 FP16 key channels，其余量化为 INT8；本文在相同 benchmark 上以更低 effective bit（6.30 vs 9.9）达到相近精度保留率（100.51% vs 100.99%）。
- **KIVI（Liu et al., 2024）**：KV cache 的双轴 2-bit 量化；本文对象是固定大小循环状态而非增长型 KV cache，量化目标与误差传播机制均不同。
- **SmoothQuant（Xiao et al., 2023）/OmniQuant（Shao et al., 2024）**：LLM 权重的单轴 outlier smoothing；本文发现循环状态在 key-row 和 value-column 双轴均有显著 outlier，且需结合 readout impact 加权。
- **标准线性注意力（RetNet/GDN）**：提供 fixed-size state 架构基础，但未解决低比特量化下的误差累积问题。

## 局限性与未来方向
- **时间误差近似**：lifetime weight 通过 gate decay 近似误差持久性，未完整建模 time-varying、key-dependent 的状态转移 $A_t$，长期累积效应估计可能存在偏差。
- **W4 权重下 KDA 4-bit 退化**：Kimi 在 4-bit + W4 设置下平均准确率下降 2.29pt，长生成任务产生更长输出但仍维持可用精度，说明极端低比特与权重量化联合时的误差补偿仍需优化。
- **评估范围有限**：仅验证 GDN（Qwen）和 KDA（Kimi）两类架构，固定 A800 硬件与静态 workload，其他 architecture（如纯 Mamba、RetNet）及动态 serving 场景的有效性待验证。
- **Pivot 选择的离散化**：FP16 pivots 比例为固定 1.39%，未探索随任务/负载自适应调整 pivot 比例的策略。

## 研究启发与可借鉴点
- **误差的时空分解范式**：将量化误差拆解为"何时累积"（时间/传播深度）与"何地影响大"（空间/敏感性）两个正交维度，为其他递归/状态空间模型的量化提供通用分析框架。
- **Row-impact 加权双轴拟合**：将 readout sensitivity（$g_t = A_t^\top q_t$）嵌入 scale 计算，而非仅依赖 magnitude；可迁移到任何具有"状态→输出"线性 readout 的结构化模型量化。
- **FP16 sparse pivots + mixed-precision 组合策略**：仅保留极少高风险单元为高精度，其余动态分配低位宽，兼顾精度与存储效率；该思路可与 weight quantization（如 AWQ/GPTQ）自然组合。
- **融合 kernel 设计模式**：reconstruction + Delta update + readout 三操作 fuse 为一个 tilewise kernel，写回阶段在独立 stream 重叠执行，为 low-bit recurrent serving 的 kernel 优化提供参考。
- **Generation length 作为辅助指标**：量化模型的推理退化不仅体现在 accuracy 下降，还表现为输出长度膨胀（"overthinking"）；建议将 generation length 纳入常规评测，辅助诊断量化损伤。

## 关键术语表
- **Gated DeltaNet (GDN)**：Yang et al. (2025) 提出的 gated linear attention 架构，使用 head-wise 标量 gate 控制 memory retention，结合 delta rule 进行 state 更新。
- **Kimi Delta Attention (KDA)**：Kimi Linear 模型使用的变体，将 retention gate 扩展为 channel-wise（per-key-row）对角矩阵，支持更细粒度的选择性遗忘。
- **Lifetime-aware Bit Allocation**：基于 mean log retention 估计各状态单元的"记忆寿命"，以 lifetime weight 加权重建失真进行混合精度位分配的离线校准策略。
- **Key-Row-Aware Dual-axis Fitting**：在 state 矩阵的 key-row 轴和 value-column 轴分别拟合尺度因子，并将 readout impact score 嵌入 row scale 的二维量化参数估计方法。
- **FP16 Pivots**：识别并保留量化风险最高的少数状态单元（head 或 row）为全精度 FP16，避免其对整体精度的破坏性影响。
- **Recurrent State Pool**：SGLang 等 serving 系统中为并发请求维护的持久循环状态集合，内存占用随并发数线性增长，是 low-bit 压缩的主要目标。
- **Overthinking Phenomenon**：量化模型在推理中产生异常长的输出序列（接近 token limit）但未提升准确率的退化现象，与均匀量化下的噪声累积相关。
- **Conditional Error Propagation**：Proposition 1 描述的量化误差递推公式 $E_t = A_t E_{t-1} + \varepsilon_t$，其中 $\|A_t\|_2 \leq 1$ 保证误差不放大，但可长期累积。

## 可复现要素
- **代码**：开源，地址 https://github.com/Dreamer-Toby/STEPQuant
- **数据集**：校准数据使用 32 段 WikiText-2（公开）；评测使用 Live-CodeBench v6、EvalPlus、AIME 2026、MATH-500、HMMT Feb 2026、GPQA Diamond、IFBench、MMLU、ARC-C、OpenBookQA、HellaSwag、WinoGrande、LAMBADA（均为公开 benchmark）。
- **模型权重**：Qwen3.8-27B（BF16 与 4-bit AWQ）、Kimi-Linear-48B-A3B-Instruct（BF16 与 4-bit AWQ），需从官方渠道获取。
- **关键超参**：
  - 校准样本数：32 段 × 2048 tokens
  - Pivot 数量：Qwen 32 heads / KDA 512 rows
  - Nominal bit budget：4-bit / 6-bit（candidate set {2,4,6,8} 或 {4,6,8}）
  - Row-impact exponent：Qwen $\gamma=0.25$（$w_i \propto \omega_i^{1/8}$），KDA $\gamma=0.25$（$w_i \propto \omega_i^{1/4}$）
  - Hardware：4× NVIDIA A800，TP4
  - SGLang 版本：0.5.12
- **复现难度**：中等；代码已开源，主要依赖 SGLang 生态与 A800 GPU；校准流程与 allocation 算法在 appendix C 中有详细描述。
