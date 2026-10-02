---
title: "STEPQUANT-WHEN-AND-WHERE-ERRORS-MATTER-IN-DELTA-RULE-RECURRE"
source: https://arxiv.org/pdf/2609.38169v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:24:37"
field: "高效LLM推理与量化"
keywords: ["recurrent state quantization", "linear attention", "post-training quantization", "Delta-rule", "mixed-precision", "Gated DeltaNet", "Kimi Delta Attention"]
innovations: ["基于寿命的混合精度位分配", "按键行影响的双轴量化拟合", "FP16稀疏pivot保护策略"]
benchmarks: ["LiveCodeBench v6", "AIME 2026", "GPQA Diamond", "MMLU", "HellaSwag"]
---

# 论文速读：STEPQUANT-WHEN-AND-WHERE-ERRORS-MATTER-IN-DELTA-RULE-RECURRE

## 一句话总结
本文针对线性注意力模型中门控Delta规则循环状态的量化误差传播问题，提出STEPQuant时空感知后训练量化框架，通过基于寿命的位分配与按键行影响的双轴拟合，在名义6-bit预算下实现接近FP32精度的低比特循环状态压缩。

## 研究问题与动机
- 循环状态的存储开销随并发请求数增长，在Qwen3.8-27B等混合模型中，FP32状态池内存可超过BF16权重内存
- 直接对循环状态施加均匀量化会导致严重精度下降，因为量化误差通过Delta更新递归传播累积
- 现有方法仅关注单一维度（时间或空间），未同时建模误差的传播持久性与关键行对输出的差异化影响

## 核心贡献（创新点）
- **时间维度：基于寿命的位分配**：根据每个状态单元的量化误差大小与记忆保留寿命动态分配位宽，本质区别于传统混合精度只考虑误差幅度
- **空间维度：按键行感知的双轴拟合**：分别为key行和value列分配独立缩放因子，并引入行影响分数加权，区别于单轴outlier处理
- **服务集成与优化**：将量化框架与SGLang的packed-state kernel融合，实现无额外per-token分配开销的在线服务

## 方法详解
- **门控Delta规则**：状态更新公式为 $S_t = (I - \beta_t k_t k_t^\top) D_t S_{t-1} + \beta_t k_t v_t^\top$，其中$D_t$为保留门控
- **误差传播分析**：命题1证明量化误差$E_t = A_t E_{t-1} + \varepsilon_t$，当$\|A_t\|_2 \leq 1$时误差不会放大但会随保留门控衰减
- **寿命权重计算**：$L_u = \sum_{j=0}^{H-1} \exp(2j\ell_u)$，其中$\ell_u$为平均对数保留率
- **行影响分数**：$\omega_i = \mathbb{E}_{cal}[g_{t,i}^2]$，衡量key行$i$对readout误差的贡献
- **双轴缩放**：$\hat{X}_{ij} = r_i c_j z_{ij}$，其中$r_i = m_i^{1/2} w_i^{-1/2}$，$c_j$通过最小化行影响加权重构误差拟合
- **FP16稀疏pivot**：保留1.39%的高风险单元于FP16，其余使用混合精度

## 实验与结果
- **模型与硬件**：Qwen3.8-27B (GDN) 与 Kimi-Linear-48B-A3B-Instruct (KDA)，4×A800 GPU
- **校准数据**：32段WikiText-2，每段2048 tokens
- **长生成基准**：LiveCodeBench v6、EvalPlus、AIME 2026、MATH-500、HMMT Feb 2026、GPQA Diamond、IFBench
- **短生成基准**：MMLU、ARC-C、OpenBookQA、HellaSwag、WinoGrande、LAMBADA
- **核心结果**：6-bit STEPQuant在Qwen上平均准确率80.59%（FP32为80.60%），Kimi上61.47%（FP32为61.52%）；4-bit配置在Qwen上80.51%，优于INT8的71.86%
- **内存压缩**：6-bit状态压缩5.03×，总服务内存减少68.7%（batch=512）
- **推理加速**：状态更新提速2.91×，decode吞吐提升20.53%（Qwen, batch=512）

## 相关工作脉络
- **Q-Mamba**：面向Mamba状态空间模型的双轴量化，本文对比显示其4-bit效果远低于STEPQuant（7.64% vs 84.72%）
- **DAMP（同期工作）**：使用衰减持久性选择FP16保护通道，量化至INT8，有效位宽9.9-bit，STEPQuant在6.3-bit下达到相当精度
- **SmoothQuant**：单轴outlier平滑，不适用于循环状态的双轴异常结构
- **Quamba/Quamba2**：针对选择性SSM的权重与激活量化，以及缓存状态8-bit量化
- **KIVI**：KV cache轴不对称量化，不同于循环状态的递归误差传播特性

## 局限性与未来方向
- 寿命权重仅通过门控衰减近似误差持久性，未完全建模时变且key依赖的状态转移
- KDA在4-bit+BF16权重下长生成任务仍有精度损失
- 仅在两款GDN/KDA模型上验证，其他架构与动态服务场景待检验
- 未来可探索更精确的误差传播建模与跨架构泛化

## 研究启发与可借鉴点
- **时空分解思路**：将量化误差分解为时间持久性与空间影响两个正交维度，为其他递归状态压缩提供分析框架
- **行影响加权双轴拟合**：基于readout误差的梯度信息设计缩放因子，可迁移至其他矩阵压缩场景
- **FP16稀疏pivot策略**：仅需保护极少量高风险单元即可显著提升低比特效果，节省硬件复杂度的同时保证精度
- **在线服务集成设计**：离线校准+在线packed-state kernel融合的方案，避免per-token开销，适合生产部署

## 关键术语表
- **Gated DeltaNet (GDN)**：头级标量保留门控的门控Delta规则线性注意力架构
- **Kimi Delta Attention (KDA)**：通道级保留门控的Delta规则扩展，支持细粒度遗忘控制
- **Lifetime-aware Bit Allocation**：基于门控保留寿命与量化误差的混合精度位分配策略
- **Key-Row-Aware Dual-Axis Fitting**：结合行影响分数与双轴magnitude的独立缩放拟合方法
- **FP16 Pivots**：保留最高风险状态单元于全精度以缓解量化误差
- **Readout Error**：循环状态量化误差经$g_t = A_t^\top q_t$加权后对输出的影响
- **SGLang Packed-State Kernel**：融合tilewise重建、Delta更新与readout的GPU内核算子

## 可复现要素
- **数据集**：WikiText-2（校准），LiveCodeBench v6、EvalPlus、AIME 2026、MATH-500、HMMT Feb 2026、GPQA Diamond、IFBench（评估）
- **代码**：已开源（https://github.com/Dreamer-Toby/STEPQuant）
- **模型权重**：Qwen3.8-27B与Kimi-Linear-48B-A3B-Instruct（官方公开）
- **关键超参**：校准样本32段×2048 tokens，候选位宽{2,4,6,8}，pivot比例1.39%，γ=0.25
- **硬件**：4×NVIDIA A800 GPU，TP4
