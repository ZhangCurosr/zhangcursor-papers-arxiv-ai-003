---
title: "VISIONWEAVE-WEAVING-ELASTIC-VISUAL-REPRE-SENTATIONS-AS-A-NAT"
source: https://arxiv.org/pdf/2610.07987v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 22:28:57"
---

# 论文速读：VISIONWEAVE-WEAVING-ELASTIC-VISUAL-REPRE-SENTATIONS-AS-A-NAT

## 一句话总结
本文提出VisionWeave，通过在前沿MLLM中进行三阶段自蒸馏，首次将弹性视觉表示编织建立为模型原生能力；该方法利用门控空间池化器和粒度路由器自适应选择细粒度/粗粒度视觉token，在平均节省43%视觉token的同时保持原模型98.9%性能，并在SGLang部署中实现2.30倍吞吐提升。

## 研究问题与动机
1. MLLMs处理高分辨率图像和长视频时，固定大小patch token序列导致计算和内存开销巨大，但视觉信息分布不均（精细区域需细节、背景可压缩），现有方法难以兼顾效率与质量。
2. 视觉token剪枝/合并方法（FastV、VisionZip）存在训练-推理不匹配、丢弃任务关键信息、依赖预设压缩比等问题，在信息密集输入和grounding任务上性能严重下降。
3. 现有自适应分辨率方法（如ViCO）基于tile级别分配单一分辨率，无法处理细节丰富区域与低信息背景交错场景的空间分配需求，且难以集成到现代MLLMs架构和SGLang等推理引擎中。
4. 缺乏将内容自适应视觉处理作为MLLMs原生能力进行大规模验证的系统性工作。

## 核心贡献（创新点）
1. **原生弹性视觉表示编织能力**：通过三阶段自蒸馏首次在前沿MLLM中原生建立内容自适应的粒度选择能力，与ViCO等tile级方法本质不同，实现了更精细的空间粒度分配并原生集成到Qwen架构。
2. **门控空间池化器设计**：提出基于内容感知门控的2×2 patch聚合模块，生成可与细粒度表示共享MRoPE坐标系的粗粒度token，与简单mean pooling或pixel unshuffle的本质区别在于引入了可学习的channel-wise门控权重。
3. **多深度粒度路由器**：设计在ViT多层特征上进行本地-全局交叉注意力探查的决策模块，学习每个空间块的粒度分配，与固定压缩比方法的本质区别在于实现了基于语义内容的动态自适应分配。
4. **三阶段自蒸馏训练策略**：提出粗粒度表示→软混合可微代理→硬路由混合序列的渐进式训练流程，使模型逐步学习弹性处理能力，与直接微调方法的本质区别在于完全依赖蒸馏信号而无需额外标注数据。
5. **SGLang服务引擎原生集成**：实现块中心MRoPE坐标支持和混合粒度序列支持的部署方案，证明弹性视觉编织可转化为实际推理吞吐增益，与仅停留在基准评估方法的本质区别在于提供了端到端服务效率的完整验证。

## 方法详解
1. **整体架构**：VisionWeave包含门控空间池化器（构建粗粒度表示V_coarse）和粒度路由器（学习自适应分配），在共享MRoPE坐标系下为每个空间块选择细粒度或粗粒度token。

2. **门控空间池化器（Gated Spatial Pooler）**：
   - 将ViT patch特征的每个2×2邻域映射为单个池化特征：V_coarse = M(P(X)) ∈ R^(H/4×W/4×D)
   - 对4个patch特征进行LayerNorm后聚合全局上下文g_k = f_glb([u_1;u_2;u_3;u_4]) + b_k
   - 通过gate网络计算channel-wise softmax权重α_k = softmax(f_gate([u_k;g_k]))
   - 最终输出P(x_{1:4}) = Σ α_k ⊙ (u_k + f_val([u_k;g_k]))，零初始化可将池化器退化为mean pooling

3. **粒度路由器（Granularity Router）**：
   - 为每个空间块初始化可学习查询h_IJ，使用块中心MRoPE坐标(4I+3/2, 4J+3/2)
   - 在ViT深度{0, L/2, L}进行探查，每个深度包含本地交叉注意力（关注4×4 patch）和全局交叉注意力（关注全图）
   - 通过线性头输出粗粒度概率p_IJ = softmax(W_r LN(h_IJ))_coarse，超过阈值t选择V_coarse，否则选择4个native fine tokens

4. **三阶段自蒸馏**：
   - **Stage 1**：仅训练池化器，路由旁路（全部粗粒度），损失L_distill = KL(teacher_top512 || student_top512)
   - **Stage 2**：训练路由器，使用软混合代理V_mix = m_IJ·N((1-p)V̂_fine + pV̂_coarse)，加入平衡损失L_bal鼓励粗路由比例达到ρ=0.8
   - **Stage 3**：在硬路由混合序列上自蒸馏，同时更新池化器和LLM，使用离线校准的粗路由偏置

5. **训练配置**：约78万样本（图像246K+视频281K+GUI grounding 250K），图像预算4096 tokens，视频每帧512 tokens最多64帧；BF16训练，Adam优化器，峰值学习率1e-4→1e-5→3e-6，总成本超30K A100 GPU小时。

## 实验与结果
1. **评估设置**：Qwen3.5-4B和Qwen3.8-27B，对比基线FastV†（在ViT后选择）、VisionZip、输入降采样；8个基准测试涵盖自然图像、文档理解、GUI grounding、视频理解等。

2. **主要内容自适应token节省**（Table 4，输入预算512 tokens/image）：
   - Qwen3.8-27B平均节省43.0% token，性能保留98.9%（损失1.08分）
   - 文档任务保守（DocVQA节省27.3%损失1.23分，InfoVQA节省16.5%损失3.40分）
   - 视频任务激进（VideoOCR节省49.6%，Video-MME节省54.2%，LongVideoBench节省53.0%）
   - Grounding任务ScreenSpotV2节省55.1% token仅损失3.17分
   - 固定50%节省目标的FastV†和VisionZip平均性能仅保留88%（损失11.82%和12.46%）

3. **匹配token节省下的性能对比**：即使给FastV†/VisionZip匹配VisionWeave的观察节省比例，VisionWeave仍保持最高平均得分（Qwen3.8-27B: 76.73 vs 70.10/70.33）。

4. **跨分辨率和帧预算鲁棒性**：在多种图像分辨率和视频帧预算下，VisionWeave始终优于输入降采样；64帧cap下使用更少token即超越128帧native模型1.20分（LongVideoBench）。

5. **SGLang服务效率**（Table 5，A100-80GB×2，TP=2）：
   - 端到端吞吐2.30倍提升（0.90→2.07 req/min）
   - 平均TTFT降低54.4%（169.90→77.54s），P95降低57.1%
   - 平均TPOT降低60.6%（355.64→140.29ms），P95降低59.6%

6. **消融实验**（Table 6）：随机打乱路由结果使Qwen3.8-27B平均性能下降5.36分，证明学习内容自适应分配至关重要。

## 相关工作脉络
1. **视觉token剪枝/合并**：FastV（Chen et al., 2024）基于attention weight提取在LLM内部剪枝，VisionZip（Yang et al., 2024）采用token merging策略，两者均依赖预设压缩比、训练-推理不匹配，在grounding和文档任务上性能下降超过47%。
2. **自适应分辨率ViT**：Dynamic Granularity（Yu et al., 2025）和AdaPatch（Choudhury et al., 2025）基于边缘密度和熵调整patch size，但主要针对独立视觉任务，未针对多模态下游目标优化粒度标准。
3. **ViCO（Cui et al., 2025）**：最接近本文工作，基于tile级别自适应分配视觉token分辨率，但tile级设计限制了在现代MLLMs中的直接应用性，且单resolution per tile的空间粒度不足以处理交错分布的细节区域。
4. **Matryoshka系列（Cai et al., 2024; Hu et al., 2024）**：支持多token预算但需外部指定，缺乏自主内容自适应能力，与本文的端到端学习内容自适应形成对比。
5. **KV-cache压缩**：与视觉token效率关注点不同，KV-cache压缩（如DeepSeek-V4）关注存储压缩而非序列缩短，两者结合可形成更完整的效率方案。

## 局限性与未来方向
1. 训练计算成本高（30K+ A100 GPU小时），限制了更全面的架构探索（如不同router设计、更多粒度级别）。
2. 当前仅支持两级粒度（细/粗），未探索多尺度表示的潜在增益。
3. 能力通过post-training self-distillation植入，未来可探索在base model预训练阶段直接引入弹性视觉编织。
4. 消融实验仅对比了pixel-unshuffle替代方案，对其他pooler架构设计的探索有限。

## 研究启发与可借鉴点
1. **渐进式自蒸馏策略**：三阶段（粗粒度→软混合→硬路由）的设计思路可用于其他需要学习离散选择的模型训练，软代理过渡到硬选择的技巧值得借鉴。
2. **共享坐标系下的多粒度表示**：将不同粒度token置于同一MRoPE坐标系中进行路由选择而非直接丢弃token，保持了空间覆盖完整性，避免了信息密集区域被误删的问题。
3. **平衡损失的可迁移设计**：从MoE训练借鉴的per-sample balance loss有效控制了路由分布，防止router坍缩，该技巧可迁移到其他需要负载均衡的模型中。
4. **服务引擎原生集成验证**：不仅停留在基准测试，而是将方法集成到SGLang中验证实际部署收益，提供了从算法设计到系统优化的完整闭环验证范式。
5. **推理时灵活性**：单一模型通过调整单一阈值即可在不同质量和成本需求间切换而无需重新训练，为实际部署提供了极大的灵活性。

## 关键术语表
**Elastic Visual Representation Weaving**：弹性视觉表示编织，指MLLM根据视觉内容自适应地在细粒度和粗粒度表示间进行选择的能力，实现计算资源按信息内容动态分配的原生机制。

**Gated Spatial Pooler**：门控空间池化器，一种可学习的空间池化模块，通过内容感知的channel-wise门控机制将4个相邻patch特征聚合为单个粗粒度特征，零初始化时可退化为mean pooling。

**Granularity Router**：粒度路由器，一个基于ViT多层特征探查的决策模块，通过本地-全局交叉注意力学习每个空间块应使用细粒度还是粗粒度表示的概率分配。

**MRoPE (Multi-Rate Positional Encoding)**：多速率位置编码，MLLM中用于视觉token的位置编码方案，为每个token分配时空坐标(τ, i, j)，支持不同粒度token在统一坐标系中准确对齐。

**Soft-Mixing Surrogate**：软混合代理，在可微分训练中通过
