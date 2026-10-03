---
title: "ShieldCLIP-Selective-Safety-Alignment-for-Harmful-Content-Mi"
source: https://arxiv.org/pdf/2609.39688v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:15:44"
field: "可信多模态人工智能"
keywords: ["AI Safety", "Vision-Language", "CLIP", "Harmful Content Mitigation", "Selective Alignment", "Multimodal Encoder", "Diffusion Models"]
innovations: ["模态级条件化安全对齐：根据每个生成模态的实际安全状态分别执行保留或重定向，而非按样本来源统一处理", "ViSUv2 数据集：195k 四元组含独立 per-modality 安全标签、578 细粒度概念与 28 类别", "四路条件训练目标：preservation/redirection/mixed-coherence 的结构化损失设计"]
benchmarks: ["ViSUv2", "I2P", "ViSU", "SneakyPrompt", "MMA-Diffusion", "Ring-A-Bell", "LLaVA", "Stable Diffusion v1.4", "SDXL"]
---

# 论文速读：ShieldCLIP: Selective Safety Alignment for Harmful Content Mitigation in Multimodal Foundation Models

## 一句话总结
本文提出了 ShieldCLIP，首个在 CLIP 类多模态编码器中按模态粒度进行选择性安全对齐的框架，通过 ViSUv2 数据集提供的独立模态级安全标签，实现对真实/生成安全内容的保留与仅对不安全模态的重定向，从而在抑制有害内容的同时避免过度清洗原始语义空间。

## 研究问题与动机
- **问题背景**：CLIP 等多模态编码器在大规模网络数据上训练，嵌入了有害关联，其共享嵌入空间会将不安全提示或图像投影到有害区域，影响下游检索、生成等任务。
- **现有方法不足**：由于伦理与现实限制无法大规模采集真实不安全数据，现有方法用合成数据配对安全样本与生成样本，但将所有生成样本统一标记为"不安全"，忽略了生成内容中模态间的不一致性——文中数据显示 47.7% 的生成对为混合状态（一安一危）， Uniform 重定向会错误修改安全表示。
- **过度安全化风险**：激进的嵌入重定向会破坏语义结构，降低零样本分类与生成质量等实用性能。
- **目标**：在编码器层面对嵌入空间进行安全对齐，使下游系统无需单独修改即可受益。

## 核心贡献（创新点）
1. **ViSUv2 数据集**：构建了 195k 四元组数据集，对每个生成的文本和图像赋予独立的安全标签（0/1），涵盖 578 个细粒度概念与 28 个类别；相比 ViSU 的成对统一标签，支持了模态级条件监督。
2. **选择性安全对齐框架 ShieldCLIP**：根据各模态的实际安全状态条件化保留与重定向目标，而非基于样本来源；与 Safe-CLIP 的本质区别在于不再将所有生成样本视为不安全，而是区分 safe-safe、unsafe-unsafe、mixed 三种情况分别建模。
3. **四路条件训练目标**：设计了保留真实/安全样本的 $\mathcal{L}_{pres}$、重定向不安全样本的 $\mathcal{L}_{redir}$、混合对的跨模态 InfoNCE $\mathcal{L}_{mix}$，以及双不安全对的相干性项 $\mathcal{L}_{coh}$，形成结构化损失；这一条件化设计是本文有别于既有 pair-level 监督的核心。
4. **全面下游评估**：在跨模态检索、Stable Diffusion v1.4/SDXL 文本到图像生成、LLaVA 图像到文本生成三个场景验证，并结合 LlavaGuard 与人类偏好研究，证明安全-效用平衡优势。

## 方法详解
**问题形式化**：设 $\mathcal{T}_0, \mathcal{V}_0$ 为冻结的 CLIP 文本/图像 oracle 编码器，$\mathcal{T}, \mathcal{V}$ 为可训练的 encoder。理想情况下对清洁算子 $c(\cdot)$ 有 $\mathcal{E}(\bar{x}) \approx \mathcal{E}_0(c(\bar{x}))$。

**数据集构成**：$\mathcal{D} = \{(v_i, t_i) \in \mathcal{R}, (\bar{v}_i^{[f_i^v]}, \bar{t}_i^{[f_i^t]}) \in \mathcal{G}\}$，其中 $f_i^v, f_i^t \in \{0,1\}$ 为独立安全标签。数据分布：safe-safe 8.8%，unsafe-unsafe 52.3%，safe-text/unsafe-image 8.4%，unsafe-text/safe-image 30.5%。

**保留目标（Real & Generated-Safe）**：
- 对真实对 $(v,t) \in \mathcal{R}$：$\mathcal{L}_{pres}^{real} = \mathcal{L}_{cos}(\mathcal{V}(v), \mathcal{V}_0(v)) + \mathcal{L}_{cos}(\mathcal{T}(t), \mathcal{T}_0(t)) + \mathcal{L}_{nce}(\mathcal{V}(v), \mathcal{T}_0(t)) + \mathcal{L}_{nce}(\mathcal{T}(t), \mathcal{V}_0(v))$，其中 cosine 项保持单模态保真，InfoNCE 项保持跨模态对齐。
- 对生成安全样本：仅对安全模态应用 cosine 对齐；仅当两模态均为安全时施加 cross-modal InfoNCE。

**重定向与混合处理**：
- $\mathcal{L}_{redir}$：将不安全文本/图像 embedding 拉向其真实安全 counterpart 的 anchor（$\mathcal{L}_{cos}(\mathcal{V}(\bar{v}^{[1]}), \mathcal{V}_0(v))$ 等）。
- $\mathcal{L}_{mix}$：混合对中仅对不安全分支施加 cross-modal InfoNCE，安全分支用 stop-gradient 冻结，避免污染。
- $\mathcal{L}_{coh}$：当两生成模态均不安全时，用 InfoNCE 保持二者间的语义一致性，防止重定向后失去对应关系。

**总体损失**：$\mathcal{L} = \lambda_{real}\mathcal{L}_{pres}^{real} + \lambda_{safe}\mathcal{L}_{pres}^{gen-safe} + \lambda_{redir}\mathcal{L}_{redir} + \lambda_{mix}\mathcal{L}_{mix} + \lambda_{coh}\mathcal{L}_{coh}$，默认权重 $(0.1, 0.1, 0.1, 0.25, 0.25)$。

## 实验与结果
**数据集与基线**：ViSUv2（195k quadruplets）、ViSU、I2P、SneakyPrompt、MMA-Diffusion、Ring-A-Bell；基线包括 Safe-CLIP、SafeR-CLIP、SafetyDPO、DES、SLD-Strong、Embedding Sanitizer 等 generator-side 与 pre-decoding 方法。

**主要数字**：
- **T2I 生成（NudeNet+Q16）**：SD v1.4 在 I2P 上从 38.1% → 3.5%，ViSUv2 从 25.4% → 1.0%；SDXL 在 I2P 上从 36.3% → 4.6%，ViSUv2 从 33.4% → 1.4%，均为最优或并列最优。
- **跨数据集鲁棒性**：在 ViSU/SneakyPrompt/MMA/Ring-A-Bell 四组上 SD v1.4 分别达 1.2%/1.1%/1.3%/0.0%；SDXL 上 1.7%/0.0%/0.7%/3.7%。
- **LLavaGuard 验证**：SD v1.4 I2P 1.8%、ViSUv2 0.3%；SDXL I2P 1.8%、ViSUv2 0.4%，与 NudeNet-Q16 趋势一致。
- **I2T 生成（LLaVA-LLaMA-2-13B）**：NudeNet 源 12.5%（优于 Safe-CLIP 14.6%），NSFW URLs 6.2%，SMID 2.5%。
- **跨模态检索**：ViSUv2 T2I 从 95.2% → 3.5%，I2T 从 95.4% → 20.8%；NudeNet/NSFW URLs/SMID 均 <0.5%。
- **零样本分类**：CIFAR-100 64.0%、Caltech-101 74.8%，接近原始 CLIP。
- **生成质量**：SD v1.4 FID 17.8（CLIP 14.7），CLIP-Sim 0.255（原 0.266）；SDXL FID 17.4，CLIP-Sim 0.258，显著优于 SafetyDPO（FID 20.9/22.3）。

**最强结果**：SDXL + ViSUv2 生成 0.4%（LlavaGuard），ViSUv2 T2I 检索 3.5%，代表当前编码器级安全对齐最优水平。

## 相关工作脉络
- **Safe-CLIP (Poppi et al., 2024)**：同作者的 pair-level 监督方法，将所有生成样本视为不安全；本文扩展为 modality-level 监督，解决混合对的错误标注问题。
- **SafeR-CLIP (Yousaf et al., 2026)**：通过约束不安全概念邻近安全替代来保持 zero-shot 准确率；本文与它在目标上互补——SafeR-CLIP 关注"重定向到哪里"，ShieldCLIP 关注"哪些该被重定向"。
- **HySAC (Poppi et al., 2025)**：将安全/不安全内容组织为双曲层次结构；本文保留 flat 嵌入空间几何，仅在原有空间中做条件化保留/重定向，兼容性更好。
- **DES / Embedding Sanitizer / SafetyDPO**：分别针对文本编码器微调、prompt embedding 净化、DPO 风格对齐；本文作用于 multimodal encoder 层，可同时影响 T2I 与 I2T 任务，且具有跨模型复用性。
- **CoPro / CoProV2 / NSFW-Caps**：提供概念级安全数据但缺乏多模态配对与模态级标签；ViSUv2 在此基础上引入独立 per-modality 标签，支撑选择性对齐。
- **SLD-Strong / ESD / SPM / UCE / SalUn / Receler**：均为 generator-side concept erasure 或 inference-time guidance 方法；本文从上游 encoder 层面统一干预，避免 per-concept 重新训练的扩展性瓶颈。

## 局限性与未来方向
- 安全定义依赖自动化分类器与大模型，可能携带文化/社会偏见，且"安全"是语境依赖的，难以跨域泛化。
- 不保证绝对安全：边缘案例（如不安全语义与场景核心描述强纠缠时）仍存在残留有害输出。
- 依赖 ViSUv2 标签准确性，标注误差或残余偏差会传播到 fine-tuned 嵌入空间。
- 预定义 28 类有害 taxonomy 无法覆盖所有风险形态；方法针对 CLIP-like 架构，对异质 decoder 或不同解码策略的迁移性未充分验证。
- 未来方向：扩展概念多样性、引入 human-in-the-loop 修正、探索动态自适应对齐机制以应对 evolving harm 定义。

## 研究启发与可借鉴点
- **模态级条件监督范式**：将安全标签从样本级粒度下沉到模态级，可有效处理多模态合成数据中常见的不对称有害情况；该思路可迁移至其他多模态对齐/安全任务。
- **冻结 oracle + LoRA 微调**：用原始 CLIP 作为 stop-gradient anchor，仅对轻量 LoRA 参数进行条件化优化，兼顾效用保留与工程效率，是可复用的efficient fine-tuning 套路。
- **双保险评估策略**：同时使用与训练无关的 LlavaGuard 和人类偏好评测，有效排除 classifier-specific artifacts；建议在安全对齐类工作中也采用"训练外评估器 + 人工验证"的对照。
- **mixed pair 的不对称处理**：仅对不安全分支施加 InfoNCE 并对安全分支冻结梯度，这一设计避免了"安全-不安全"冲突下的表示污染，可作为多模态对比学习的通用 trick。
- **安全-效用权衡的量化扫描**：通过 $\lambda$ sweep 联合报告 Harmful Rate / FID / CLIP-Sim，为后续工作提供可复用的 trade-off 曲线分析方法。

## 关键术语表
**Selective Safety Alignment**：根据各模态实际安全状态条件化执行保留或重定向的安全对齐策略，避免对安全内容的误修改。
**ViSUv2**：195k 四元组多模态安全数据集，对生成的文本和图像赋予独立安全标签，含 578 细粒度概念与 28 类别。
**Cross-modal InfoNCE**：用于在跨模态对齐中区分正负对的对比损失，本文用于 safe-safe 锚定与 mixed/unsafe 重定向。
**Oracle Encoder**：冻结的原始 CLIP 编码器，作为保留/重定向目标的锚点表示，训练期间不走梯度。
**Mixed Pair**：生成文本与图像中一者为安全、另一者为不安全的样本对，占生成数据约 38.9%，需单独建模。
**Coherence Loss ($\mathcal{L}_{coh}$)**：对双不安全生成对施加的跨模态 InfoNCE，确保二者在重定向后仍保持语义一致性。
**Safe Zone**：由 real 与 generated-safe 样本共同构成的嵌入区域，通过 $\mathcal{L}_{pres}$ 保持与 oracle 对齐，避免过度清洗。
**Pre-decoding Intervention**：在 diffusion decoder 之前对 conditioning embedding 做净化或重定向的干预方式，区别于 decoding-time concept erasure。

## 可复现要素
- **数据集**：ViSUv2 195k quadruplets；基于 COCO 与 Flickr30k；在 controlled-access protocol 下公开（https://aimagelab.github.io/ShieldCLIP/）。
- **代码与权重**：source code 与 trained models 将开源至上述主页。
- **关键超参**：LoRA rank r=16；Adam lr=1e-4；max 50 epochs，patience=5；λ=(0.1, 0.1, 0.1, 0.25, 0.25)；ViT-L/14 用 8×A100 batch=16，ViT-bigG/14 用 16×A100 batch=8 accum=2。
- **生成器**：NewRealityXL（SDXL variant）与 uncensored FLUX，由 LLaMA-3.1-8B 根据是否含 nude 暗示切换。
- **分类器**：文本用 LLaMA-3.1-8B；图像用 NudeNet+Q16 联合（任一判为 unsafe 即标 unsafe）。
- **评估分类器**：NudeNet+Q16 与 LlavaGuard 双轨验证；零样本用 CIFAR-10/100、SUN-397、Food-101、Caltech-101、Imagenette。
