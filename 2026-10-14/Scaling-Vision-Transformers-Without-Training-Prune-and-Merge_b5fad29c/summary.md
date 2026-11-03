---
title: "Scaling-Vision-Transformers-Without-Training-Prune-and-Merge"
source: https://arxiv.org/pdf/2610.18039v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 08:51:19"
field: "高效视觉Transformer推理"
keywords: ["Vision Transformer", "token compression", "training-free", "token pruning", "token merging", "model efficiency", "robustness evaluation"]
innovations: ["双阶段无训练token压缩框架（先语义裁剪再相似度合并）", "提出RAD和PFR两个预测稳定性评估指标"]
benchmarks: ["ImageNet-100"]
---

# 论文速读：Scaling-Vision-Transformers-Without-Training-Prune-and-Merge

## 一句话总结
论文提出 **Prune-and-Merge ViT**，一种即插即用、无需训练的双阶段 token 压缩框架：先用 [CLS] token 注意力裁剪背景 token，再通过余弦相似度 + 二分图匹配合并冗余 token；同时引入 **RAD**（相对准确率下降）和 **PFR**（预测翻转率）两个鲁棒性指标。在 ImageNet-100 上，该方法在激进压缩（K=0.25）下对 ViT-Base 实现约 **3.01× 吞吐加速**（236.85 imgs/sec），同时显著优于单一裁剪或合并基线。

## 研究问题与动机
- ViT 的计算复杂度随 token 数呈 $O(N^2)$ 增长，推理成本高；现有无训练 token 压缩方法在极端压缩下容易出现精度"断崖式"下降。
- 单纯裁剪（如 EViT、Top-K）会丢弃背景语义，导致输入变换下预测不稳定；单纯合并（如 ToMe）在高压缩率下受限于二分图匹配上限（最多保留 50% token），且无法去除噪声。
- 混合策略的潜在收益尚未被系统研究：如何在保持语义完整性的同时精细化控制 token 预算？
- 缺乏统一的鲁棒性度量：现有工作只报告 top-1 精度，未评估模型在语义保留变换（旋转、颜色抖动、水平翻转）下的预测一致性。

## 核心贡献（创新点）
1. **提出 Prune-and-Merge ViT 双阶段框架**：Stage 1 基于 [CLS] 注意力的 top-k 裁剪负责语义过滤，Stage 2 基于余弦相似度的 bipartite merging 负责冗余聚合；与 EViT/Top-K（仅裁剪）或 ToMe（仅合并）的本质区别在于"先语义筛选再信息融合"的互补设计。
2. **引入 RAD 与 PFR 两个鲁棒性指标**：RAD 衡量框架间在变换下的相对精度退化，PFR 量化单样本预测翻转概率；这是首次系统地将"预测稳定性"纳入 token 压缩评估体系。
3. **提出 "Controlled Aggression" 策略**：在不同压缩强度下动态平衡两阶段负担——低保持率时优先让 Stage 2 承担合并（保留关键语义），高保持率时 Stage 1 负责去噪；相比固定策略的基线，避免了纯裁剪在 K=0.25 时的精度断崖。
4. **在 ViT-Base 激进压缩下实现 3× 吞吐提升**：K=0.25 时达到 236.85 imgs/sec（相对基线 78.61 提升 3.01×），且 top-1 精度（71.16%）显著优于 EViT（61.00%）和 Top-K（58.72%）。

## 方法详解
**整体架构**：在预训练 ViT/DeiT 的层 $l \in \{3, 6, 9\}$ 处插入双阶段压缩模块（位于 attention 之后、MLP 之前），采用分层递减策略——浅层保留高分辨率空间信息，深层逐步压缩序列长度。总保持率 $K = k_p \cdot k_m$，其中 $k_p$ 为裁剪保持率，$k_m$ 为合并保持率。

**Stage 1：基于 [CLS] 注意力的裁剪**
- 计算多头 self-attention 后，提取 [CLS] token 对 patch token 的注意力图：$\bar{\mathbf{a}} = \frac{1}{H}\sum_{h=1}^{H}\mathbf{A}_{\text{cls}}^{(h)} \in \mathbb{R}^{B \times P}$。
- 按 $\bar{\mathbf{a}}$ 取 top-k（$k = \max(1, \lfloor k_p \cdot P \rfloor)$）保留重要 token $\mathcal{T}_{\text{att}}$。
- 对未选中的背景 token $\mathcal{T}_{\text{inatt}}$ 做注意力权重平均，聚合为一个代表 token $\mathbf{x}_{\text{fused}} = \sum_{i \in \mathcal{T}_{\text{inatt}}} \bar{a}_i \mathbf{x}_i$，防止全局上下文完全丢失。
- 输出序列 $\mathbf{x}_{\text{stage1}} = [\mathbf{x}_{\mathcal{T}_{\text{att}}}, \mathbf{x}_{\text{fused}}]$，长度 $N_p = k+1$。

**Stage 2：基于余弦相似度的 token 合并**
- 将 Stage 1 输出的 $N_p$ 个 token 分为两组 A、B（近似等分），对 key 向量计算余弦相似度，做二分图匹配，每条边连接 A 中一个 token 与其在 B 中最相似的 token。
- 合并次数 $r = \lfloor (1 - k_m) \cdot N_p \rfloor$，保留最相似的 r 条边执行合并。
- 合并采用 token-size 加权平均：$\mathbf{x}_{\text{merged}} = \frac{s_i \mathbf{x}_i + s_j \mathbf{x}_j}{s_i + s_j}$，其中 $s_i$ 记录该 token 代表的原始 patch 数。
- 引入 **proportional attention** 修正 softmax：$\mathbf{A} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \log \mathbf{s}\right)$，确保合并后的 token 在后续层中对注意力分布的贡献与原始 patch 数成正比。

## 实验与结果
- **数据集**：ImageNet-100（ImageNet-1K 的子集，100 类），使用完整 ImageNet-1K 预训练权重。
- **模型与规模**：ViT 和 DeiT 的 Tiny / Small / Base 三档，patch 大小 16×16，分辨率 224×224，FP32 精度推理。
- **基线**：Raw（无压缩）、Random Drop、Top-K、EViT、ToMe。
- **评测维度**：top-1 精度、GFLOPs、延迟（ms/img）、吞吐（imgs/sec）、RAD、∆PFR。
- **核心结果**：
  - **ViT-Base K=0.25**：Hybrid top-1 = **71.16%**，显著高于 EViT（61.00%）和 Top-K（58.72%）；吞吐 **236.85 imgs/sec**（3.01× 加速）；延迟 4.222 ms/img。
  - **DeiT-Base K=0.25**：Hybrid = 47.88% vs EViT 51.32%（此处 Hybrid 略低，但整体趋势在 K=0.5/0.7/0.9 均占优）。
  - **ViT-Tiny K=0.7**：吞吐达 **1073.18 imgs/sec**，适合边缘部署。
  - **鲁棒性**：∆PFR 在 K=0.5 时 ViT-Base 仅 0.0422（Rotation），接近原始 backbone（0.0022 量级）；RAD 在多数配置下接近 0 或为负，表明幻觉率低于 ToMe/EViT。
  - **Controlled Aggression 有效性**：K=0.25 时 Merging-heavy 策略（Hybrid 变体）在 ViT-Tiny 上达到 71.42%，明显优于 Balanced（60.12%）和 Pruning-heavy（60.06%），验证了激进压缩下"重合并、轻裁剪"的优越性。

## 相关工作脉络
- **EViT / Top-K**：基于 [CLS] 注意力排名裁剪的无训练单策略方法，仅做"去"不做"合"，在 K≤0.5 时精度断崖；本文在其基础上加入合并阶段弥补语义损失。
- **ToMe**：通过 bipartite matching 合并相似 token 的物理缩减方法，但独立使用最多只能保留 50% token，且无法主动去噪；本文将其作为 Stage 2，前置 Stage 1 裁剪降低噪声干扰。
- **DynamicViT**：学习型动态 token 稀疏化方法，需要额外训练剪枝模块；本文方法是 training-free 即插即用，无需微调预训练权重。
- **V-Pruner**：全局感知的裁剪框架，但仅做剪枝；本文与之对比展示混合策略在激进压缩下的稳定性优势。
- **VisionTrim**：面向硬件感知延迟优化的 token 压缩方法，侧重吞吐量而非预测一致性；本文补充了鲁棒性（RAD/PFR）维度的评估。
- **ToCom**：预训练的 plug-and-play 压缩模块，可跨压缩率迁移；本文与其定位不同——不需要额外训练模块，直接利用原模型注意力进行两阶段压缩。

## 局限性与未来方向
- 实验仅在 ImageNet-100 上进行，未验证在大规模 ImageNet-1K 或其他视觉基准（COCO、ADE20K）上的泛化性。
- 仅评估了三种标准语义保留变换（Rotation、Color Jitter、Horizontal Flip），未涉及遮挡、模糊等更具挑战性的扰动。
- 双阶段串行流程引入了额外计算开销（尤其在高层），在 K=0.9 等高保持率下延迟略高于部分单策略基线。
- 未探索自适应 $k_p / k_m$ 分配策略（当前实验为固定比例），缺乏针对输入内容动态调节的机制。
- 代码仓库未在论文中明确开源声明（仅提到 PyTorch + timm），可复现性存疑。

## 研究启发与可借鉴点
1. **"先筛后合"的双阶段范式可迁移**：将 token 压缩拆解为"语义过滤 → 冗余聚合"两步，对 Video Transformer、Vision-Language Model 等多序列场景具有直接参考价值。
2. **RAD/PFR 指标体系值得沿用**：未来做任何 token 压缩工作时，可将这两个指标作为标配，填补仅报告精度时的鲁棒性空白。
3. **Controlled Aggression 思想可泛化**：根据压缩强度动态调整各阶段"分工权重"（如本表 Table 2 所示 Merging-heavy 在 K=0.25 更优），这一思路可推广至多阶段推理优化（如 skip layer、early exit）。
4. **Proportional attention 的数学修正技巧**：$\log \mathbf{s}$ 修正项保证 softmax 分布一致性，这一形式可在任意 token 合并/聚合操作中复用，避免精度损失。
5. **与团队方向的结合机会**：若团队研究低资源机器翻译或长序列建模，可将此 training-free 压缩框架适配至 Transformer 的 sequence dimension，配合自研的 RAD/PFR 风格稳定性指标评估生成任务中的忠实性。

## 关键术语表
**Prune-and-Merge ViT**：本文提出的双阶段无训练 token 压缩框架，Stage 1 裁剪背景 token，Stage 2 合并相似 token。

**[CLS] Token**：ViT 中附加在序列开头的可学习分类标记，其注意力分数被用来衡量各 patch token 对分类任务的语义重要性。

**Proportional Attention**：在 token 合并后，在 QK^T 之上叠加 $\log \mathbf{s}$ 偏置，使合并 token 的注意力权重与其代表的原始 patch 数成正比，维持 softmax 分布一致性。

**RAD（Relative Accuracy Drop）**：衡量两种框架 X、Y 在语义变换 T 下 top-1 精度比值的差异，RAD≈0 表示与基线同等忠实度，正值为更高幻觉率。

**PFR（Prediction Flip Rate）**：原始输入与变换输入预测结果不一致的样本比例，∆PFR 越小表示模型对扰动越鲁棒。

**Controlled Aggression**：根据总保持率 K 动态调整两阶段负担（低 K 时重合并、高 K 时重裁剪），以避免纯裁剪基线的精度断崖。

**Bipartite Matching（二分图匹配）**：将 token 集合划分为 A、B 两组后，按余弦相似度建立最大权匹配边，每条边决定一次 token 合并操作。

**Composite Keep Rate（复合保持率）**：$K = k_p \cdot k_m$，控制总体 token 缩减强度，乘积形式保证两阶段压缩比例相互独立可调节。

## 可复现要素
- **数据集**：ImageNet-100（ImageNet-1K 子集），公开可用。
- **代码**：论文依赖 PyTorch + timm 库；未明确提供 GitHub 仓库链接（"论文未提及开源代码"）。
- **权重**：使用官方 pretrained ViT / DeiT（Tiny/Small/Base）在 ImageNet-1K 上的预训练权重，公开可用。
- **关键超参**：插入层 $l \in \{3, 6, 9\}$；复合保持率 $K \in \{0.25, 0.5, 0.7, 0.9\}$；Stage 1 裁剪保持率 $k_p$、Stage 2 合并保持率 $k_m$（满足 $K = k_p \cdot k_m$）；相似度度量 = cosine similarity on token embeddings。
- **硬件**：Google Colab T4 GPU（16 GB VRAM），FP32 推理。
- **环境**：PyTorch，timm 库（具体版本论文未声明）。
