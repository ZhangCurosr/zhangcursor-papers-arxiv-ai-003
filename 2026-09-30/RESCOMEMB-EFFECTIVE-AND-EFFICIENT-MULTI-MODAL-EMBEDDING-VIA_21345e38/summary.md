---
title: "RESCOMEMB-EFFECTIVE-AND-EFFICIENT-MULTI-MODAL-EMBEDDING-VIA"
source: https://arxiv.org/pdf/2609.37225v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:18:04"
field: "多模态表征学习"
keywords: ["多模态 embedding", "多向量检索", "视觉 token 压缩", "晚期交互匹配", "Matryoshka 表示学习"]
innovations: ["残差同质性压缩（RHC）显式分离粒度内冗余与粒度间重复", "基于 MRL 的嵌套监督使任意前缀均保持语义有效性", "长度自适应双向晚期交互匹配提升检索鲁棒性"]
benchmarks: ["MMEB", "ViDoRe V1", "ViDoRe V2"]
---

# 论文速读：RESCOMEMB: EFFECTIVE AND EFFICIENT MULTI-MODAL EMBEDDING VIA RESIDUAL HOMOGENEITY COMPRESSION

## 一句话总结
论文提出 **RESCOMEMB**，一种基于残差同质性压缩（RHC）的可训练多向量多模态 embedding 框架，通过粗到细的层级压缩与嵌套监督，在仅使用 37.5% visual token 预算的情况下，于 MMEB 和 ViDoRe 基准上超越现有最强方法。

---

## 研究问题与动机

- **单向量 vs 多向量的权衡困境**：现有 MLLM-based embedding 要么将视觉输入压缩为单个向量（表达力有限），要么保留全部视觉 token（存储与交互成本高）。
- **已有压缩方法的缺陷**：Token pruning/merging 牺牲全局上下文；learnable-token 方法容量固定，难以保留输入特定的细节；MURE 的非可训练聚类缺乏 loss 反馈，可能丢失关键表示。
- **冗余结构未被显式建模**：视觉 token 存在两类冗余——粒度内冗余（同一分辨率下相似 token）和粒度间重复（更细粒度包含已捕获的信息），现有方法未分别处理。
- **缺乏可微的端到端压缩训练**：压缩应与表征联合优化，而非作为独立的后处理步骤。

---

## 核心贡献（创新点）

1. **残差同质性压缩（RHC）模块**：在 MLLM 上下文化之后、嵌入投影完成的前提下，显式分离并压缩粒度内冗余与粒度间重复，满足给定的 visual token 预算。
2. **基于 Matryoshka 的嵌套监督**：将多层压缩输出拼成前缀嵌套结构，每一级都接受对比损失训练，使任意截断长度均保持语义有效性。
3. **长度自适应的双向晚期交互匹配**：融合查询→文档和文档→查询两个方向的 Top-K 最大相似度均值，并按有效 token 数动态加权，缓解弱匹配累积问题。
4. **显著的 effectiveness–efficiency 优势**：在 MMEB 以 67.4 Precision@1 超越 VLM2Vec-V2 2.5 分；在 ViDoRe V1/V2 以仅 37.5% 的 visual token 预算（384 tokens vs ColQwen2.5 的 1024 tokens）实现更高 NDCG@5。

---

## 方法详解

### 整体架构
RESCOMEMB 以 Qwen2.5-VL 为共享 MLLM backbone，对每个视觉输入构建三级粒度视图：全局（1×1）、中间（1×2 或 2×1）、细粒度（2×2），经 MLLM 上下文编码和线性投影后，送入 RHC 模块压缩。

### 1. 多级粒度视觉编码（Multi-Granularity Visual Encoding）
- 将同一图像切分为 $c_1=1, c_2=2, c_3=4$ 个 crop，分别用共享 MLLM 编码，得到视觉 token 序列 $\mathbf{g}_1, \mathbf{g}_2, \mathbf{g}_3$。
- 文本部分统一保留为 $\mathbf{T}$，不压缩。

### 2. 残差同质性压缩（RHC）
每个阶段 $k$ 包含两个子模块：

**Intra-Resolution Token Assessor（粒度内评估器）**
- 通过 MLP 将投影后的 token $\mathbf{g}_k$ 转为上下文化表示 $\mathbf{h}$。
- **重要性得分**：$S_i = \mathrm{MinMax}(f_{\mathrm{imp}}(\mathbf{h}_i))$，MinMin 归一化到 $[0,1]$。
- **粒度内相似度**：将交替位置分为 source（奇数位）和 destination（偶数位），计算余弦相似度矩阵 $\mathcal{C}_{ij}$。

**Inter-Resolution Token Compressor（粒度间压缩器）**
- **新颖度得分**：对 $k>1$，$\mathcal{N}_i = \mathrm{MinMax}(1 - \max_{\mathbf{a} \in \mathbf{A}_k} \mathrm{cos}(\mathbf{h}_i, \mathbf{a}))$，衡量当前 token 未被粗粒度 anchor 集合 $\mathbf{A}_k$ 覆盖的程度；$k=1$ 时 $\mathcal{N}_i=1$。
- **保留优先级**：$\mathcal{P}_i = S_i + \eta \mathcal{N}_i$。
- **合并得分**：$\mathcal{U}_{ij} = \mathcal{C}_{ij} - \alpha \mathcal{P}_i$，高分表示高相似且低优先级（易被合并）。
- 每个 source token 指派给最高 $\mathcal{U}_{ij}$ 的 destination，直到输出达到阶段预算 $B_k$；聚合后 $\ell_2$ 归一化得 $\mathbf{r}_k$。

### 3. 嵌套表示（Nested Representation）
- 将各阶段压缩结果按顺序拼接：$\mathbf{E}_{x,k} = \mathrm{Concat}(\mathbf{T}_x, \mathbf{r}_{x,1}, \ldots, \mathbf{r}_{x,k})$。
- 形成 Matryoshka 前缀：$\mathbf{E}^{(1)} \subset \mathbf{E}^{(2)} \subset \mathbf{E}^{(3)}$，支持弹性推理。

### 4. 双向晚期交互匹配（Bidirectional Late-Interaction Matching）
- 对 query $q$ 和 document $d$，取双方 Top-$K$ 最大相似度 token 子集 $\Omega_q, \Omega_d$。
- 双向 Top-K 均值：
  $$S_{QT}^{(K)} = \frac{1}{K_q}\sum_{i \in \Omega_q}\max_j (\mathbf{e}_q^i)^\top \mathbf{e}_d^j, \quad S_{TQ}^{(K)} = \frac{1}{K_d}\sum_{j \in \Omega_d}\max_i (\mathbf{e}_q^i)^\top \mathbf{e}_d^j$$
- 最终得分：$S_{\mathrm{final}} = w_1 S_{QT}^{(K)} + w_2 S_{TQ}^{(K)}$，其中 $w_1$ 随文档长度增大而增大，避免长文档主导。

### 5. 训练目标
- 对每一活跃层级 $k$ 计算对比损失：
  $$\mathcal{L}_k = -\frac{1}{|\mathcal{I}_k|}\sum_{i \in \mathcal{I}_k}\log\frac{\exp(s_{ii}^{(k)}/\tau)}{\sum_{d_j \in \mathcal{C}_{i,k}}\exp(s_{ij}^{(k)}/\tau)}$$
- 总损失：$\mathcal{L}_{\mathrm{train}} = \sum_{k} \lambda_k \mathcal{L}_k$，所有嵌套前缀均受监督。

---

## 实验与结果

| 数据集 | 指标 | RESCOMEMB | 最强对比 | 提升幅度 |
|---|---|---|---|---|
| **MMEB** | Precision@1 (Avg) | **67.4** | VLM2Vec-V2 (64.9) | **+2.5** |
| **ViDoRe V1** | NDCG@5 (Avg) | **90.4** | ColMate (89.9) / ColQwen2.5 (89.4) | **+0.5 / +1.0** |
| **ViDoRe V2** | NDCG@5 (Avg) | **61.6** | ColMate (60.5) / ColQwen2.5 (60.6) | **+1.1 / +1.0** |

- **Token 效率**：RESCOMEMB 仅用 384 个 visual token（128/128/128），为 ColQwen2.5（1024 tokens）的 **37.5%**，ViDoRe 平均仍高出 1.0 分。
- **MRL 消融**：去掉嵌套监督，ViDoRe V2 下降 5.7–7.6 分，MMEB 下降 2.8–3.0 分，证明嵌套训练对多预算有效性至关重要。
- **RHC 信号消融**：去掉 novelty score 下降更多（V2 降 3.5 分），说明跨粒度去重复贡献更大。
- **预算分析**：48→384 tokens 提升显著（V1 +3.3，V2 +5.1）；384→768 边际收益趋零且 MMEB 反而下降，384 为最优折中。

---

## 相关工作脉络

- **CLIP / SigLIP**：双塔对比学习，生成全局单向量，无法保留局部细粒度证据。
- **VLM2Vec / VLM2Vec-V2**：将 MLLM 用于通用多模态 embedding，但采用单向量输出，丢失局部信息。
- **ColPali / ColQwen2.5**：将 late interaction 扩展到 visual patch，保留全量 token，存储与检索成本高昂。
- **MURE**（Zhu et al., 2026）：结合多分辨率采样与 token 聚类，但聚类不可训练、无 loss 反馈，无法区分粒度内/粒度间冗余。
- **MetaEmbed**：用可学习 abstract tokens 聚合，紧预算下可能丢失输入特定的局部证据。
- **VisionZip / TokenPacker**：上游压缩，在 MLLM 内部减少 token 数，不同于本文的后上下文化压缩策略。

---

## 局限性与未来方向

- **粒度数量固定**：当前仅使用 3 级粒度（1/2/4 crop），未探索更细或自适应粒度划分。
- **静态预算分配**：每阶段固定 128 tokens，未考虑输入复杂度自适应分配。
- **仅评估文档检索与通用任务**：未验证在视频、3D 等更长序列上的表现。
- **论文自述方向**：任务自适应冗余建模（task-adaptive redundancy modeling），将可训练压缩扩展到其他表征生成场景。

---

## 研究启发与可借鉴点

1. **可训练压缩 + 嵌套监督的组合设计**：将压缩作为可微分模块并与 Matryoshka 学习结合，使任意前缀均具语义效力，值得迁移到其他多向量表示任务。
2. **显式分解两类冗余**：粒度内冗余（via similarity）与粒度间重复（via novelty）分别建模，思路清晰且 ablation 充分，可作为后续工作的一般性分析框架。
3. **长度自适应双向晚期交互**：Top-K + 均值聚合 + 动态权重，在长文档检索场景下优于简单 MaxSim，可直接复用。
4. **梯度流的可微分析**：Appendix B 证明了离散 assignment 在值聚合路径上的可微性，为类似 hard merging 模块提供了理论支撑。

---

## 关键术语表

- **Residual Homogeneity Compression (RHC)**：论文核心模块，通过重要性打分与新颖度打分联合决策，分别消除粒度内冗余与粒度间重复的可训练压缩机制。
- **Matryoshka Representation Learning (MRL)**：嵌套表示学习，使粗粒度前缀包含细粒度信息，支持多预算弹性推理。
- **Bidirectional Late-Interaction Matching**：双向晚期交互匹配，分别从查询→文档和文档→查询方向取 Top-K 最大相似度均值并加权融合。
- **Multi-Granularity Visual Encoding**：多级粒度视觉编码，将同一图像以 1×1、1×2/2×1、2×2 三档 crop 并行输入共享 MLLM。
- **Novelty Score**：新颖度得分，衡量当前 token 与已累积粗粒度 anchor 的余弦距离，用于跨粒度去重。
- **Preservation Priority**：保留优先级 $\mathcal{P}_i = S_i + \eta \mathcal{N}_i$，融合重要性与新穎度，决定 token 是否易被合并。
- **NDCG@5 / Precision@1**：ViDoRe 用 NDCG@5，MMEB 用 Precision@1 作为主要评估指标。

---

## 可复现要素

- **数据集**：MMEB（https://huggingface.co/datasets/TIGER-Lab/MMEB-train）、ViDoRe V1/V2（https://huggingface.co/collections/vidore），均为公开数据集。
- **代码**：论文未提供开源代码链接，但列出了完整 Hugging Face 数据集地址。
- **权重初始化**：ColQwen2.5-base（https://huggingface.co/vidore/colqwen2.5-base）。
- **关键超参**：LoRA rank=32，scale=32，dropout=0.1；learning rate $10^{-4}$ 线性衰减；contrastive temperature $\tau=0.03$；每阶段 RHC budget=128/128/128（共 384 visual tokens）；batch size=64，8×A100，30k steps。

---
