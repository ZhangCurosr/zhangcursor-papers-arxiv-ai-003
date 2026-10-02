---
title: "RESCOMEMB-EFFECTIVE-AND-EFFICIENT-MULTI-MODAL-EMBEDDING-VIA"
source: https://arxiv.org/pdf/2609.37225v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:18:40"
---

# 论文速读：RESCOMEMB: EFFECTIVE AND EFFICIENT MULTI-MODAL EMBEDDING VIA RESIDUAL HOMOGENEITY COMPRESSION

## 一句话总结
提出 RESCOMEMB，一种基于残差齐性压缩（RHC）的可训练多粒度多模态嵌入框架，在显式视觉 token 预算下联合消除粒度内冗余与跨粒度重复，并以仅 37.5% 的视觉 token 预算在 MMEB、ViDoRe V1/V2 上均取得最优或超越全量基线的效果。

## 研究问题与动机
- **保真度与效率的权衡困境**：现有 MLLM 驱动的嵌入模型要么将视觉输入压缩为单向量（损失细粒度局部证据），要么保留完整长序列，带来高昂存储与两两交互成本。
- **现有压缩方法的缺陷**：Token 剪枝/合并易破坏整体上下文；可学习槽位方法在紧预算下容量固定、难以捕捉输入特定细节；MURE 等依赖非训练后处理聚类，压缩标准缺乏损失反馈，易丢失相关表征或保留无关内容。
- **跨粒度冗余与粒度内重复未解耦**：多分辨率方法通常独立处理各粒度，未能显式区分并联合优化“粒度内相似 token 重复”与“精细粒度对粗粒度信息的重复覆盖”，导致压缩效率受限。

## 核心贡献（创新点）
1. **提出残差齐性压缩（RHC）模块**：通过粒度内重要性评估与粒度间新颖度评估双信号，在显式预算下联合消除重复与冗余；与 MURE 等静态/非训练压缩方法的本质区别在于 RHC 实现了压缩操作与检索损失的端到端联合优化。
2. **设计 Matryoshka 嵌套监督机制**：将粗到细的压缩序列拼接为前缀嵌套表示，使每一级紧凑表征均受对比损失直接监督，突破固定容量限制，支持按预算弹性推理。
3. **开发长度自适应双向晚期交互匹配**：分别取查询→文档与文档→查询两个方向的最强 Top-K 匹配均值，并按有效 token 数量动态加权融合；与单向 Late Interaction 基线相比，有效缓解长短输入不对称导致的弱匹配累积问题。
4. **在极低 token 预算下实现 SOTA**：仅使用 384 个视觉 token（为全量 ColQwen2.5 的 37.5%）即在 ViDoRe V1/V2 上分别超越全量基线 1.0 分，并在通用 MMEB 基准上以 67.4 的 Precision@1 超越 VLM2Vec-V2 2.5 分。

## 方法详解
- **多粒度视觉编码**：共享 MLLM（Qwen2.5-VL）对同一输入生成全局（1×1）、中间（1×2/2×1）、细粒度（2×2）三组裁剪视图，经最后层隐藏状态投影后分离为文本 token 序列 $\mathbf{T}$ 与三组视觉序列 $\mathbf{g}_1, \mathbf{g}_2, \mathbf{g}_3$。
- **残差齐性压缩（RHC）**：
  - *粒度内评估器*：MLP 上下文编码生成特征 $\mathbf{h}$，重要性 $S_i = \mathrm{MinMM}(f_{\mathrm{imp}}(\mathbf{h}_i))$；将交替位置分作源/目标对，余弦相似度 $\mathcal{C}_{ij}$ 估计粒度内冗余。
  - *粒度间压缩器*：累积粗粒度输出构成锚点集 $\mathbf{A}_k$，新颖度 $\mathcal{N}_i = \mathrm{MinMM}(1 - \max_{\mathbf{a} \in \mathbf{A}_k} \mathrm{cos}(\mathbf{h}_i, \mathbf{a}))$（$k=1$ 时为 1）。保留优先级 $\mathcal{P}_i = S_i + \eta \mathcal{N}_i$，合并得分 $\mathcal{U}_{ij} = \mathcal{C}_{ij} - \alpha \mathcal{P}_i$。源 token 按最高得分分配至目标，按阶段预算截断后聚合并 $\ell_2$ 归一化得 $\mathbf{r}_k$。
- **嵌套表示与 MRL 监督**：按 $\mathbf{E}_{x,k} = \mathrm{Concat}(\mathbf{T}_x, \mathbf{r}_{x,1}, \ldots, \mathbf{r}_{x,k})$ 组装前缀，各级前缀共享文本且视觉部分递增扩展。训练时对所有活跃层级
