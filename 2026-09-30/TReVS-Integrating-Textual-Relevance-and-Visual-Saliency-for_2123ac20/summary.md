---
title: "TReVS-Integrating-Textual-Relevance-and-Visual-Saliency-for"
source: https://arxiv.org/pdf/2609.37581v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:25:49"
---

# 论文速读：TReVS-Integrating-Textual-Relevance-and-Visual-Saliency-for

## 一句话总结
提出TReVS，一种免训练的视觉-语言模型双阶段视觉Token剪枝框架，通过在LLM输入前融合文本相关性与视觉显著性保留查询关键证据，并在浅中层利用高方差注意力头筛选任务相关Token，在大幅压缩计算开销的同时保持高性能。

## 研究问题与动机
- 现有双阶段Token剪枝在Pre-LLM阶段仅依赖视觉编码器显著性（如[CLS]注意力），属于无查询感知（query-agnostic）操作，可能不可逆地丢弃对当前文本查询至关重要的视觉证据。
- Pre-LLM阶段的盲目削减形成信息瓶颈，导致后续In-LLM阶段必须在残缺的视觉上下文中进行查询感知剪枝，引发性能骤降。
- 现有In-LLM剪枝方法通常对所有text-to-vision注意力头取平均，忽略了不同头对查询语义变化的敏感度差异。
- 视觉显著性与文本相关性在Token排序上呈负相关，两者捕获的视觉证据具有强互补性，但未被有效整合到统一的免训练剪枝流程中。

## 核心贡献（创新点）
1. **揭示Pre-LLM阶段引入文本相关性的必要性**：通过相关性补偿显著性，保留被后者忽略的查询关键证据；与纯视觉剪枝的本质区别在于将查询语义前置，打破“先裁后查”的信息瓶颈。
2. **提出基于注意力方差的高敏感度头筛选准则**：高方差的text-to-vision头对查询变化更敏感，能提供更判别性的Token评分；区别于平均所有头或依赖可学习头部重要性网络的做法，方差仅需前向计算即可提取。
3. **设计免训练双阶段协调剪枝框架TReVS**：Stage 1融合显著性、相关性（含一致性奖励）与多样性（FPS）完成Pre-LLM粗剪；Stage 2在浅中层利用高方差头完成In-LLM精剪；与LearnPruner/DUET-VLM等依赖可学习模块或固定层剪枝的方法本质不同，全链路由显式语义信号驱动且无需微调。
4. **系统验证跨分辨率/跨模态的泛化与效率收益**：在LLaVA-1.5/NeXT-7B与Video-LLaVA-7B上均以94.4%压缩率保留92.8%性能；相比前作不仅提升绝对精度，更在极端压缩下显著改善幻觉鲁棒性（POPE）。

## 方法详解
**Stage 1：Pre-LLM Query-Aware 视觉冗余削减**
- *视觉显著性得分*：取ViT末层[CLS]到各Patch token的跨头注意力均值 $s_i^v = \frac{1}{H}\sum_{h=1}^H A_{[\mathrm{CLS}]}^h[i]$。
- *文本相关性得分*：将ViT输出经Projector映射至文本对齐空间得 $z_i$，计算与文本token嵌入 $\bar{t}_j$ 的截断余弦相似度 $m_{j,i}=\max(\bar{t}_j^\top \bar{z}_i, 0)$，再对M个文本token取RMS聚合得 $s_i^t = \sqrt{\frac{1}{M}\sum_{j=1}^M m_{j,i}^2}$，压制弱相关噪声并放大被广泛查询关注的内容。
- *归一化与温度缩放*：对 $s_i^v, s_i^t$ 分别用MAD鲁棒标准化并加ReLU截断，再除以温度 $\tau_v=1.4, \tau_t=1.0$。
- *统一打分与一致性奖励*：$s_i = \max(\hat{s}_i^v, \hat{s}_i^t) + \lambda\sqrt{\hat{s}_i^v \hat{s}_i^t}$（$\lambda=1.0$），兼顾“显著或相关”与“两者兼具”的Token，选取Top-$K_r$ 作为Pivot集合 $S_r$。
- *多样性补充*：在剩余候选集 $\mathcal{C}$ 上用Farthest Point Sampling (FPS) 采样 $K_d$ 个特征最分散的Token构成 $S_d$，最终Pre-LLM保留 $K_1=K_r+K_d$ 个Token送入LLM。

**Stage 2：In-LLM Query-Driven 视觉Token压缩**
- *高方差头筛选*：在选定层 $l^*$（默认第8层），计算每个头 $h$ 的text-to-vision注意力矩阵方差 $u_h = \frac{1}{L_\mathrm{text}}\sum_q \mathrm{Var}_i(A_{TV}^h[q,i])$，保留方差Top-50%的头构成 $\mathcal{H}^*$。
- *Token重评分与剪枝*：对Stage 1保留的 $K_1$ 个Token按 $\mathcal{H}^*$ 的最大文本注意力均值重打分 $p_i = \max_q \frac{1}{|\mathcal{H}^*|}\sum_{h \in \mathcal{H}^*} A_{TV}^h[q,i]$，最终保留Top-$K_2$ 个Token（默认 $K_1:K_2=3:1$）参与后续推理。

## 实验与结果
- **数据集与模型**：LLaVA-1.5-7B（576 tokens）、LLaVA-NeXT-7B（2880 tokens）、Video-LLaVA-7B（2048 tokens）；基准涵盖GQA、ScienceQA、TextVQA、POPE、MME、MMBench（EN/CN）及视频基准TGIF-QA、MSVD-QA、MSRVTT-QA。
- **基线**：FastV、SparseVLM、DivPrune、VisionZip、VScan、DUET-VLM 等免训练/轻量剪枝方法。
- **核心结果（LLaVA-1.5-7B）**：在保留128/64/32 Token时，TReVS的RelAcc.分别达98.8%/96.5%/92.8%，全面领先。32 Token下POPE准确率82.7%，较DivPrune/VScan分别提升1.2%/2.8%，极端压缩下幻觉鲁棒性最优。
- **高分辨率与视频扩展**：LLaVA-NeXT-7B在320/160 Token下RelAcc.达96.6%/93.0%
