---
title: "TReVS-Integrating-Textual-Relevance-and-Visual-Saliency-for"
source: https://arxiv.org/pdf/2609.37581v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:25:59"
---

# 论文速读：TReVS-Integrating-Textual-Relevance-and-Visual-Saliency-for

## 一句话总结
提出免训练的TReVS两阶段视觉Token剪枝框架，在LLM前融合视觉显著性与文本相关性以保留查询关键证据，并在LLM浅中层利用高方差注意力头进行二次筛选，在LLaVA-1.5-7B上以94.4%的剪枝率保留92.8%相对性能。

## 研究问题与动机
- 现有两阶段Token剪枝方法在第一阶段（预LLM）通常仅依赖视觉编码器显著性，会不可逆地丢弃与文本查询相关的视觉证据，形成信息瓶颈。
- 预LLM阶段的查询无关剪枝使后续文本引导阶段缺乏必要视觉依据，导致低预算下性能急剧下降。
- LLM内部各注意力头对文本查询的敏感度存在显著差异，但主流方法简单平均所有头的跨模态注意力，未充分利用高敏感头。
- 视觉显著性得分与文本相关性得分的排名呈负相关，说明两者捕捉的是互补的视觉证据，应在首次不可逆削减前协同融合。

## 核心贡献（创新点）
1. 揭示预LLM阶段引入文本相关性的必要性，证明视觉显著性与文本相关性排名负相关且高度互补；与以往仅靠视觉侧信号做预剪枝的方法不同，本文强调在 irreversible reduction 前注入查询感知线索可打破信息瓶颈。
2. 提出基于文本到视觉注意力方差的免训练头筛选准则，识别对查询变化更敏感的高方差头；与直接平均全头注意力权重的做法相比，该准则提供更 discriminative 的跨模态选择信号。
3. 设计TReVS免训练两阶段协调剪枝框架，统一融合[CLS]显著性、文本相关性、多样性补充与高方差头筛选；在LLaVA-1.5-7B上实现94.4%剪枝率下92.8% RelAcc.，全面超越同期SOTA免训练方法。

## 方法详解
- **阶段一：预LLM查询感知冗余削减**
  - 视觉显著性得分 $s_i^v$：取ViT最后一层[CLS] token对各patch的注意力，跨所有头平均（公式2、3）。
  - 文本相关性得分 $s_i^t$：将ViT输出经Projector映射到文本空间后，计算与每个文本token的修正余弦相似度，再取RMS聚合抑制弱相关噪声（公式4、5）。
  - 归一化与温度缩放：分别对 $s_i^v$ 和 $s_i^t$ 使用MAD进行鲁棒归一化，并施加温度 $\tau_v=1.4, \tau_t=1.0$（公式6）。
  - 统一打分与融合：$s_i = \max(\hat{s}_i^v, \hat{s}_i^t) + \lambda\sqrt{\hat{s}_i^v \cdot \hat{s}_i^t}$，第二项为一致性奖励（$\lambda=1.0$），优先保留双信号均高的Token（公式7）。
  - Pivot选择+多样性补充：取Top-$K_r$ 作为pivot集 $S_r$，剩余候选集用FPS选取 $K_d$ 个多样性Token，合并得 $K_1 = K_r + K_d$ 送入LLM（默认 $K_r:K_d=3:1$）。
- **阶段二：LLM内查询驱动Token压缩**
  - 高方差头筛选：在预设层 $l^*$（默认第8层），计算各头文本到视觉注意力矩阵 $\mathbf{A}_{TV}^h$ 沿视觉维度的方差 $u_h$（公式8），取方差最大的前50%头构成 $\mathcal{H}^*$。
  - Token重打分与裁剪：对每个视觉token取其在所有文本位置上的最大跨注意力值，再对 $\mathcal{H}^*$ 内头取平均得 $p_i$（公式9），保留Top-$K_2$ 继续后续层（默认 $K_1:K_2=3:1$）。

## 实验与结果
- **数据集与模型**：LLaVA-1.5-7B（576 tokens）、LLaVA-NeXT-7B（2,880 tokens）、Video-LLaVA-7B（2,048 tokens）；基准涵盖GQA、ScienceQA-IMG、TextVQA、POPE、MME、MMBench（EN/CN）、TGIF-QA、MSVD-QA、MSRVTT-QA。
- **主要结果**：
  - LLaVA-1.5-7B：保留32 token（剪除94.4%）时RelAcc.达92.8%，MME 1651领先；保留128/64 token时RelAcc.分别为98.8%/96.5%，全面超越DivPrune、VScan、DUET-VLM、SparseVLM等。
  - LLaVA-NeXT-7B：高分辨率任务GQA在160/320 token设定下略逊最优0.5%~0.6%，作者归因于高方差头偏向局部细节而损失部分全局关系上下文。
  - Video-LLaVA-7B：保留136 token（剪除93.4%）时RelAcc.达99.0%，超DUET-VLM 3.2%，三个视频基准均为最优。
- **效率**：RTX 4090上32 token设定，Prefill加速2.2×，端到端加速1.4×，KV-cache减少6.5×。
- **消融**：引入文本相关性使RelAcc.提升0.9%（91.8%→92.7%），改用高方差头再提升0.5%（→93.2%），两阶段协同效应明确。

## 相关工作脉络
- **Vision-guided pre-LLM reduction（如VisionZip、VisPruner）**：仅依赖视觉侧显著性/相似度做预剪枝，完全独立于文本查询；本文指出其不可逆丢弃查询相关证据的缺陷，并通过文本相关性注入加以修正。
- **Query-aware in-LLM pruning（如FastV、PyramidDrop、SparseVLM）**：在LLM极浅层直接用全头平均的text-to-vision注意力剪枝，易受early-layer attention shift/dispersion影响；本文证明高方差头更具查询敏感性，并建议推迟至浅中层执行。
- **Two-stage pruning（如DUET-VLM、LearnPruner）**：虽分两阶段，但第一阶段仍独立于查询（LearnPruner需额外训练预测器）；本文强调两阶段均需查询感知，且全程免训练、零参数修改。
- **Diversity-based pruning（如DivPrune）**：仅用特征距离采样多样性Token，缺乏跨模态对齐；本文将FPS作为次要补充机制，主路径由双信号融合打分主导，兼顾任务聚焦与背景覆盖。

## 局限性与未来方向
- 高方差头倾向于聚焦稀疏查询相关细节，在高分辨率全局关系推理任务（如GQA）中可能轻微损失背景上下文，导致小幅性能差距。
- 未来可探索基于输入复杂度（图像/视频分辨率、序列长度）与查询需求，自适应分配预LLM与in-LLM两阶段的Token预算，替代当前固定比例策略。

## 研究启发与可借鉴点
- **互补信号融合结构**：视觉显著性与文本相关性排名负相关，采用 $\max + \text{consistency\_reward}$ 而非简单加权平均，可更好保留异构证据；该结构可迁移至多模态检索、跨模态对齐等任务。
- **免训练头筛选启发**：用注意力分布方差作为查询敏感度代理指标，无需
