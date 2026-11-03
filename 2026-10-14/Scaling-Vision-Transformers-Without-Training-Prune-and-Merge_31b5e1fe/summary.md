---
title: "Scaling-Vision-Transformers-Without-Training-Prune-and-Merge"
source: https://arxiv.org/pdf/2610.18039v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:33:21"
---

# 论文速读：Scaling-Vision-Transformers-Without-Training-Prune-and-Merge

## 一句话总结
本文提出 Prune-and-Merge ViT，一种无需训练的即插即用双阶段 Token 压缩框架，通过类注意力筛选与相似度合并的协同作用，在显著降低 ViT/DeiT 推理计算量的同时，保持了更高的分类精度与预测稳定性，并首次引入 RAD/PFR 指标系统化评估压缩过程的幻觉风险。

## 研究问题与动机
- ViT 固定 Token 序列长度导致自注意力计算呈 $O(N^2)$ 增长，推理开销难以满足资源受限场景需求。
- 现有免训练削减方法（仅剪枝或仅合并）在激进压缩下易引发语义丢失或“幻觉”式预测不稳定，缺乏系统性评估。
- 单一策略存在固有局限：纯剪枝会丢弃背景上下文，纯合并受限于二分图匹配无法实现超过 50% 的降采样率。
- 现有工作多仅报告 Top-1 精度，缺少对输入变换下预测一致性与表征忠实度的量化度量。

## 核心贡献（创新点）
- 提出 Prune-and-Merge ViT 双阶段免训练框架，先以 [CLS] 注意力筛选重要 Token 再经余弦相似度二分匹配合并冗余 Token。与已有工作本质区别在于打破剪枝/合并单策略割裂，通过 $K=k_p\cdot k_m$ 乘积结构实现细粒度总预算调控。
- 引入“可控激进（Controlled Aggression）”调度策略，根据总保持率动态调整两阶段负担分配，低 $K$ 时优先合并而非剪枝以维持表征稳定。与已有工作本质区别在于将静态压缩流程升级为自适应强度迁移，有效规避纯剪枝方法的精度断崖。
- 定义 RAD（相对精度下降）与 PFR（预测翻转率）两个鲁棒性度量，量化变换输入下的预测一致性。与已有工作本质区别在于首次将“幻觉”与“语义忠实度”转化为可对比的数值指标，填补稳定性评测空白。
- 在 ViT/DeiT 多尺度（Tiny/Small/Base）上系统验证框架有效性，激进压缩（$K=0.25$）下实现约 3.01× 吞吐量加速且精度显著优于基线。与已有工作本质区别在于证明混合压缩可在突破单策略压缩上限的同时兼顾效率与稳定性。

## 方法详解
- **分层双阶段架构**：在 Transformer 块层 $l \in \{3, 6, 9\}$ 处插入压缩模块，早期层保留高分辨率空间信息，深层逐步缩短序列。总保持率 $K = k_p \cdot k_m$，$k_p$ 控制语义过滤强度，$k_m$ 控制信息融合强度。
- **Stage 1：类注意力剪枝**：提取 [CLS] Token 对所有 Patch Token 的多头注意力 $\mathbf{A}_{\mathrm{cls}}$，经多头平均得重要性分数 $\bar{\mathbf{a}}$；按 $k = \max(1, \lfloor k_p \cdot P \rfloor)$ 选取 Top-K 高重要性 Token。为防背景完全丢失，将其余低注意力 Token 按 $\bar{a}_i$ 加权聚合成全局代表 Token $\mathbf{x}_{\mathrm{fused}} = \sum_{i \in \mathcal{T}_{\mathrm{inatt}}} \bar{a}_i \mathbf{x}_i$，输出序列长度为 $k+1$。
- **Stage
