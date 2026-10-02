---
title: "SELECTING-THE-MOST-INFORMATIVE-TOKENS-INNATURAL-LANGUAGE-AUT"
source: https://arxiv.org/pdf/2609.37040v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:20:29"
---

# 论文速读：SELECTING THE MOST INFORMATIVE TOKENS IN NATURAL LANGUAGE AUTOENCODERS

## 一句话总结
本文针对自然语言自动编码器（NLA）对长 transcript 逐一生成解释成本过高的问题，系统评估了 13 种基于模型计算的轻量信号与仅依赖聊天结构的排名器，证明在多数审计任务中，利用 chat 元信息即可在 5% 的位置预算下保留近 96% 的威胁检测成功率，且预训练 verbalizer 无需重训即可恢复微调后模型的隐蔽词。

## 研究问题与动机
1. **NLA 解释成本过高**：NLA verbalizer 平均为每个 token 位置生成约 130 个 token 的自回归解释，面对含数百至数千位置的审计 transcript 时无法支持实时监控。
2. **现有位置选择策略盲目**：Prior work 多采用固定规则（如等间距采样、仅看答案前最后一位），或使用基于 attention weight 的启发式方法，难以覆盖真正携带威胁信息的 informative token。
3. **缺乏信号-解释相关性的量化基准**：如何衡量仅凭单次前向传播可得的各种计算信号（预测分布、注意力、激活向量）与“解释是否针对特定威胁”之间的关联，尚未被系统研究。
4. **审计场景的预算约束**：审计员通常只能负担极少量解释（k ≪ n），亟需一种在生成解释前即可计算的 ranker，将高风险位置优先排序以最大化有限预算的命中率。

## 核心贡献（创新点）
1. **首个大规模 NLA 位置选择实证研究**：在 4 个模型与 4 个数据集上生成并评估了 4,705,657 条解释，建立了从单 token 信号到解释 on-task 标签的精确映射。
2. **激活向量信号在个体信号中表现最强**：在 14 个 dataset-model 组合中有 11 个的最优信号来自 activation 家族；组合 247 个候选信号可进一步提升跨 transcript 的泛化排名能力。
3. **聊天结构排名器显著优于计算信号**：仅使用 chat role、segment 与边界序位训练的轻量基线，在 12/14 个组合的 pooled AUROC 上超越最优单信号，且完全不需要前向传播。
4. **5% 位置预算即可保留近乎完整审计成功率**：在 OpenPromptInjection、taboo organisms 和 Tensor Trust 三个数据集上，仅解释 5% 的位置可保留 95.8%
