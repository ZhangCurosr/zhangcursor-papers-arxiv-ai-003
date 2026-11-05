---
title: "The-Router-Already-Knows"
source: https://arxiv.org/pdf/2610.27421v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:50:51"
---

# 论文速读：The-Router-Already-Knows

## 一句话总结
本文提出 NR-SAE（Native Router Sparse Autoencoder），将 MoE 模型中预训练路由器的稀疏选择结果直接复用为稀疏字典的寻址先验，每个被路由的专家绑定独立的小型字典库。在 Gemma 4 26B-A4B 第 15 层上，该方法在重构与预测双任务中均显著优于参数量/候选数/架构严格匹配的对照组，证明预训练路由器已为字典学习提供了有价值的子空间划分先验。

## 研究问题与动机
- MoE 层已通过路由器为每个 token 完成专家稀疏选择，而现有稀疏自编码器（SAE）仍需在全局重叠的激活空间中二次学习寻址，路由器的划分结构通常被丢弃。
- 若字典重复学习路由器已完成的划分，不仅浪费参数容量，还会削弱特征的结构化与可解释性。
- MoE 专家的原始输出受限于
