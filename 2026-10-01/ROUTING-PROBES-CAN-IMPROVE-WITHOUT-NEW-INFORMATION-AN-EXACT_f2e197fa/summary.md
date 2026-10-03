---
title: "ROUTING-PROBES-CAN-IMPROVE-WITHOUT-NEW-INFORMATION-AN-EXACT"
source: https://arxiv.org/pdf/2609.38956v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-03 18:27:14"
---

# 论文速读：ROUTING-PROBES-CAN-IMPROVE-WITHOUT-NEW-INFORMATION-AN-EXACT

## 一句话总结
本文通过构建精确标签空值（Exact Label Null）对照实验，证明现代 ViT 路由探针（专家门、注意力残差权重、halt 分数）对错误预测的“增益”主要源于评估流程缺陷（如 accuracy 选点与 log-loss 评估的错配），而非路由携带了超越模型输出的增量信息。

## 研究问题与动机
- 现有研究普遍将 ViT 路由信号与模型错误的关联性解读为“路由携带了超出模型输出的错误信息”，并据此设计探针以提升预测。
- 该推断忽略了“可有用信息 ≠ 信息内容”（Hewitt et al., 2021；Xu et al., 2020）：即使 $T \perp R \mid O$，受限函数类仍可能从冗余 $R$ 中获得正增益（Proposition 1），导致假阳性检测。
- 既有实验缺乏严格的概率空值对照，容量匹配、噪声控制与路由分配控制等混淆因素未被剥离，历史基准结论可能存在系统性偏差。

## 核心贡献（创新点）
1. **提出精确标签空值（Exact Label Null）检验框架**：通过冻结基于 $O_0$ 的 logistic 生成器并重绘 $T^*$，从构造上强制 $T^* \perp R \mid O_j$，首次实现对路由增益宣称的严格概率校准。
2. **揭示原始检测几乎全由评估管道缺陷驱动**：证明 accuracy-based checkpoint selection 与 log-loss/Brier 评估之间的 epoch 错配是假阳性的主因，填补了路由探针研究的方法论盲区。
3. **形式化“冗余增益悖论”（Proposition 1）**：构造 $T \perp R \mid O$ 但受限函数类仍可获信息增益的严格反例，证明探测改进并非 $(T, O, R)$ 的泛函，澄清了信息论与 ML 评估之间的概念混淆。
4. **建立多维对照与统计筛选体系**：整合 RAW、GLOBAL/MATCHED shuffle、NOISE、CPT 条件置换检验与 R1/R2 双重筛选规则，提供了一套可复现的路由探针稳健性审计基准。

## 方法详解
- **精确标签空值构建**：将数据划分为 disjoint 半集 A 与 B，在 A 上拟合 logistic 生成器 $g(O_0)$ 并冻结，在 $B_{\text{test}}$ 上仅依赖 $O_j$ 重绘 $T^*$，由构造保证条件独立性 $T^* \perp R \mid O_j$。
- **四种输出视图**：$O_0$（top-class confidence/clipped logit）；$O_1$（confidence + predictive entropy + top-two margin + logit $L_2$ norm + runner-up probability）；$O_2$（sorted class probabilities，无类别身份）；$O_3$（sorted logits，无类别身份）。
- **探针与训练**：逻辑回归（linear probe）与 scikit-learn MLP（1隐层64单元，$\alpha=10^{-3}$，patience=15）；特征经标准正态化，采用 5折×2次 cross-fit。
- **对照设计**：zero-padded $(O_j, \mathbf{0})$、全局 shuffle $\pi(R)$、output-matched shuffle MATCHED、同宽高斯噪声 $\varepsilon$。
- **统计量与检验**：$\hat{\Delta}_{\text{raw}}$、$\hat{\Delta}_{\text{shuf}}$、$\hat{\Delta}_{\text{noise}}$（正值 favor routing）
