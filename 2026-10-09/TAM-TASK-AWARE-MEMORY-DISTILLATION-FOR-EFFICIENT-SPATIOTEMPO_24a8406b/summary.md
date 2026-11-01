---
title: "TAM-TASK-AWARE-MEMORY-DISTILLATION-FOR-EFFICIENT-SPATIOTEMPO"
source: https://arxiv.org/pdf/2610.11617v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:29:05"
---

# 论文速读：TAM-TASK-AWARE-MEMORY-DISTILLATION-FOR-EFFICIENT-SPATIOTEMPORAL-PREDICTION

## 一句话总结
本文提出 TAM（Task-Aware Memory Distillation），通过构建任务感知的冻结教师历史记忆库，将跨样本的预测结构与残差先验蒸馏给轻量学生模型；在视频预测、天气建模与交通流量预测三大任务上，在不增加推理成本的前提下显著提升 SSIM 并降低 MSE。

## 研究问题与动机
- 现有知识蒸馏方法通常逐样本独立匹配教师输出或特征，忽略了样本间的跨样本预测结构，导致监督信号利用率不足。
- 不同时空预测任务对“有效历史参考”的定义差异显著（视频依赖运动/占用区域、天气依赖地理位置、交通依赖观测流量与近期趋势），通用特征记忆库无法显式适配任务动态。
- 教师模型的绝对精度并不直接保证蒸馏效果，需设计学生可稳定利用的监督形式；历史样本“存什么、如何用、选谁做参考”缺乏统一且任务适配的机制。
- 现有关系蒸馏与内存队列（如 CIRKD、MoCo）主要针对图像分类或语义分割，未充分考虑预测任务中时间演化、空间位置与观测条件对参考选择的约束。

## 核心贡献（创新点）
- **提出任务感知的教师历史记忆接口**：将冻结教师的预测与中间表征显式组织为可检索的有界历史序列，实现跨样本监督信号的显式存储与选择，区别于仅依赖单样本匹配或通用 FIFO 队列的既有方法。
- **设计两种任务特化蒸馏目标**：开发共享参考系下的相似分布匹配（Relation Matching）与观测条件残差原型回归（Residual Prototype），前者保留跨样本相对相似度结构，后者直接将历史教师残差聚合为回归目标，分别适配视频/天气与交通流场景。
- **构建零推理增量的联合训练框架**：将历史记忆监督与监督预测、输出蒸馏及可选特征对齐统一优化；教师、记忆库与辅助 Adapter 仅在训练阶段参与，学生推理架构与延迟完全不变。

## 方法详解
- **基础设定与监督损失**：给定观测序列 $X=(X_1,\dots,X_T)$ 预测未来 $Y=(Y_1,\dots,Y_K)$，监督目标为条件最大似然，在高斯观测假设下等价于 MSE $\mathcal{L}_{\mathrm{sup}}=\mathrm{MSE}(\widehat{Y}_s, Y)$。教师 $F_t$ 与学生 $F_s$ 固定方差下并行训练，引入输出蒸馏 $\mathcal{L}_{\mathrm{out}}=\mathrm{MSE}(\widehat{Y}_s, \mathrm{sg}(\widehat{Y}_t))$ 与可选特征对齐 $\mathcal{L}_{\mathrm{feat}}$。
- **任务感知表征与参考选择**：定义任务特化表示 $u_i^a = \phi_a(X, Z_a, \widehat{Y}_a)_i$（可为潜特征、预报差分或流量残差）。参考选择函数 $\mathscr{R}_i = \mathrm{Select}(M; X, u_i^t, \gamma_i)$ 依据地理坐标、运动掩码或观测键动态筛选历史条目，确保跨样本对比语义相关。
- **共享参考系关系匹配**：对归一化教师参考 $\nu_j$ 与查询 $q_i^a = u_i^a/\|u_i^a\|_2$，计算温度 $\tau$ 下的相似度分布 $p_{i,j}^a = \exp((q_i^a)^\top \nu_j/\tau) / \sum_{\ell\in\mathcal{R}_i}\exp((q_i^a)^\top \nu_\ell/\tau)$。学生通过 KL 散度拟合教师分布：$\mathcal{L}_{\mathrm{rel}} = \sum_{i\in\mathcal{A}} w_i D_{\mathrm{KL}}(\mathrm{sg}(p_i^t) || p_i^s)$，传递跨样本相对结构。
- **观测条件残差原型回归（交通特化）**：以最新观测 $X_T$ 为持久化基准，计算教师残差 $\widehat{Y}_{a,h} - X_T$ 并按近/远预报步分组聚合为向量 $r_a^g$。构造观测键 $k(X) = \mathrm{norm}(\mathcal{P}([X_T, X_T-X_{T-1}]))$，按余弦相似度检索 Top-$k$ 历史条目，加权
