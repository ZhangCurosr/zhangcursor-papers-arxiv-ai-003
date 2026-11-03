---
title: "When-Does-Momentum-Help-in-Reinforcement-Learning-with-Verif"
source: https://arxiv.org/pdf/2610.20413v1.pdf
model: agnes-2.5-flash
chunks: 3
summarized_at: "2026-11-03 09:02:29"
---

# 论文速读：When-Does-Momentum-Help-in-Reinforcement-Learning-with-Verif

## 一句话总结
本文系统揭示了EMA动量在可验证奖励强化学习（RLVR）中“固定大β失效、缩放β有效”的渐近机理，提出将动量系数按学习率动态缩放（$\beta=1-\eta/\rho$）的优化策略，并通过Bandit数值实验与Qwen系列LLM后训练验证了该策略可稳定提升奖励目标并压制梯度估计方差。

## 研究问题与动机
- RLVR是当前提升大语言模型推理能力的主流后训练范式，但针对其优化器设计与动量超参配置的系统研究严重缺失，工业界仍普遍沿用Adam/AdamW及固定一阶动量（β=0.9）。
- 现有实证报告（如Mukherjee et al., 2026）指出，在传统固定β设置下动量不仅无法持续改善RLVR性能，反而可能拖累训练，引发社区对“动量是否适配RLVR”的争议。
- 本文旨在从理论层面厘清EMA对RLVR有益的确切条件，解释固定动量失效的数学原因，并给出可落地的动量系数设置准则。
- 为此，论文在KL-正则化响应级Bandit设定下建立自然策略梯度（NPG）与bias-corrected EMA的分析框架，推导期望增益与方差随η、β变化的渐近行为。

## 核心贡献（创新点）
- **揭示固定动量失效的渐近机理**：严格证明固定β下期望增益以$O(\eta)$趋于零，慢于标准差$O(\sqrt{\eta/B})$的衰减速度，导致动量优势被采样噪声完全淹没；与以往仅依赖经验调参或黑盒对比的工作不同，本文首次给出RLVR动量失效的数学证明。
- **提出缩放动量策略（Scaled Momentum）**：论证将动量系数设为$\beta = 1 - \eta/\rho$时，期望增益可恢复为正常数阶$O(1)$，且性能优势在有限步内稳定可观测；区别于传统一阶动量固定窗口设定，该策略将动量历史长度与学习率动态耦合。
- **建立动量系数与梯度方差的定量权衡理论**：证明在相同样本预算下，配合较小η增大β可在维持期望性能的同时显著降低目标值方差；现有RLVR文献多聚焦收敛速率，本文首次刻画动量配置对输出估计方差的压制作用。
- **理论与大规模LLM实验的双向验证**：在10-arm Bandit中精确复现渐近正态性与胜率趋近现象，并在Qwen3-4B/Qwen2.5-32B的RLVR管线中验证缩放动量的可行性；与纯理论分析不同，本文在真实RLVR基准上闭环检验了渐近结论的有限步实用性。

## 方法详解
- **目标建模**：采用KL-正则化响应级Bandit形式（Xiong et al., 2024），目标函数为$J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[r(x,y)] - \gamma \mathrm{KL}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$，其中$r(x,y)\in\{0,1\}$为确定性二元奖励。
- **优化器动力学**：基于softmax参数化的自然策略梯度，结合bias-corrected EMA更新一阶动量：$m_{t+1} = \beta m_t + (1-\beta)\mathcal{F}^\dagger(\theta_t^\beta)\nabla J(\theta_t^\beta)$，参数更新$\theta_{t+1}^\beta = \theta_t^\beta + \eta \frac{m_{t+1}}{1-\beta^{t+1}}$。
- **关键无量纲量**：定义$\rho = \eta/(1-\beta)$刻画有效记忆窗长度，该参数统一
