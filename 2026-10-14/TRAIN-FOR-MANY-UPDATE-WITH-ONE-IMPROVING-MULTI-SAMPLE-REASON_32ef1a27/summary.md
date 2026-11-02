---
title: "TRAIN-FOR-MANY-UPDATE-WITH-ONE-IMPROVING-MULTI-SAMPLE-REASON"
source: https://arxiv.org/pdf/2610.17735v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 09:07:14"
---

# 论文速读：TRAIN-FOR-MANY-UPDATE-WITH-ONE-IMPROVING-MULTI-SAMPLE-REASON

## 一句话总结
本文针对多样本推理中候选覆盖率（Pass@K）的优化难题，提出有限K后验学习框架与Fenchel–Bregman（FB）单次抽样优化方法FBI，在递归推理与自回归数学推理任务上显著提升了候选覆盖率，同时将训练计算成本降至直接蒙特卡洛基线的约1/8。

## 研究问题与动机
- 多样本推理通过生成K个候选解并评估“至少一个正确”的概率（Pass@K）衡量系统能力，但最小化个体平均损失无法直接优化这一群体级目标。
- 现有对齐训练与多样本推理的方法（如Pass@K Training、MMPO等）多针对单模型策略优化或需估计多个矩，未显式学习参数分布以控制候选池分布形态。
- 直接蒙特卡洛估计（Direct-K）虽无偏，但每次更新需K次独立后验抽样，计算开销随K线性增长，难以在实际推理预算下高效训练。
- 动机：在保留有限K目标完整学习信号的前提下，实现每次更新仅需单次新后验抽样的高效优化，弥合训练目标与多样本推理评测之间的失配。

## 核心贡献（创新点）
- 提出有限K后验学习框架，直接优化K候选池全失败概率；与现有单模型微调或固定权重采样方法的本质区别在于将训练目标明确对齐至多样本推理的群体覆盖率而非个体期望损失。
- 设计Fenchel对偶解耦机制实现单次抽样梯度更新；与直接重用同一样本（会退化为线性期望$\bar{F}$而非$\bar{F}^K$）的本质区别在于通过凸共轭重构严格保留$K\bar{F}^{K-1}$权重，使单抽样无偏估计高次幂目标。
- 推导闭式Bregman权重追踪更新；与需要额外蒙特卡洛重估或对偶重优化的本质区别在于将复杂对偶更新简化为问题特异失败率的指数移动平均（EMA），实现零额外计算成本的自适应权重演化。
- 实例化FBI并验证跨架构泛化性；与仅针对特定推理范式的优化方法的本质区别在于FB机制与IVON/EVON等变分优化器正交可插拔，在递归模型与自回归LLM上均有效。

## 方法详解
- **有限K目标形式化**：定义二值失败指示器$F(\theta, x) \in \{0,1\}$，后验失败概率$\bar{F}_\phi(x) = \mathbb{E}_{\theta \sim q_\phi}[F(\theta,x)]$。K个独立抽样的全失败概率为$\bar{F}_\phi(x)^K$，优化目标为$\min_\phi \mathbb{E}_x[\bar{F}_\phi(x)^K] + \tau_{KL} D_{KL}(q_\phi \| p_0)$。其对数据项的梯度可分解为问题特异权重$\lambda_x^* = K\bar{F}_\phi(x)^{K-1}$与单样本梯度$\nabla_\phi \bar{F}_\phi(x)$的乘积。
- **Fenchel重构解耦**：利用恒等式$r^K = \max_{\lambda \in [0,K]} \lambda r - \psi_K(\lambda)$（其中$\psi_K(\lambda) = (K-1)(\lambda/K)^{K/(K-1)}$），将原目标转化为鞍点问题$\min_\phi \max_{\lambda_x} \mathbb{E}_x[\lambda_x \bar{F}_\phi(x) - \psi_K(\lambda_x)] + \tau_{KL} D_{KL}$。固定$\lambda_x$时，后验更新仅依赖单次抽样估计加权损失$\lambda_x f(\theta_t, x)$。
- **Bregman对偶权重追踪**：为在$q_\phi$变化时自适应更新$\lambda_x$而不引入新抽样，构造Bregman-proximal对偶更新：$\lambda_{x,t+1} = \arg\max_\lambda \{\lambda F_t - \psi_K(\lambda) - \eta^{-1} D_{\psi_K}(\lambda, \lambda_{x,t})\}$。解析求解得失败率EMA更新$\widehat{F}_{x,t+1} = \beta \widehat{F}_{x,t} + (1-\beta)F_t$（$\beta = 1/(1+\eta)$），随后$\lambda_{x,t+1} = K\widehat{F}_{x,t+1}^{K-1}$。
- **FBI算法流程**：每步采样小批量→按当前$\widehat{f}_x$计算权重$\lambda_{x,t}$→单次抽取$\theta_t \sim q_{\phi_t}$计算失败分数$f(\theta_t, x)$→以加权损失$\mathcal{L}_t^{data} = \frac{1}{|B_t|}\sum_{x \in B_t} \lambda_{x,t} f(\theta_t, x)$执行IVON后验更新→用同一$f$更新EMA $\widehat{f}_x$。KL系数由IVON有效样本量（ESS）设定为$1/ESS$，温度$T$用于将连续损失映射为$[0,1]$失败分数$f = 1-\exp(-\ell/T)$。

## 实验与结果
- **递归推理任务**：在Sudoku-Extreme、Maze
