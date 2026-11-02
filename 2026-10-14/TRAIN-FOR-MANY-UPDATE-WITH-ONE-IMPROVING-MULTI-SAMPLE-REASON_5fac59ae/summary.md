---
title: "TRAIN-FOR-MANY-UPDATE-WITH-ONE-IMPROVING-MULTI-SAMPLE-REASON"
source: https://arxiv.org/pdf/2610.17735v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 09:06:50"
field: "多样本推理与变分后验学习"
keywords: ["多样本推理", "有限K后验学习", "Fenchel-Bregman优化", "IVON", "Pass@K", "变分学习", "候选覆盖"]
innovations: ["提出有限K后验学习以直接优化K候选覆盖率", "Fenchel对偶分离使单次采样完成后验更新", "闭式Bregman EMA追踪避免额外采样"]
benchmarks: ["Sudoku-Extreme", "Maze-Hard", "ARC-AGI-1", "ARC-AGI-2", "MATH500", "OlymMATH", "AMC23", "AIME24", "AIME25"]
---

# 论文速读：TRAIN FOR MANY, UPDATE WITH ONE: IMPROVING MULTI-SAMPLE REASONING VIA FENCHEL–BREGMAN POSTERIOR LEARNING

## 一句话总结
本文提出有限K后验学习（finite-K posterior learning）框架，将求解器参数上的分布训练为直接优化K个独立候选的覆盖率（Pass@K），并设计Fenchel–Bregman（FB）优化器FBI，仅用每次更新一次后验采样即可达成与直接使用K次采样的Direct-K相当的覆盖率，训练时间仅为后者的12.3%。

## 研究问题与动机
- 多样本推理以Pass@K（至少一个候选正确）为指标，但传统变分学习最小化单次采样的期望损失，与多候选覆盖目标不匹配。
- 直接用K次独立后验采样估计全失败概率（Direct-K）是无偏的，但每次更新需K次采样，计算成本高。
- 现有方法（GRPO、DMPO、MMPO、Pass@K Training等）多优化单模型生成候选，或限制置信度/最佳N选择，未直接从求解器参数分布层面对齐有限候选池的覆盖目标。
- 多路径复杂推理问题中，单次尝试可能失败而另一次成功，因此学习候选生成分布比固定分布更有优化空间。

## 核心贡献（创新点）
- 提出有限K后验学习：以最小化所有K个独立候选全失败概率为核心目标，对齐多样本推理的Pass@K指标。本质区别在于训练对象从单模型转为参数分布，目标从个体损失转为群体覆盖。
- 提出Fenchel–Bregman（FB）优化：通过精确Fenchel对偶将问题相关权重与后验更新解耦，使一次后验采样即可完成梯度更新。本质区别在于利用凸共轭表示把高阶失败概率分解为可单采样估计的加权形式。
- 提出闭式Bregman权重追踪：用指数滑动平均（EMA）更新观测失败估计，无需额外后验采样。本质区别在于权重更新来自同一观测，而非重新采样估计。
- 以IVON实例化得到FBI：在递归与自回归推理任务上统一实现，并扩展到对角与结构化高斯后验（EVON）。本质区别在于FB可与不同变分优化器结合，通用性强。
- 实验验证FBI在匹配比较下以12.3%的训练时间达到与Direct-K相当的Pass@K，且在多骨干和数学推理上显著提升覆盖率。

## 方法详解
- 有限K目标：对问题x，定义单次失败指示$F(\theta,x)\in\{0,1\}$，后验失败概率$\bar{F}_\phi(x)=\mathbb{E}_{\theta\sim q_\phi}[F(\theta,x)]$。K次独立采样的全失败概率为$\bar{F}_\phi(x)^K$，最大化覆盖率等价于最小化$\mathbb{E}_x[\bar{F}_\phi(x)^K]+\tau_{KL}D_{KL}(q_\phi\|p_0)$。
- 问题相关梯度权重：数据项梯度为$K\bar{F}_\phi(x)^{K-1}\nabla_\phi \bar{F}_\phi(x)$，因子$\lambda_x^*=K\bar{F}_\phi(x)^{K-1}$给其余K-1个候选仍失败的难题更大权重。
- Fenchelreformulation：利用$r^K=\max_{\lambda\in[0,K]}\lambda r-\psi_K(\lambda)$，其中$\psi_K(\lambda)=(K-1)(\lambda/K)^{K/(K-1)}$，将原目标改写为鞍点问题$\min_\phi\max_{\lambda_x}\mathbb{E}_x[\mathbb{E}_{\theta}[ \lambda_x F(\theta,x)]]-\psi_K(\lambda_x)+\tau_{KL}D_{KL}$。
- 固定权重的单采样后验更新：当$\lambda_x$固定时，$\nabla_\phi\mathbb{E}_{\theta}[ \lambda_x F(\theta,x)]=\lambda_x\nabla_\phi\bar{F}_\phi(x)$，一次采样即可无偏估计；在$\lambda_x=\lambda_x^*$时恢复有限K梯度。
- Bregman权重追踪：由$\psi_K$生成的Bregman散度给出闭式更新$\hat{F}_{x,t+1}=\beta\hat{F}_{x,t}+(1-\beta)F_t$，$\lambda_{x,t+1}=K\hat{F}_{x,t+1}^{K-1}$，其中$\beta=1/(1+\eta)$，等价于镜像上升步，无需额外采样。
- FBI算法流程：每步先由EMA估计形成$\lambda_{x,t}$，单次采样$\theta_t$计算加权损失$\mathcal{L}_t^{\text{data}}=\frac{1}{|B_t|}\sum_{x}\lambda_{x,t}f(\theta_t,x)$，用IVON更新后验均值与对角精度，再用同一样本更新EMA。
- 失败分数构造：使用$f=1-\exp(-\ell/T)$作为连续代理，T为问题相关温度；对自回归任务结合RLOO策略梯度估计。

## 实验与结果
- 递归推理基准：Sudoku-Extreme、Maze-Hard、ARC-AGI-1、ARC-AGI-2。FBI在Sudoku-Extreme达Pass@10=99.55%、选中精度99.53%；Maze-Hard从W-PTRM的89.80%提升至91.30%，选中精度88.00%；ARC-AGI-1/2在Pass@K上达到或并列最高。
- 自回归数学推理：Qwen3-4B-Base，训练MATH7.5K，评测MATH500、OlymMATH、AMC23、AIME24、AIME25。FBI平均Pass@8=60.44%，高于DMPO的59.60%（+0.84pp），在AMC23（85.20%）和AIME25（33.70%）上最佳。
- 覆盖率曲线：Maze-Hard上FBI与Direct-10（均设$K_{train}=10$）相当，但FBI用1/10后验采样；ARC与数学任务上FBI在K=8/16超越GRPO、Pass@K Training、DMPO。
- 跨骨干与贝叶斯方法：TRM与FPRM上FB均提升Pass@10；FBI/FBE超过IVON/EVON/LA/SWAG，FBE达92.7% Pass@10。
- 训练成本：FBI用Direct-10的12.3%时间与10.4% FLOP，峰值显存0.611GB vs 3.707GB。
- 最强结果：Maze-Hard Pass@10=91.30%（FBI vs Direct-10相当但成本极低）；数学推理平均Pass@8=60.44%（+0.84pp over DMPO）。

## 相关工作脉络
- Direct-K蒙特卡洛估计：无偏但需K次采样，FBI通过Fenchel分解保留同一学习信号而只需一次采样。
- IVON变分优化：用于对角高斯后验，FBI在其上叠加Fenchel权重与Bregman追踪，扩展至有限K目标。
- EVON结构化后验：Kronecker因式预条件器的块对角协方差，FBE将其与FB结合，证明方法不限于对角假设。
- GRPO/Pass@K Training/DMPO/MMPO/MaxRL：均为单模型多候选训练，FBI训练参数分布，目标与多样性机制不同。
- Laplace近似/SWAG：频域或频域近似后验，FBI通过变分迭代直接优化覆盖目标，优于这些近似基线。
- 固定高斯权重采样（W-PTRM/W-FPRM）：非学习扰动控制，FBI学习后验尺度与均值，动态适应数据。

## 局限性与未来方向
- 代理分数与真实二值失败的失配可能弱化权重追踪的可靠性。
- 随机梯度噪声与后验漂移影响EMA估计稳定性，尤其在早期训练阶段。
- 当前方法假设独立采样候选，未利用候选间相关性或自适应池。
- 未训练选择器（selector-aware training），部分正确候选未被选中。
- 未见问题的摊销估计与更广泛后验族/任务泛化仍需探索。
- 未来方向：改进代理校准、方差缩减追踪、自适应/相关候选池、-selector感知训练、更广泛基准与后验家族扩展。

## 研究启发与可借鉴点
- Fenchel对偶分离权重与后验更新的思路可迁移至其他需要高阶矩估计的变分训练场景。
- Bregman/EMA权重追踪提供闭式、低成本的二次信号估计，适用于在线/流式训练。
- 有限K目标与问题相关重加权机制可直接借鉴到多候选排序、best-of-N微调等任务。
- FBI与不同变分器（IVON/EVON）及不同骨干（TRM/FPRM）兼容，提示"目标-优化器-分布族"解耦的设计范式。
- 与RLOO、Math-Verify验证管线结合，展示在自回归推理中的落地可行性。

## 关键术语表
- **有限K后验学习**：训练参数分布以最小化K次独立采样全部失败的概率，对齐Pass@K覆盖目标。
- **Fenchel reformulation**：利用凸函数的共轭表示将高阶目标分解为线性加权形式，实现权重与后验更新解耦。
- **Bregman update**：由Fenchel共轭生成的闭式权重追踪，等价于镜像上升，避免重新采样。
- **IVON**：用于对角高斯后验的变分优化器，同时学习均值与逐参数采样尺度。
- **Pass@K**：K个候选中至少一个正确的概率，衡量多样本推理的覆盖率。
- **Direct-K**：用K次独立后验采样乘积估计全失败概率的无偏蒙特卡洛估计量。
- **失败分数f**：基于训练损失的连续代理$1-\exp(-\ell/T)$，用于可微训练。
- **EMA系数β**：控制过去失败估计保留强度的指数滑动平均参数。

## 可复现要素
- 数据集：Sudoku-Extreme、Maze-Hard、ARC-AGI-1、ARC-AGI-2公开；数学推理用MATH7.5K训练、MATH500/OlymMATH/AMC23/AIME24/AIME25评测。
- 代码/权重：论文未明确声明开源；使用TRM/FPRM/Qwen3-4B-Base公开checkpoint与IVON/EVON实现。
- 关键超参：$K_{train}\in\{1,2,5,10\}$，$\beta\in\{0.9,0.99\}$，相对KL强度2，ESS从$10^4$到$10^9$，学习率$10^{-4}$到0.5，温度T=1.0，RLOO基线，采样temperature=0.6、top-p=0.95。
