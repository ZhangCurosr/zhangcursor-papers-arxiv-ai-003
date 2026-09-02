---
title: "When-Safety-Routing-Breaks-Understanding-Alignment-Fragility"
source: https://arxiv.org/pdf/2609.01455v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:21:56"
---

# 论文速读：When-Safety-Routing-Breaks-Understanding-Alignment-Fragility

## 一句话总结
本文通过Fisher几何分析与因果激活干预，揭示了大语言模型在仅用100条良性数据微调后安全拒绝行为就会迅速崩塌的根本原因：对齐过程将安全相关的Fisher曲率大幅压缩并平铺，但保留了低秩的输出路由通路；良性微调选择性地使该通路末端的MLP模块重新尖锐化，导致拒绝映射被覆盖，而内部安全表征并未丢失。

## 研究问题与动机
- 核心问题：为何即使使用完全不含有害指令的良性下游数据，后续的SFT微调仍会严重削弱已对齐LLM的拒绝行为？
- 梯度冲突解释的不足：Prior work认为少数对抗性良性样本的梯度方向与安全参数冲突导致崩塌。本文实验表明，随机采样、低冲突与高冲突样本子集均引发相近程度的安全崩塌，梯度冲突仅为调节因素而非必要条件。
- 表征结构的几何缺口：已有研究指出安全编码于极少神经元或低维子空间，但缺乏对“微调如何局部破坏安全表征到拒绝输出的映射路径”的几何与因果机制解释。
- 可逆性谜题：为何极少量安全示例或固定拒答前缀就能快速恢复拒绝行为？现有框架难以统一解释脆弱性与可逆性，亟需一个能同时刻画“易损”与“可修复”的底层机制。

## 核心贡献（创新点）
- 提出Fisher几何解释框架，将安全脆弱性定位到输出侧路由通路的局部曲率漂移，而非全局梯度冲突或知识遗忘。
- 通过块级Fisher谱分析、logit-lens追踪与跨条件激活插桩，在参数几何、信号相关性与因果效力三个层面交叉验证了“末端MLP down_proj选择性重新尖锐化”的机制。
- 系统量化了安全与效用降级的不对称性（ASR可从0%升至85.7%，而SciQ等效用指标仅下降约10%），并揭示了LoRA与ASAM在小样本阶段有效、但在大规模微调下保护失效的两阶段规律。
- 从机制出发解释了安全对齐的可逆性：内部安全表征完好，仅需少量监督或推理时前缀引导即可修复输出路由，为低开销安全修复提供了理论依据。

## 方法详解
- **块级Fisher几何度量**：定义安全偏离损失 $\mathcal{L}_{\mathrm{safe}}(\theta;\theta^*) = \mathbb{E}_{x} D_{\mathrm{KL}}(\pi_{\theta^*}(\cdot|x) \| \pi_{\theta}(\cdot|x))$，其局部泰勒展开的二阶项系数即为安全Fisher矩阵 $F_{\mathrm{safe}}$。实验中使用块级经验Fisher近似：$\widehat{F}_b(\theta) = \frac{1}{N}\sum_i g_b^{(i)} g_b^{(i)\top}$，其中 $g_b^{(i)} = \nabla_{\theta_b} \log \pi_\theta(y_i|x_i)$，$b$ 索引层-模块块（如 down_proj）。
- **曲率漂移度量**：以各模块的最大特征值 $A_b(\theta) = \lambda_{\max}(\widehat{F}_b(\theta))$ 作为局部sharpness，定义对数变化率 $\Delta A_b^{\mathrm{src}\to\mathrm{tgt}} = \log_{10}(A_b(\theta_{\mathrm{tgt}})/A_b(\theta_{\mathrm{src}}))$，正值表示重新尖锐化，负值表示进一步平坦化。
- **低秩与谱集中分析**：计算各模块Top-k特征值质量比 $R(k) = \frac{\sum_{b}\sum_{j=1}^k \lambda_{b,j}}{\sum_{b}\sum_{j=1}^{64}\lambda_{b,j}}$，揭示安全Fisher比效用Fisher更集中（Avg $R(1)$: 0.491 vs 0.271；第30层: 0.718 vs 0.223）。
- **Logit-lens与拒绝-顺从边界**：通过最终未嵌入矩阵 $W_U$ 与RMSNorm解码各层隐状态 $h_\ell(x)$，计算 $\bar{m}_\ell = \frac{1}{|\mathcal{D}|}\sum_x (\max_{t\in\mathcal{R}} z_\ell(x)_t - \max_{t\in\mathcal{C}} z_\ell(x)_t)$，追踪拒答信号的层间演化趋势。
- **跨条件激活插桩（Activation Patching）**：在相同输入下，将源
