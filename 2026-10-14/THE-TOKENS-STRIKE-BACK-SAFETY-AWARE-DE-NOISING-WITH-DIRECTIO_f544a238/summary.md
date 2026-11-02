---
title: "THE-TOKENS-STRIKE-BACK-SAFETY-AWARE-DE-NOISING-WITH-DIRECTIO"
source: https://arxiv.org/pdf/2610.17636v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-11-02 09:08:20"
---

# 论文速读：THE-TOKENS-STRIKE-BACK-SAFETY-AWARE-DE-NOISING-WITH-DIRECTIO

## 一句话总结
本文针对文本质扩散模型（TDM）的安全漏洞，提出 **SADDER** 方法，通过在 embedding 空间提取 position-wise 安全指导信号并选择性修改少数风险位置，在不增加推理成本（NFE 不变）的前提下显著降低有害输出率，同时保持良性生成质量。

## 研究问题与动机
- **TDM 安全性研究匮乏**：现有安全防御主要针对自回归 LLM（AR-LLM），而 Text Diffusion Models 面临毒物生成与 jailbreak 攻击时缺乏专用防护机制。
- **传统 AR-LLM 方法不可直接迁移**：微调、内容过滤、推理时解码修改等策略依赖自回归生成的序贯特性，难以套用于扩散模型的并行去噪架构。
- **现有 SOTA 方法（SAD）的局限**：SAD（Yusuf et al. 2026）仅在 token 概率空间进行训练无关的安全引导，忽略了语义更丰富的 embedding/activation 空间，无法实现位置特异性的精准干预。
- **安全-质量-成本的权衡难题**：亟需一种既能在表征空间精准定位风险、又保持单轮生成成本的方法，以突破当前防御手段的瓶颈。

## 核心贡献（创新点）
- **提出 SADDER 框架**：首次在 embedding 空间提取 position-wise 安全方向并进行选择性门控干预，突破了 SAD 局限于概率空间的瓶颈，实现更安全精准的防御。
- **改进 CSD 的零边界构造**：沿用 Lee & Han (2026) 的配对样本构造，但将决策边界从零阈值 $\theta$ 替换为安全/非安全质心的欧氏垂直平分超平面，使门控对称性更强、对分布偏移更鲁棒。
- **建立位置-wise 安全干预的理论保证**：给出 Theorem 3.1，精确刻画了最优干预强度条件，并讨论各位置-wise 分布何时可合并为全局安全分布。
- **引入选择性变体与增强策略**：提出 Selective SADDER 及 remasking（`top-b`/`aligned`）、warm-up 等可选插件，以极低计算开销实现有害率大幅下降与质量几乎无损。

## 方法详解
- **Phase 1（风险位置选择）**：对每个去噪步 $t$，计算位置 $l$ 的隐藏状态 $h_{k,l}^{(t)}$ 与单位方向 $\hat{v}_k$ 的投影得分：
  $$s_{k,l}^{(t)} = \langle h_{k,l}^{(t)} - m_k, \hat{v}_k \rangle$$
  其中 $m_k$ 为安全/非安全均值的中间点。若 $s_{k,l}^{(t)} > 0$ 则置二值门控 $a_{k,l}^{(t)} = \mathbb{1}\{s_{k,l}^{(t)}>0\}$。MDLM 仅在 masked 位置计算门控；USDM 因任何 token 均可被改变，每步对所有位置计算门控。
- **Phase 2（SAD 修正）**：仅对 $a_{k,l}^{(t)}=1$ 的位置应用安全引导更新：
  $$\tilde{p}_{0|t,l} = \mathbb{E}_\mathcal{D}[\boldsymbol{x}_0^l|\boldsymbol{x}_t] + \eta\beta(\boldsymbol{x}_t)a_{k,l}^{(t)}\big(\mathbb{E}_\mathcal{D}[\boldsymbol{x}_0^l|\boldsymbol{x}_t] - \mathbb{E}_{\mathcal{D}_\text{unsafe}}[\boldsymbol{x}_0^l|\boldsymbol{x}_t]\big)$$
- **可选增强策略**：
  - **Remasking**：`top-b` 重新打开得分最高的 $b$ 个已揭示位置；`aligned` 重新打开所有正分已揭示位置。
  - **Warm-up**：前 $w$ 步对所有遮蔽位置执行 Full SAD，之后切换为选择性引导，帮助模型早期稳定安全方向。
- **理论支撑（Theorem 3.1）**：当干预强度满足 $\eta\beta(\boldsymbol{x}_t)a_{k,l}^{(t)} = \frac{\pi_{t,l}(\boldsymbol{x}_t)}{1-\pi_{t,l}(\boldsymbol{x}_t)}$（$\pi_{t,l}$ 为位置 $l$ 处于非安全状态的后验概率）时，SADDER 的输出一一对应位置-wise 安全目标分布的条件均值。

## 实验与结果
- **评估模型**：LLaDA-8B-Base / LLaDA-8B-Instruct；Dream-7B-Base / Dream-7B-Instruct。
- **评测数据集**：RTP（有害续写）、BeaverTails（指令-响应安全标注）、JailbreakBench、AdvBench、HarmBench、WildJailbreak、XSTest。
- **评估指标**：Unsafe Rate（↓）、Harmful/Benign PPL（↓）、Benign BERTScore（↑）、NFE（不变更优）、Llama-Guard-3-8B 与 HarmBench 分类器安全评分。
- **LLaDA-Base on RTP**：
  | 方法 | Unsafe%↓ | Harmful PPL↓ | Benign PPL↓ | Benign BERT↑ | NFE |
  |---|---|---|---|---|---|
  | Unmodified | 38.6 | 21.1 | 17.5 | 84.65 | 32 |
  | Full SAD | 32.4 | 29.3 | 30.0 | 83.92 | 32 |
  | **SADDER** | **29.4** | 42.1 | 18.4 | 84.59 | 32 |
  | SADDER+warm-up | 31.2 | 31.6 | 44.3 | 83.01 | 32 |
  | CSD† | 22.4 | 225 | 21.6 | 84.59 | 61.9 |
  | BoN-8 | 19.0 | 20.5 | 19.5 | 84.70 | 256 |
- **LLaDA-Base on BeaverTails**：SADDER unsafe rate 29.9%，SADDER+warm-up 降至 24.4%。
- **核心结论**：SADDER 在 **NFE 保持 32（与 Unmodified 相同）** 的条件下，将 RTP 有害率从 38.6% 降至 29.4%（↓9.2%），Harmful PPL 升至 42.1（模型更难生成有害内容），Benign PPL 仅微升 0.9，质量近乎无损；显著优于 NFE 高达 61.9 的 CSD 与 NFE=256 的 BoN-8。Dream-7B 结果因原文截断未完整呈现。

## 相关工作脉络
- **SAD (Yusuf et al. 2026)**：TDM 训练无关安全引导的先驱工作，但仅在 token 概率空间操作；本文将其拓展至 embedding 空间并引入选择性门控，实现更精准的干预。
- **CSD (Lee & Han 2026)**：基于配对样本的激活空间干预方法；本文沿用其构造思路但改进决策边界为中点超平面，并在保持相近安全性的同时将 NFE 从 61.9 降至 32。
- **传统 AR-LLM 安全机制**（微调/过滤/解码修改）：依赖自回归生成特性，无法直接迁移至扩散架构；本文填补了 TDM 专属安全干预的方法学空缺。
- **BoN-8 / 采样类防御**：通过多次采样择优提升安全性，但计算开销极高（NFE
