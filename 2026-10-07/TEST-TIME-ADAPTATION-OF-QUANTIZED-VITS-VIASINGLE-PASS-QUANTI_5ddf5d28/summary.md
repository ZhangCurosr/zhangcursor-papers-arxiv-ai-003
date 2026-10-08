---
title: "TEST-TIME-ADAPTATION-OF-QUANTIZED-VITS-VIASINGLE-PASS-QUANTI"
source: https://arxiv.org/pdf/2610.08358v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-08 22:38:08"
---

# 论文速读：TEST-TIME-ADAPTATION-OF-QUANTIZED-VITS-VIASINGLE-PASS-QUANTI

## 一句话总结
针对分布偏移下量化ViT激活量化器失配导致的性能骤降，提出QuAR（单遍量化器对齐重校准）方法。通过单遍累积干净源统计量，在晚期block的局部接口与全局CLS接口进行部分缩合匹配（λ<1），无需额外优化或污染数据即可显著提升W3A3~W8A8各精度下的鲁棒准确率。

## 研究问题与动机
- 量化ViT在测试时遭遇分布偏移（如ImageNet-C污染）后，激活均值偏移（0.235σ）与尺度收缩（0.92倍）导致量化器失配，引发大量码字饥饿与死信道，准确率大幅下滑。
- 现有TTA方法（ZOA/NEO等）多依赖运行时统计或仅做均值重中心化，无法同步恢复源校准的量化器尺度，对激活码分布的校正效果有限。
- 基于优化器的方法（FOA/FOZO）需多次前向/反向或超参敏感，计算开销大且易在持续流下产生累积漂移。
- 完全匹配源统计（λ=1）会注入估计噪声方差，在batch较小或标签偏移时反而过校正，导致泛化性下降。

## 核心贡献（创新点）
- 提出单遍双分支量化器对齐重校准框架QuAR，首次将“源统计部分匹配”引入量化ViT的测试时适应。
- 设计局部分支（blocks 8-11的qkv/proj/fc1/fc2接口，lagged更新）与全局分支（post-LN CLS特征，inclusive更新），利用深度后向增益衰减特性实现偏差-方差最优。
- 理论推导最优收缩系数λ*的结构形式，证明部分匹配（默认λ=0.5/1）优于完全匹配，并为不同架构提供λ选取依据。
- 引入方差比clamp机制（ρ=3），在小batch、class-sorted流及长尾持续适应中保障统计估计稳定性。
- 提供细粒度量化误差诊断（码分布距离削减52%、失活信道从20.6%降至6.9%），厘清“位置性偏移”为主因而非量化步长变粗。

## 方法详解
- **双分支校准设计**：局部分支覆盖blocks 8-11的17个接口，采用历史batch的矩做修正（lagged，更新在后）；全局分支作用于post-LN CLS位置的pooled feature，采用含当前batch的矩做修正（inclusive，更新在前）。
- **缩合校准形式**：激活缩放采用 $a_\lambda = (1-\lambda) + \lambda \cdot \sigma_{s,c}/\hat{\sigma}_{t,c}$ 的凸组合形式，通过匹配源均值$\mu_{s,c}$与标准差$\sigma_{s,c}$使激活重新占据量化器校准码字范围。
- **最优λ理论**：推导得出 $\lambda^* = (1-r)^2 / [(1-r)^2 + r^2\delta^2]$，随源统计估计噪声$\delta^2$增大而减小；默认局部λ=0.5、全局λ=1为稳健配置。
- **方差比Clamp**：对协方差/方差比施加裁剪阈值$\rho=3$，防止极端batch下的统计爆炸，在bs=1 class-sorted流下挽回1.6 pts。
- **单遍源锚定**：仅在source阶段用12,800张干净ImageNet验证图单次前向累积$(\mu_s, \sigma_s)$，测试时冻结不参与梯度更新，杜绝累积漂移风险。
- **深度选址依据**：理论bound表明后期block后向增益$K_{>\ell}$更小（~0.059 vs 早期0.26，衰减~4.4倍），故晚期recalibration对logits扰动最小；全网络校准会导致accuracy从53.42崩塌至31.93。

## 实验与结果
- **数据集与基线**：ImageNet-C（50k，severity 5）、ImageNet-R/Sketch/3DCC及4种held-out污染；基线包括No Adapt、T3A、FOA、FOZO、ZOA、NEO、Local-only、Global-only。
- **主结果（ImageNet-C，3-seed均值±std）**：
  | 方法 | W3A3 | W4A4 | W6A6 | W8A8 |
  |---|---|---|---|---|
  | No Adapt | 28.26 | 47.24 | 47.60 | 54.21 |
  | **QuAR** | **36.18±0.02** | **53.79±0.01** | **54.64±0.01** | **60.82±0.01** |
  QuAR在全部bit-width上均最优，W6A6较No Adapt提升+7.04 pts。
- **消融与敏感度**：Local-only（53.74）> Global-only（51.20）于W6A6，主要增益来自恢复量化器尺度；λ=0.5为局部最优（54.35%）；batch size 1~128稳定在54.27-54.47。
- **OOD泛化**：ImageNet-Sketch（+5.52）、ImageNet-C held-out污染（+3.3~+6.7）均为最优；ImageNet-R上recentering基线略优但QuAR综合最强。
- **机制验证**：W6A6码分布距离削减52%；失活信道（≥0.5 bit损失）从20.6%降至6.9%；完全匹配λ=1仅恢复84.2%但准确率（53.52）反低于QuAR（54.35）。
- **CNN迁移**：ResNet-50（GroupNorm/BatchNorm）上QuAR同样有效，W6A6 ImageNet-C从30.12提升至31.40（BatchNorm版17.32→24.57），λ最优值随架构微调（CNN峰在0.25）。

## 相关工作脉络
- **PTQ4ViT (Yuan et al., 2022)**：ViT双均匀量化器预训练基线，本文理论模型与W6A6/W8A8实验的直接来源。
- **AdaLog (Wu et al., 2024)**：引入log量化器，本文指出其仅mlp.fc2接口为log形式，其余仍为uniform，QuAR可兼容其特定接口。
- **ZOA (Deng et al., 2025)**：EMA适配LayerNorm仿射参数，本文证明其码分布几乎不变（-4.0% vs QuAR -51.7%），仅消除1%漂移。
- **NEO (Murphy et al., ICLR 2026)**：无参数latent re-centering（仅减运行测试均值），头接口恢复20.6%，中间接口<1.5%，缺乏尺度恢复能力。
- **FOA/FOZO (ICML/CVPR 2024-2026)**：基于CMA-ES或SPSA的扰动优化方法，需多步前向/反向，本文证明其计算开销大且在持续流下存在漂移风险。
- **Banner et al. (2019) & Ledoit & Wolf (2004)**：定量误差granular+clipping分解与协方差shrinkage的理论奠基，本文将其推广至量化ViT测试时自适应场景。

## 局限性与未来方向
- 理论界仅覆盖一阶矩级陈述（A1-A4），未严格bound标准化形状变化（高阶矩偏移）时的残差误差。
- 未建立bit-width换汇率与准确率点之间的直接非线性映射，跨精度对比仅依赖实验观测。
- Proposition U.4对log量化接口的理论论证尚未在W6A6测试时实证（该接口实际为twin-uniform），log量化下的尺度校正
