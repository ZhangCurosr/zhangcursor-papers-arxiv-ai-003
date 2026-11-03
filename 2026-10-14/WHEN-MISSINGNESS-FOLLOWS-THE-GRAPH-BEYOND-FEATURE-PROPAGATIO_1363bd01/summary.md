---
title: "WHEN-MISSINGNESS-FOLLOWS-THE-GRAPH-BEYOND-FEATURE-PROPAGATIO"
source: https://arxiv.org/pdf/2610.18113v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:38:21"
---

# 论文速读：WHEN-MISSINGNESS-FOLLOWS-THE-GRAPH-BYOND-FEATURE-PROPAGATIO

## 一句话总结
本文指出图神经网络缺失特征研究长期局限于均匀随机缺失假设，忽视了缺失模式沿图拓扑聚集的现实场景。为此提出递归传输（RT）机制与 PEMix 模型，前者刻画图结构化缺失，后者通过全局结构表示条件化概率特征模型进行补全，在复杂缺失 regime 下显著优于传统传播基线。

## 研究问题与动机
- **评估协议偏差**：现有图缺失特征方法（FP、PCFI、FISF 等）主要在 Uniform MCAR (U-MCAR) 或 Structural MCAR (S-MCAR) 下评估，假设缺失条目独立均匀分布，未覆盖传感器故障、通信中断等导致相邻节点特征同时缺失的真实场景。
- **缺失依赖维度的盲区**：传统 MCAR/MAR/MNAR 分类仅刻画掩码与数据 $(\mathbf{X},\mathbf{Y})$ 的依赖关系，忽略了掩码内部条目之间的空间相关性，即“缺失模式本身如何被图结构塑造”。
- **传播方法的固有缺陷**：特征传播类方法依赖从观测边界向缺失区域扩散信息，当缺失节点聚集成连通块时，可供注入的边界节点极少，导致补全质量断崖式下降。
- **缺乏理论解释与针对性方法**：现有工作多为经验性调优传播策略，缺少对传播瓶颈的严格界定，也缺乏不依赖值传播、直接利用全局结构先验的替代方案。

## 核心贡献（创新点）
1. **形式化图结构化掩码依赖并提出 RT 基准**：严格定义缺失掩码联合分布在给定图结构下的非因子化特性，并设计 Recursive Transmission 机制模拟从共享种子节点沿边扩展的局部故障。与已有工作仅关注缺失与特征/标签依赖不同，本文首次将掩码内部的空间拓扑依赖作为独立研究对象。
2. **证明特征传播的边界瓶颈理论极限**：推导传播迭代收敛映射的秩上界 $\operatorname{rank}(\mathbf{B}) \leq |\partial_O \mathcal{M}|$，从线性代数角度严格证明聚集缺失会切断信息流。与以往仅报告性能下降不同，本文给出了可计算的数学界限。
3. **提出 PEMix 结构条件化概率补全框架**：绕过观测值传播，利用全局拉普拉斯位置编码条件化高斯混合模型，在消息传递前直接建模缺失特征的联合分布。与 GCNmf 等纯特征空间方法相比，PEMix 显式引入结构位置先验以区分节点，从根本上摆脱了对局部扩散路径的依赖。
4. **系统性压力测试与消融验证**：在人工注入 RT/U-MCAR 及真实缺失数据集（AQI、PV-US、METR-LA、RelBench）上全面对比；通过控制实验证明性能提升源于“结构条件化”而非单纯的位置编码接入，论证链条完整。

## 方法详解
- **问题设定**：给定图 $G=(V,E,\mathbf{X},\mathbf{Y})$ 与缺失掩码 $\mathbf{M}$，模型接收 $(\mathbf{A}, \mathbf{X}_{\text{obs}}, \mathbf{M})$ 及训练节点标签。完整数据分布分解为 $p_\theta(\mathbf{X}|\mathbf{A}) p_\gamma(\mathbf{Y}|\mathbf{X},\mathbf{A}) p_\lambda(\mathbf{M}|\mathbf{X},\mathbf{Y},\mathbf{A})$。预测分布为 $\int p_\gamma(\mathbf{Y}|\mathbf{X},\mathbf{A}) p_\theta(\mathbf{X}_{\text{mis}}|\mathbf{X}_{\text{obs}},\mathbf{A}) d\mathbf{X}_{\text{mis}}$。
- **递归传输（RT）机制**：对目标缺失率 $\mu$，按特征通道均分缺失预算 $m_j$。每个通道从同一种子节点 $s$ 初始化，迭代选择边界邻居（优先连接已缺失集合度高的节点）扩展缺失区域，直至达到 $m_j$。若种子选择与断键独立于 $(X,Y)$，则 RT 满足 MCAR 但掩码内部强相关且依赖拓扑（Proposition 1）。
- **边界瓶颈理论**：特征传播迭代可写为 $x_\mathcal{M}^{(t+1)} = \widehat{\mathbf{A}}_{\mathcal{MM}} x_\mathcal{M}^{(t)} + \widehat{\mathbf{A}}_{\mathcal{MO}} x_\mathcal{O}$。当谱半径 $\rho(\widehat{\mathbf{A}}_{\mathcal{MM}}) < 1$ 时收敛至 $\hat{\mathbf{x}}_\mathcal{M} = (\mathbf{I} - \widehat{\mathbf{A}}_{\mathcal{MM}})^{-1} \widehat{\mathbf{A}}_{\mathcal{MO}} \mathbf{x}_\mathcal{O}$，其变换矩阵秩 $\leq |\partial_O \mathcal{M}|$（Proposition 2），即内部缺失值只能由观测边界节点的信息线性组合生成。
- **PEMix 框架**：
  1. **结构表征**：预计算拉普拉斯矩阵标准化非平凡特征向量 $\mathbf{P} = g(\mathbf{A}) \in \mathbb{R}^{n \times q}$，仅依赖图结构，一次性计算且不参与后续优化。
  2.
