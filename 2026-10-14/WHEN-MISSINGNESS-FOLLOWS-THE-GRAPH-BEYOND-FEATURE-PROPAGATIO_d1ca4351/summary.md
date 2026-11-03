---
title: "WHEN-MISSINGNESS-FOLLOWS-THE-GRAPH-BEYOND-FEATURE-PROPAGATIO"
source: https://arxiv.org/pdf/2610.18113v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:38:15"
field: "图表示学习与缺失数据处理"
keywords: ["缺失特征", "图神经网络", "图结构缺失", "特征补全", "高斯混合模型", "位置编码", "Recursive Transmission"]
innovations: ["形式化图结构掩码依赖性并提出RT可控生成机制", "建立特征传播的边界瓶颈理论", "提出结构条件化概率补全方法PEMix"]
benchmarks: ["TADPOLE", "GraphMSO", "PEMS-BAY", "Engrad", "AQI", "PV-US", "METR-LA", "RelBench"]
---

# 论文速读：WHEN-MISSINGNESS-FOLLOWS-THE-GRAPH-BYOND-FEATURE-PROPAGATION

## 一句话总结
论文研究了图神经网络中节点特征缺失问题，揭示了当缺失模式具有图结构依赖性（即局部故障导致成簇缺失）时，现有基于特征传播的方法会遭遇"边界瓶颈"；为此提出 PEMix，利用全局结构表征条件化概率模型来补全缺失特征，在挑战性图结构缺失场景下显著优于传播基线。

## 研究问题与动机
1. **评估协议过于乐观**：现有图缺失特征方法主要在均匀 MCAR（U-MCAR）或结构 MCAR（S-MCAR）下评估，忽略了真实场景中缺失沿图拓扑传播的模式（如传感器网络局部故障）。
2. **传播方法存在边界瓶颈**：当缺失节点在图结构中成簇聚集时，观察边界节点数量有限，传播方法的恢复能力受限于边界规模，理论上证明其恢复矩阵的秩不超过观察边界节点数。
3. **缺失机制分类的盲区**：现有 MCAR/MAR/MNAR 分类仅刻画掩码与数据的条件依赖关系，未考虑掩码内部条目之间的图结构依赖性（即缺失模式的空间组织）。
4. **图结构信息未被充分利用**：现有方法要么忽略图结构（统计填充），要么仅用图结构传播观察值，未将全局结构信息直接用于条件化缺失特征的概率分布。

## 核心贡献（创新点）
1. **形式化图结构掩码依赖性与 RT 机制**：定义图结构掩码依赖性，提出递归传输（Recursive Transmission, RT）可控生成机制，模拟从共享种子节点出发沿边传播的局部故障模式。
2. **建立特征传播的边界瓶颈理论**：证明 Feature Propagation 的恢复矩阵秩受限于观察边界节点数，解释为何成簇缺失对传播方法构成根本性挑战。
3. **提出 PEMix 结构条件化概率补全方法**：不通过传播观察值补全缺失特征，而是用全局位置编码条件化高斯混合模型，在消息传递前建模缺失特征分布，突破边界瓶颈。
4. **系统性实验验证**：在受控 RT/U-MCAR 设置和含自然缺失的真实数据集（AQI、PV-US、METR-LA、RelBench）上验证，PEMix 在图结构缺失下提升显著，尤其在缺失率较高时优势更突出。

## 方法详解
**整体框架**：PEMix 保留图结构完全可观测的假设，利用 Laplacian 特征向量作为全局位置编码，条件化一个高斯混合模型（GMM）来建模不完整节点特征。

**位置编码计算**：
- 从完全观测的图 $G=(V,E)$ 计算标准化非平凡 Laplacian 的前 $q$ 个特征向量，得到结构表征矩阵 $\mathbf{P} = g(\mathbf{A}) \in \mathbb{R}^{n \times q}$。
- 该编码一次性预计算，不参与 GNN 训练梯度更新。

**结构条件化混合模型**：
- 每个节点 $i$ 有一个隐式混合成分 $c_i \in \{1,\dots,K\}$，条件于 $c_i=k$ 时，特征和位置分别建模为独立对角高斯分布：$q_\eta(\mathbf{x}_i|c_i=k) = \mathcal{N}(\boldsymbol{\mu}_k^X, \text{diag}((\boldsymbol{\sigma}_k^X)^2))$，$q_\eta(\mathbf{p}_i|c_i=k) = \mathcal{N}(\boldsymbol{\mu}_k^P, \text{diag}((\boldsymbol{\sigma}_k^P)^2))$。
- 混合责任更新公式（同时利用观测特征和位置编码）：
$$\gamma_{ik} \propto \pi_k \cdot q_\eta(\mathbf{p}_i|c_i=k) \cdot \prod_{j \in \Omega_i} \mathcal{N}(X_{ij}|\mu_{kj}^X, (\sigma_{kj}^X)^2)$$
- 其中 $\Omega_i$ 为节点 $i$ 的观测特征坐标集合。与 GCNmf 的本质区别：PEMix 的起始先验 $\pi_k(\mathbf{p}_i)$ 因节点位置不同而异，而 GCNmf 所有节点从相同先验出发。

**缺失特征补全**：
- 在独立对角假设下，缺失坐标的条件分布为：
$$q_\eta(\mathbf{x}_{i,\bar{\Omega}_i}|\mathbf{x}_{i,\Omega_i}, \mathbf{p}_i) = \sum_{k=1}^K \gamma_{ik} \prod_{j \in \bar{\Omega}_i} \mathcal{N}(X_{ij}|\mu_{kj}^X, (\sigma_{kj}^X)^2)$$
- 保留完整条件不确定性，通过解析边缘化第一层消息传递中的分量矩来融合概率估计。

**端到端训练**：
- 用 EM 在均值填充的扩展矩阵上初始化 GMM 参数，随后与下游 GNN 联合优化（cross-entropy loss）。
- 位置编码仅影响混合责任，不拼接至节点表示；复杂度与 GCNmf 同阶，额外开销为 $O(nKq)$。

## 实验与结果
**数据集与设置**：
- 受控实验：TADPOLE（分类）、GraphMSO/PEMS-BAY/Engrad（预测），注入 RT 和 U-MCAR，$\mu \in \{0, 0.1, \dots, 0.99\}$。
- 自然缺失实验：AQI（25.67%缺失）、PV-US（54.02%）、METR-LA（8.11%）、RelBench 六个任务。
- 基线：GNN-Zero/Median/MI、GCNmf、GNN-MIM、FP、PCFI、FISF。

**RQ1 结果（受控缺失）**：
- RT 下 PEMix 在 TADPOLE、GraphMSO、PEMS-BAY 三项均排名第一（Table 1）：
  - TADPOLE Macro-F1 AUC：**0.8243**（FP=0.7254，提升约 13.5%）
  - PEMS-BAY MAE AUC：**3.7645**（GCNmf=3.9219，降低约 4.0%）
  - GraphMSO MAE AUC：**1.3688**（GCNmf=1.3690，基本持平）
- U-MCAR 下 PEMix 同样具竞争力（TADPOLE 0.8094 第一，PEMS-BAY 0.3464 第二）。
- 传播方法平均 AUC：RT=0.725，U-MCAR=0.754，S-MCAR=0.751，RT 较 U-MCAR 下降约 4%。

**RQ2 结果（自然缺失）**：
- AQI MAE：PEMix **18.73 ± 0.05**（最优）；METR-LA MAE：**6.45 ± 0.01**（最优）；PV-US MAE：12.02（第二，GCNmf=11.99）。
- RelBench：RDL-PEMix 在 6 个任务中 4 个最优（Table 3）。

**RQ3 消融**：
- PE 仅用于下游或作为输入（PE-only）均远逊于 PEMix（TADPOLE AUC：PEMix=0.824 vs PE-only=0.292）。
- 移除结构条件（设 PE=0）后 AUC 降至 0.733，差距在高缺失率时扩大至 0.49。
- 联合优化 GMM 优于冻结 GMM（Table 40），证实端到端训练的重要性。

## 相关工作脉络
1. **Feature Propagation (FP)** (Rossi et al., 2022)：最早的图特征传播补全方法，通过 Dirichlet 能量最小化扩散观察值；PEMix 不依赖此传播范式，绕过边界瓶颈。
2. **PCFI / FISF** (Um et al., 2023, 2025)：FP 的增强版，引入置信度/迭代注入策略；实验中同样受图结构缺失影响，PEMix 在 RT 下显著超越。
3. **GCNmf** (Taguchi et al., 2021)：使用高斯混合建模特征分布并用条件矩补全缺失，但不利用图拓扑进行特征恢复；PEMix 在 GCNmf 基础上引入位置编码条件化，是本质升级。
4. **GNN-MIM** (Ferrini et al., 2026)：将缺失掩码拼接至特征，属非拓扑感知方法；PEMix 明确建模掩码的图结构依赖性并针对性设计。
5. **TADPOLE 基准** (Ferrini et al., 2026)：被推荐为缺失特征鲁棒性评估基准；本文在此基础上引入 RT 扩展评估维度。
6. **RelBench** (Robinson et al., 2024)：关系数据库深度学习基准；本文验证 PEMix 可适配至 RDL 架构（RDL-PEMix）。

## 局限性与未来方向
1. **对角协方差假设**：GMM 采用对角协方差限制了特征间相关性的建模能力，全文认此为模型主要局限。
2. **完全可观测图结构**：假设邻接矩阵 $\mathbf{A}$ 完全已知，未考虑边缺失场景。
3. **任务范围有限**：仅在节点分类和时空预测上验证，未涉及边预测、图分类等其他任务。
4. **时序建模缺失**：预测实验将时间步转化为独立静态图样本，未引入显式时序建模。
5. **RT 机制的普适性**：RT 是受控压力测试而非真实缺失的生成模型，未来需探索更多样化的图结构缺失机制。

## 研究启发与可借鉴点
1. **缺失模式评估的新维度**：将掩码的内部依赖性与图结构关联纳入评估体系，RT 可作为通用基准机制推广至其他图学习方法研究。
2. **结构信息用于补全的先验条件化**：用全局位置编码而非局部传播来条件化缺失特征分布，是一种突破局部信息瓶颈的有效范式，可迁移至其他补全场景。
3. **边界瓶颈理论的通用性**：传播方法的恢复秩受边界节点数约束这一结论，可启发对其他基于扩散的图学习方法鲁棒性分析。
4. **联合优化 GMM 的设计选择**：EM 初始化+端到端联合训练优于固定 GMM，这一训练策略可复用于其他概率补全模型。
5. **成簇缺失与位置编码的结合思路**：当缺失呈现空间聚集时，位置编码能有效区分不同结构位置的节点，这一洞察对工业场景（传感器网络、电网故障）有直接参考价值。

## 关键术语表
**Graph-structured mask dependence**：缺失掩码的内部联合分布由图拓扑塑造，使得相邻节点的缺失状态相关，区别于独立均匀缺失。
**Recursive Transmission (RT)**：从共享种子节点出发，沿图边递归扩展缺失区域的结构化缺失生成机制，模拟局部故障传播。
**Boundary bottleneck**：特征传播方法的恢复矩阵秩受限于缺失区域观察边界节点数，成簇缺失时信息注入通道狭窄。
**PEMix**：利用全局 Laplacian 位置编码条件化高斯混合模型的结构条件化缺失特征补全方法。
**Structure-conditioned mixture**：混合成分的起始先验由节点位置编码决定，使不同结构位置的节点获得差异化初始责任。
**Feature-MAR / Label-MAR / MCAR / MNAR**：描述缺失机制与数据变量条件依赖关系的分类体系（Rubin, 1976）。
**Excess co-missingness ratio**：衡量相邻节点同时缺失比例超出空间独立假设程度的指标，用于量化真实数据集的图结构缺失程度。
**Trapezoidal AUC**：对缺失率曲线下的面积用梯形法则数值积分，作为综合评估指标。

## 可复现要素
- **数据集**：TADPOLE、GraphMSO、PEMS-BAY、Engrad、AQI、PV-US、METR-LA、RelBench（均为公开数据集，论文 Appendix B 提供详细来源与预处理说明）。
- **代码开源**：论文未明确声明代码开源仓库，但提供了完整的超参数表（Tables 26-38）和复杂度分析（Appendix E）。
- **关键超参**：PE 维度 $q \in \{4, 8, 16\}$，混合成分数 $K \in \{2, 3, 5, 8\}$，GNN hidden dim=16/32/64/128，学习率 0.001-0.01，dropout 0-0.5。
- **实验细节**：五个随机种子，500-1000 最大训练轮次，patience 40-50，验证集选模，未见测试集。
