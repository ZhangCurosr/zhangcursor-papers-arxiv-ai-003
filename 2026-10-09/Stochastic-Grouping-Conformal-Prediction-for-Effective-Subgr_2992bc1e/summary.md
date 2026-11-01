---
title: "Stochastic-Grouping-Conformal-Prediction-for-Effective-Subgr"
source: https://arxiv.org/pdf/2610.11957v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:28:09"
field: "公平共形预测与子群体可靠性"
keywords: ["conformal prediction", "subgroup fairness", "uncertainty quantification", "calibration", "covertage gap"]
innovations: ["提出基于随机分组映射的局部 score law 学习框架，避免显式敏感属性与最差组瓶颈", "理论上证明 SGCP 保留 split conformal 边际有效性，并给出子群体覆盖率下界的 Kolmogorov 距离控制"]
benchmarks: ["SYN synthetic cardiopulmonary diagnosis", "Nursery UCI", "MIMIC-IV in-hospital mortality and ICU delay", "BACH breast cancer histology"]
---

# 论文速读：Stochastic-Grouping-Conformal-Prediction-for-Effective-Subgr

## 一句话总结
本文提出随机分组共形预测（SGCP），通过从校准数据中学习一个随机分组映射，使每个样本能够共享具有相似校准行为的参考样本的证据，从而在无需预设敏感属性的情况下提升子群体的覆盖可靠性，同时保持标准共形预测的边际有效性保证与高效的预测集大小。

## 研究问题与动机
- **子群体覆盖不均**：标准共形预测（CP）仅在总体层面提供覆盖率保证（1−α），在临床等公平敏感场景中，某些人口统计子群或疾病亚型的覆盖率可能显著低于名义水平，被其他子群的超额覆盖所"平均掉"。
- **直接按组校准存在两大缺陷**：①相关分组往往不可预先观测，且不一定与敏感属性对齐；②将校准集细分到各组会导致某些组样本过少，经验分位数估计不稳定（最差组瓶颈）。
- **局部校准同样面临保守性困境**：如 RLCP 等局部化方法虽能改善子群覆盖，但预测集往往显著扩大，且跨随机种子的稳定性较差，增加决策者的认知负担。
- **异质性是潜藏的**：对校准至关重要的样本间差异反映在校准行为（非 conformity score 分布）上，而非简单地体现在特征空间几何邻近性上，因此需要一种能在"全局校准"和"过度局部化校准"之间取得平衡的新机制。

## 核心贡献（创新点）
1. **提出 SGCP 框架**：通过学习基于校准行为相似度的随机分组映射，使每个样本获得一个样本自适应的局部 score law，替代不可行的逐点无条件条件分布估计——与已有离散分组或最近邻局部方法不同，本方法通过共享潜在校准分量实现结构化证据借用。
2. **理论证明边际有效性保留**：在标准 split-conformal exchangeability 假设下，SGCP 仍满足 P{Y_{n+1} ∈ Ĉ_α(X_{n+1})} ≥ 1−α，且该保证不依赖局部 score law 的学习精度。
3. **给出子群体覆盖差距的理论界**：Proposition 3 表明，子群体覆盖率下界由局部 score law 变换后 transformed-score 分布在子群与全局之间的 Kolmogorov 距离 Δ_G 控制，揭示了提高 subgroup reliability 的原理路径。
4. **设计三目标训练损失**：rank-uniformization loss（使各分量内 transformed score 接近 Uniform(0,1)）、负对数似然密度拟合 loss（令混合密度忠实于观测 score）、以及样本-分量间归一化互信息最大化（防止分量坍缩或均匀分配退化）。

## 方法详解
**整体流程**：将训练数据 D_tra 拆为 D_0 和 D_1，D_0 用于训练骨干分类器 f_θ、学习潜在校准状态后验 q_φ(z|x) 和随机分组映射 w(x)；D_1 用于拟合 K 个潜在校准分量 {F_k} 与聚合局部 score law F̂(s|x)；独立的校准集 D_cal 用于 split conformal 阈值计算。

- **潜在校准状态（Latent calibration state）**：对每个样本 x，通过骨干特征 h_θ(x) 建模一个高斯潜在后验 q_φ(z|x) = N(μ_φ(h_θ(x)), diag(σ²_φ(h_θ(x))))，允许相同观测特征对应不同的潜在校准行为。
- **随机分组映射（Stochastic grouping map）**：对 z ~ q_φ(z|x) 采样 M 次，经可微映射得到 K 维 simplex 向量 ã(x,z)，再对后验期望做 Monte Carlo 近似：w_k(x) ≈ (1/M) Σ_m ã_k(x, z^(m))，从而得到样本 x 在 K 个潜在分量上的平滑权重。
- **聚合局部 score law**：F̂(s|x) = Σ_k w_k(x) F_k(s)，其中每个 F_k 是单调可微映射（代表一种反复出现的校准模式）。经此变换后的 score V(x,y)=F̂(s(x,y)|x) 在 oracle 情形下服从 Uniform(0,1)，使不同子群的 transformed score 可比。
- **训练损失**（Eq. 9）：L_SGCP = L_rank + λ L_nll − β Î_norm(X;Z)，其中 Î_norm 为样本与分量间的归一化互信息，促进分量的多样性使用和样本特异性权重的保持。
- **Calibration 与预测**：对 D_cal 中样本计算 transformed score V_i = F̂(s_i|X_i)，排序取分位数作为阈值 q̂_{1−α}；测试时对候选标签 y 计算 V^y_{n+1}，保留满足 V^y_{n+1} ≤ q̂_{1−α} 的标签构成预测集。

## 实验与结果
**数据集**：四个基准——合成 cardiopulmonary 诊断（SYN，6类，含交叉异质性偏置子群）、Nursery（4类，模拟算法偏差）、MIMIC-IV（医院死亡率 2 类 + ICU 延迟 3 类）、BACH 乳腺癌病理图像（4 类，以类别为代理分组变量）。敏感属性仅用于公平评估，不作为模型输入。

**基线**：Marginal CP、Partial Equalized、Exhaustive Equalized、RLCP、CluCP、AFCP、FaReG。

**主要结果（Table 1，1−α=0.90 最大样本量）**：
- **SYN**：SGCP Cov=90.13±0.39%，CovGap=4.2±0.7%（最优），AvgSize=2.225±0.035（最优，6 类一半为 3）。
- **Nursery**：SGCP Cov=91.65±0.70%，CovGap=11.8±0.7%（最优），AvgSize=1.089±0.035（最优）。
- **MIMIC-IV**：SGCP Cov=90.13±0.07%（最优），CovGap=0.8±0.0%（仅次于 FaReG 的 0.5），AvgSize=1.107±0.001（最优）。
- **BACH**：SGCP Cov=91.90±2.70%（最优），CovGap=6.2±1.1%（最优），AvgSize=2.224±0.215（最优，4 类一半为 2）。

**核心结论**：SGCP 是唯一在所有四个数据集上同时保持接近名义覆盖率、最小 CovGap 且预测集大小最小或接近最小的方法。在稀疏样本场景（N=500~2000）下优势更明显，基线方法趋向于极端保守或覆盖不足，而 SGCP 退化更平稳。消融实验（Table A4）证实去除随机性、rank loss、NLL 项或互信息项均会损害性能。

## 相关工作脉络
- **Group-wise CP（CluCP、AFCP、FaReG、Partial/Exhaustive Equalized）**：这类方法依赖显式分组（先验敏感属性或数据驱动聚类/发现），SGCP 与它们本质区别在于：不需要预定义分组，而是通过潜在校准分量实现每样本自适应的软混合，避免了对敏感属性的依赖与最差组瓶颈。
- **Localized CP（RLCP）**：RLCP 基于特征空间邻域加权，SGCP 以"校准行为相似"为 locality 定义，且在理论保证与稳定性上更优（消融显示特征空间分组比潜在校准状态性能差）。
- **Class-conditional CP（CluCP）**：CluCP 仅在类别层面做结构化共享，无法捕捉类内异质性；SGCP 的潜在分量可进一步细化同一类别内的不同校准模式。
- **SAGCCI [11]**：同样利用 score 分布相似度共享校准信息，但 SAGCCI 在观测到的受保护组间聚类并借助 surrogate outcomes，SGCP 则在样本层面学习无监督的潜在分量混合，更细粒度且不依赖 surrogate。
- **Kandinsky CP [3] 与 homogeneous-group 发现 [18]**：均以函数族或显式分区定义条件校准对象，SGCP 选择"soft mixture over latent components + 标准 split conformal on transformed scores" 的互补路径，避免了严格分组带来的保守性与不稳定性。

## 局限性与未来方向
- 所有实验均为回顾性基准验证，尚需前瞻性临床工作流中的实地评估。
- 当前框架仅面向多类分类任务，尚未扩展到回归或其他预测设置。
- 潜在校准分量的数目 K 需要折中选取：过小时表达力不足，过大时分量易重叠冗余；实践中需结合任务特性调参（尽管超参敏感性实验表明整体对超参相对稳健）。
- 临床部署场景下的计算/数据约束（如 D_0/D_1 划分比例对性能的影响）有待进一步系统研究。

## 研究启发与可借鉴点
- **以校准行为而非特征几何定义 locality**：这一思路可迁移至任何需要对齐误差结构而非输入空间的不确定性量化场景（如自适应阈值、误差校准）。
- **后验平均稳定软分组**：将离散 hard assignment 替换为高斯后验采样的 Monte Carlo 平均，能有效抑制单点校准的波动，可作为不确定分组设计的通用技巧。
- **三目标损失的结构**：rank-uniformization + 密度拟合 + 互信息最大化这一组合兼顾了校准可比性、真实性与分组信息量，适用于类似混合分布学习问题。
- **对 team 的结合机会**：本方法与团队现有分类/医学影像工作可直接嫁接——在已有 backbone 上接入 SGCP 校准模块，即可在不重训模型的前提下获得子群友好的不确定性输出；此外，将潜在校准状态扩展至回归或半监督场景是自然的研究延伸。

## 关键术语表
**Conformal Prediction（共形预测）**：一种在弱假设（exchangeability）下为任意预测模型提供有限样本覆盖率保证的框架，通过非 conformity score 和经验分位数构造预测集。
**Nonconformity Score（非一致分数）**：衡量样本 (x,y) 与模型预测不一致程度的标量，常用形式为 1−f_θ^y(x)。
**Local Score Law（局部 score law）**：给定样本 x 条件下的非 conformity score 分布 F_x^⋆，是精准子群校准的"oracle"目标。
**Stochastic Grouping Map（随机分组映射）**：通过潜在后验采样后平均得到的样本到 K 个潜在分量的软权重 w(x) ∈ Δ^{K−1}。
**Latent Calibration Component（潜在校准分量）**：K 个单调可微映射 F_k，各自捕获一种反复出现的校准模式，样本的局部 score law 由它们的加权和构成。
**CovGap（覆盖差距）**：各子群覆盖率偏离名义水平 1−α 的平均绝对偏差，用于量化子群体可靠性。
**Rank-Uniformization Loss（秩统一化损失）**：促使每个分量内 transformed score 的加权分布接近 Uniform(0,1) 的训练目标。
**Split Conformal（分裂共形）**：将数据划分为训练集和独立校准集，在 calibration set 上计算分位数阈值，在 test 时应用的标准共形协议。

## 可复现要素
- **数据集**：SYN（代码内程序化生成）、Nursery（UCI 公开）、MIMIC-IV（PhysioNet 需 credential 访问）、BACH（Zenodo 公开）。
- **代码**：SGCP 源码匿名开源（https://anonymous.4open.science/r/sgcp/）；基线代码均来自各论文公开仓库（Table A7）。
- **关键超参**：K（潜在分量数，SYN=3, Nursery=6, MIMIC-IV=3, BACH=6）、d_z（16/8/8/64）、M_train/M_eval（3~16）、λ（0.001~0.5）、β（5×10⁻⁵~0.5）、SGCP 训练轮次 100、骨干训练 40~200 轮、batch size 32/128、SGCP lr 10⁻³~10⁻²、weight decay 5×10⁻⁴（详见 Table A3）。
- **硬件**：单卡 NVIDIA RTX 4090 D，PyTorch 实现，每项实验 20 次随机种子重复。
