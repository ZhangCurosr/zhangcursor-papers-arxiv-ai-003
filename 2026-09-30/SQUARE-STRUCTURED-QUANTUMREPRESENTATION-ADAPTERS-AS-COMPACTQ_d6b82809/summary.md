---
title: "SQUARE-STRUCTURED-QUANTUMREPRESENTATION-ADAPTERS-AS-COMPACTQ"
source: https://arxiv.org/pdf/2609.37134v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:23:05"
field: "冻结语言模型的低参适配器与结构化特征映射"
keywords: ["Parameter-Efficient Fine-Tuning", "Quantum Machine Learning", "Frozen Language Models", "Adapters", "Quadratic Feature Maps", "Amplitude Encoding", "Variational Quantum Circuits"]
innovations: ["提出 SQUARE：通过 RY-RZ-Ring-CNOT-RY 电路测量生成耦合 PSD 二次型特征适配器", "证明单一酉电路联合约束秩≤2 PSD 系数矩阵族并满足恒等分解", "实现电路仿真训练到精确批量 PyTorch 部署的完整编译路径"]
benchmarks: ["GLUE-derived Controlled Interaction Classification", "MNLI 2D Decision Boundary", "Alpaca Candidate Reranking", "OPT-350M / GPT-2 / OpenLLaMA-3B / Mistral-7B Frozen Backbones"]
---

# 论文速读：SQUARE: Structured Quantum Representation Adapters as Compact Quadratic Feature Maps for Frozen Language Models

## 一句话总结
SQUARE 是一种面向冻结语言模型瓶颈特征的适配器，通过振幅编码 + 参数化量子电路 + 测量，将低维瓶颈向量映射为规范化的二次型特征；实验表明，在 GLUE 派生的受控交互任务上，SQUARE 以仅 157 个可训练参数达到 0.7565 平均测试准确率，优于参数匹配的 Givens-12（0.7271）和 MLP（0.6817），且学习到的映射可精确编译为批量 PyTorch 部署，无需量子硬件。

## 研究问题与动机
1. **冻结 LM 下游适配的表征瓶颈问题**：冻结语言模型（如 BERT）作为固定特征提取器后，下游模块如何在小维度瓶颈 $z \in \mathbb{R}^d$ 中有效建模坐标间的交互依赖，是核心难题。
2. **线性/低秩适配器无法显式暴露二次交互**：BitFit、LoRA、Prefix 等 PEFT 方法在瓶颈阶段本质上是线性的，不直接输出 $z_i z_j$ 项，难以捕获标签对特征交互的依赖。
3. **已有二次建模方案缺乏结构化系数共享**：显式多项式展开、因子机、核方法等虽能引入二阶项，但各系数独立估计，缺少由单一结构单元联合约束系数的归纳偏置。
4. **量子特征映射的精确经典可实现性未被充分验证**：幅度编码+参数化电路测量的理论输出即为规范化的二次型，但其在冻结 LM 下游适配中的系统性实证及精确 PyTorch 编译部署路径仍存空白。

## 核心贡献（创新点）
1. **提出 SQUARE 适配器**：将冻结 LM 瓶颈向量振幅编码为量子态，经 RY–RZ–Ring-CNOT–RY 参数化电路测量，输出包含规范化二阶交互项的特征向量；与 LoRA 等线性 PEFT 的本质区别在于显式构造交叉乘积特征而非保持线性变换。
2. **建立精确二次型对应关系并刻画系数结构**：证明每个基概率特征均为瓶颈坐标的规范化二次型，系数矩阵为秩 ≤ 2 的 PSD 矩阵且联合满足恒等分解；这与无约束显式二次模型的任意对称矩阵有本质区别。
3. **实现"电路训练 → 经典部署"的完整路径**：训练使用 PennyLane 仿真器反向传播联合优化电路角度与轻量级分类头；部署时将学习到的 $U(\theta)$ 编译为等价批量复数 PyTorch 模块，最大数值偏差 $<1.12 \times 10^{-7}$，无需量子硬件。
4. **在受控交互任务上提供机制匹配的比较证据**：在 8 个 GLUE 派生任务的同一管道对比中，SQUARE（0.7565）超越参数匹配的 Givens-12（0.7271，Δ=+0.029）、仿射规范化二次预测（0.7355，Δ=+0.021）及 MLP（0.6817，Δ=+0.075），并跨 BERT/OPT-350M/GPT-2/OpenLLaMA-3B/Mistral-7B 多个冻结骨干保持定性优势。

## 方法详解
**整体框架**：冻结 LM 输出 $h = f_{LM}(x) \in \mathbb{R}^D$，固定投影 $z = P(h) \in \mathbb{R}^d$，SQUARE 在 $z$ 之后构建测量特征映射 $q_\theta(z)$，接轻量级任务头 $g_\phi$，总预测 $\hat{F}_{\theta,\phi}(z) = g_\phi(q_\theta(z))$ 或残差形式 $g_\phi([z; q_\theta(z)])$。

**（1）振幅编码**：将 $z$ 归一化后填充至 $B=2^n$ 维（$n=\lceil\log_2 d\rceil$），制备量子态 $|\psi(z)\rangle = \sum_i \frac{z_i}{\|z\|_2}|i\rangle$，保角几何但不保范数，输出对非零缩放和全局符号不变。

**（2）参数化量子电路**：默认深度 $L=1$，每层结构为 $RYRZ$–Ring-CNOT–$RY$：
$$U_\theta = \prod_{\ell=1}^L \left[ \left(\prod_{q=1}^n RY_q(\theta^{(3)}_{\ell q})\right) U_{ring} \left(\prod_{q=1}^n RZ_q(\theta^{(2)}_{\ell q}) RY_q(\theta^{(1)}_{\ell q})\right) \right]$$
环 CNOT：$\text{CNOT}_{n,1} \prod_{q=n-1}^1 \text{CNOT}_{q,q+1}$，每层 $n$ 个 CNOT。可训练量子参数总量 $|\theta_q| = 3Ln$，无额外 entanglement 参数。

**（3）测量读出**：拼接计算基概率与局部 Pauli-Z 期望：
$$p_j(z;\theta) = |\langle j|U_\theta|\psi(z)\rangle|^2, \quad \zeta_r(z;\theta) = \langle\varphi|Z_r|\varphi\rangle$$
特征向量 $q_\theta(z) = [p_0,\dots,p_{B-1}, \zeta_1,\dots,\zeta_n]$，维度 $B+n$。

**（4）测量诱导的二次型展开**：对任意 Hermitian 测量算子 $M_k$，有效算子 $A_k(\theta)=U_\theta^\dagger M_k U_\theta$ 也为 Hermitian，测得坐标：
$$\mu_k(z;\theta) = \sum_i A_{k,ii} \alpha_i^2 + 2\sum_{i<j} \text{Re}(A_{k,ij}) \alpha_i \alpha_j = \frac{z^\top A_k(\theta) z}{\|z\|_2^2}$$
其中 $\alpha_i = z_i/\|z\|_2$。概率特征对应的系数矩阵 $Q_j = a_j a_j^\top + b_j b_j^\top \succeq 0$，$\text{rank}(Q_j)\leq 2$，且 $\sum_j Q_j = I$。Pauli-Z 期望为概率的有符号线性组合，不提供额外信息。

**（5）训练**：端到端混合量子-经典优化，经典头用标准反向传播，电路角度通过 PennyLane `diff_method="backprop"` 精确求梯度；同构参数在原生 PyTorch 中亦可微分。部署时将 $U(\theta)$ 作为有序门积组装，以批量复数矩阵运算计算 $|U(\theta)\psi(z)|^2$。

**参数规模**：主实验 H16 配置（$d=16, n=4, L=1$）共 157 参：12 电路角 + 145 头参数（Lin(16→8)-tanh-Lin(8→1) 含偏置）。

## 实验与结果
**数据集**：
- GLUE 派生受控交互分类：8 个 GLUE 子任务（CoLA/SST-2/RTE/MRPC/QQP/MNLI/QNLI/WNLI），原始标签替换为基于非线性交互规则生成的二元标签（Eq. 22），冻结 BERT 瓶颈 PCA 降至 $d=16$。
- 二维决策边界诊断：MNLI 输入 + PCA-2 瓶颈 + 棋盘格/径向环等生成规则。
- Alpaca 候选重排序：120 测试查询，每查询 8 候选。
- 多骨干扩展：OPT-350M、GPT-2、OpenLLaMA-3B、Mistral-7B。

**评估基线**：BitFit、LoRA（r=4/8）、AdaLoRA、Prefix、MLP（同预算）、Givens-12（参数匹配）、仿射规范化二次预测、QAA、QPA、Frozen Circuit、显式多项式、Fourier 特征、RBF-SVM、因子机等。

**主要结果**：
- **GLUE 派生受控分类（表 3）**：SQUARE 平均测试准确率 **0.7565**，Δ vs. Givens-12 **+0.029**，Δ vs. 规范化二次 **+0.021**，Δ vs. MLP **+0.075**，Δ vs. Frozen Circuit **+0.141**。
- **二维边界任务（表 1）**：SQUARE Acc 0.7416 / F1 0.7536 / AUC **0.8259**，优于 MLP（AUC 0.7217）和 Prefix（0.6734），仅 180 参数。
- **棋盘格+径向环（表 2）**：SQUARE 平均 AUC **0.992**，优于 Classical Fourier（0.980）和 Explicit Polynomial（0.975）。
- **多骨干一致性（表 11）**：OPT-350M +0.0334、GPT-2 +0.0411、OpenLLaMA-3B +0.0662、Mistral-7B +0.0469（均 vs. MLP）。
- **Alpaca 重排序**：SQUARE Hybrid 在 OpenLLaMA-3B 上 Label Acc 0.708 / F1 0.724 / AUC 0.745；Mistral-7B 上 0.731 / 0.713 / 0.779，优于 MLP。
- **原始 GLUE 标签（表 9）**：SQUARE 49 参数 avg=0.686，与 5041 参数 MLP 持平，体现参数效率。
- **部署精度**：PennyLane ↔ 原生 PyTorch 特征最大偏差 $1.118 \times 10^{-7}$，梯度偏差 $2.1 \times 10^{-8}$。

## 相关工作脉络
1. **PEFT 方法（LoRA/BitFit/Prefix/AdaLoRA）**：在 Transformer 权重中做低秩或偏置更新；SQUARE 的定位是**不更新任何骨干权重**，仅在冻结瓶颈下游构建非线性交互映射。
2. **经典二阶特征模型（Factorization Machine、Polynomial Expansion、Bilinear Head）**：均直接参数化二次项；SQUARE 的本质区别是用**单个酉电路联合生成**一组耦合的 PSD 低秩系数矩阵，引入结构化系数共享。
3. **量子增强 NLP 适配（QAA、QPA）**：QAA 将量子模块插入激活级，QPA 用 PQC 生成 PEFT 权重；SQUARE 的直接操作对象是**投影后的低维瓶颈向量**，测量读出直接供给下游头，不涉及权重生成或全隐状态编码。
4. **Amplitude-Encoded QML 特征映射（Havlíček et al. 2019; Schuld 2021）**：理论已知幅度编码+电路测量对应低次多项式核；本文的贡献是对**特定耦合酉结构**的系统刻画及其在冻结 LM 瓶颈适配上的受控实证检验。
5. **随机特征/核近似（Rahimi & Recht 2007; Williams & Seeger 2001）**：随机傅里叶特征等近似核函数；SQUARE 在随机角度下等价于一个各向异性的二次核 Monte Carlo 近似，通过训练将随机特征升级为结构化学习特征。
6. **QuIC（Raj & Coyle 2025）**：量子启发的正交适配器，直接更新 Transformer 权重；SQUARE 与之定位不同，专注下游瓶颈特征交互建模且不触碰骨干。

## 局限性与未来方向
1. **实验范围限于受控交互标签**：主要增益来自设计用于检验交互恢复的构造标签，在自然标签 GLUE 任务上仅体现参数效率而非精度优势，尚未在更广泛自然监督下验证。
2. **瓶颈维度较低（d=16）**：主要实验使用 4 量子比特；高维场景下方阵存储复杂度 $O(B^2)$ 可能成为瓶颈， scaling 行为未系统探索。
3. **模拟器训练开销大**：当前 PennyLane 实现训练时单样本 21.3 ms，远低于专业经典适配器；原生 PyTorch 部署后前向延迟仍慢于 MLP 约 5.7×（表 18）。
4. **环 CNOT 拓扑的非唯一性**：诊断实验显示 RY-only 下 Ring 弱于 Linear CNOT，说明所得增益来自**旋转-拓扑联合设计**而非纠缠本身，通用最优拓扑仍需搜索。
5. **尺度不变性限制**：纯 SQUARE 映射对 $\|z\|$ 不变，无法建模依赖范数的任务规则；残差路径可部分缓解但不改变特征子空间本身的几何约束。

## 研究启发与可借鉴点
1. **"结构化系数共享"作为适配设计轴**：SQUARE 的核心洞察是互动系数的参数化方式（单一酉联合生成 vs. 独立估计）本身就是关键设计变量，而非仅关注参数量或多项式阶数；这一思路可迁移到经典适配器的设计，如构造低秩 PSD 因子族替代独立双线性矩阵。
2. **Circuit-native 训练 + 精确经典编译的混合范式**：用仿真器进行电路结构搜索和端到端训练，后将学习参数编译为等价经典模块部署，既保留了电路带来的归纳偏置又规避了硬件依赖；该流程可直接复用于其他基于 PQC 的特征构造模块。
3. **受控交互标签作为适配器压力测试**：用预定义非线性规则在冻结特征上生成标签，剥离了表示学习的混杂因素，是隔离评估下游模块表达能力的有效实验范式，可借鉴用于比较各类 PEFT/特征映射方法。
4. **振幅编码的尺度不变性可作为正则化先验**：输出对输入范数不变这一性质在语义向量比较场景下天然合理（余弦相似性），在需要范数信息的任务上则提示需配合残差路径或显式范数特征。
5. **参数化测量基视角**：将 $U^\dagger M U$ 理解为"学习测量基"而非"学习特征映射"，为设计紧凑可微的特征提取器提供了新的概念框架，可推广至其他酉群参数化方案（如 Givens、Householder 混合）。

## 关键术语表
**Amplitude Encoding**：将经典向量 $z$ 归一化后编码为量子态 $|\psi\rangle = \sum_i \frac{z_i}{\|z\|}\|i\rangle$，保角几何但不保范数。
**Parameterized Quantum Circuit (PQC)**：由可训练旋转角控制的量子门序列，此处指 RY–RZ–Ring-CNOT–RY 结构。
**Measured Feature Map**：对 PQC 输出态在计算基的概率分布及 Pauli-Z 期望的拼接读出，构成经典特征向量 $q_\theta(z)$。
**Normalized Quadratic Form**：测得坐标可写为 $z^\top A z / \|z\|_2^2$，其中 $A$ 为有效测量算子，每个概率特征的 $A$ 为秩 ≤ 2 PSD 矩阵。
**Givens-12**：参数匹配的对照模型，使用 12 个 Givens 旋转角代替 SQUARE 的 RYRZ-Ring-RY 角度，后续接平方特征和相同非线性头。
**Controlled Interaction Task**：保留真实 GLUE 输入分布但用预定义非线性规则覆盖原始标签，用于隔离评估下游适配模块的交互建模能力。
**Pure Path vs. Residual Path**：Pure Path 仅用 $q_\theta(z)$ 作输入；Residual Path 额外拼接原始瓶颈 $z$，可补偿尺度信息。
**PyTorch Compilation**：训练后将 $U(\theta)$ 编译为批量复数矩阵运算，精确复现测量输出，无需 PennyLane 运行时。

## 可复现要素
- **数据集**：GLUE 派生数据（内部构造，非公开原始数据但构造规则与系数已披露）；Alpaca 候选重排序数据集。
- **代码/权重**：论文附录提供 PennyLane 实现代码（Listing 1）及 PyTorch 精确编译路径描述；未提供正式开源仓库链接（匿名审稿包中包含生成器代码）。
- **关键超参**：$d=16, n=4, L=1$；学习率 $\{1\text{e-}4, 3\text{e-}4, 1\text{e-}3\}$；batch size $\{16,32\}$；AdamW + Linear Scheduler；头架构 Lin(16→8)-tanh-Lin(8→1)。
- **仿真器**：PennyLane `default.qubit`，`diff_method="backprop"`，接口 `torch`。
- **随机种子**：主实验 5 seeds，边界实验 6 seeds。
