---
title: "PROBABILITY-SIGNATURE-DYNAMICS-UNPACKING-MODULAR-ADDITION-LE"
source: https://arxiv.org/pdf/2610.11833v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:17:11"
field: "机械可解释性与训练动力学"
keywords: ["概率签名", "模加学习", "Fourier电路", "Grokking", "训练动力学", "标签噪声"]
innovations: ["从训练分布前向推导Fourier基涌现机制，无需预设Fourier结构", "提出概率签名算子共同特征向量框架预测网络表示基", "统一解释早期噪声加速损失下降与Fourier频率选择机制"]
benchmarks: ["Modular Addition (P=113)", "XOR (64-bit)"]
---

# 论文速读：PROBABILITY-SIGNATURE-DYNAMICS-UNPACKING-MODULAR-ADDITION-LE

## 一句话总结
本文利用概率签名（Probability Signature）框架，从训练数据分布出发**前向推导**了两层网络在模加任务中涌现 Fourier 结构化表示的机制，解释了频率匹配、相位对齐等现象的形成原因，并统一解释了早期随机标签噪声加速损失下降的悖论，该方法可推广至 XOR 等运算任务以**预测**网络将发展的表示基。

## 研究问题与动机
- **核心问题**：已有研究已从后验角度确认训练后的网络会发展出稀疏 Fourier 电路（Fourier circuits），并验证其支持泛化；但**缺少前向解释**——梯度下降为何能从模加数据分布中自然选择这些 Fourier 模式。
- **现有不足**：He et al. (2026) 等工作虽已解释频率竞争和相位对齐，但其理论分析**预设了 Fourier 基**与模加的关联；本文希望不依赖这一先验知识，从数据分布的统计结构出发推导 Fourier 基的出现。
- **噪声悖论**：在含随机标签噪声的训练中，噪声样本的早期损失下降速度**快于**干净样本，这一现象已被观察到（Liu & Lu, 2026）但缺乏机理解释。
- **一般化需求**：如何在不针对特定任务预设基函数的情况下，从训练分布推导出网络参数将形成的结构？

## 核心贡献（创新点）
- **从数据分布前向推导 Fourier 结构**：证明模加任务的条件概率签名算子是循环移位算子，其共同特征向量恰好是 DFT 基，从而无需预设 Fourier 基即可解释频率模式的涌现。
- **揭示权重衰减的滤波机制**：提出权重衰减作为统一的"生存阈值"，通过比较各频率模式的爆破时间（blow-up time）与 $C/\lambda$ 的关系，解释哪些神经元能发展出主导频率。
- **统一解释早期噪声加速现象**：从概率签名出发推导定理 3，证明随机噪声增加了条件标签碰撞（conditional label collisions），强化了共享坐标方向的梯度驱动，从而解释噪声样本早期损失下降更快。
- **框架可迁移至其他运算**：将方法推广至 XOR 任务，证明联合概率签名算子的共同特征基为 Walsh–Hadamard 基，并在实验中观测到预测的 Hadamard 频率结构。

## 方法详解
- **概率签名定义**：基于训练分布 $\pi$ 定义三类条件统计量：
  - 一阶签名：$\varphi^{(a)}(x,z) = \mathbb{P}(Z=z \mid X=x)$，$\varphi^{(b)}(x,z) = \mathbb{P}(Z=z \mid Y=x)$
  - 二阶联合签名：$\Phi^{(a)}(x,y,z) = \mathbb{P}(Z=z, Y=y \mid X=x)$，$\Phi^{(b)}(x,y,z) = \mathbb{P}(Z=z, X=y \mid Y=x)$
  - 回溯签名：$\psi^{(a)}(x,y) = \mathbb{P}(Y=y \mid X=x)$，$\psi^{(b)}(x,y) = \mathbb{P}(X=y \mid Y=x)$
- **梯度流主导项推导**（Proposition 1 & Theorem 1）：在 small initialization 假设（$\|U\|_\infty, \|V\|_\infty, \|W\|_\infty \leq \epsilon$）下，对激活函数做二阶 Taylor 展开，得到参数梯度的主导项仅依赖于上述概率签名，高阶项为 $O(\epsilon^3)$。
- **Fourier 基的自然涌现**：对于完整模加数据集，$\Phi_x^+$ 构成循环移位算子族 $S_x$，满足交换性，其共同特征向量为 DFT 基 $F_r(s) = P^{-1/2} e^{-2\pi i rs/P}$。在此基下，各频率 $r \neq 0$ 的动力学方程**近似解耦**：
  $$\dot{\hat{u}}_r \approx C \hat{w}_r \overline{\hat{v}}_r - \lambda \hat{u}_r, \quad \dot{\hat{v}}_r \approx C \hat{w}_r \overline{\hat{u}}_r - \lambda \hat{v}_r, \quad \dot{\hat{w}}_r \approx C \hat{u}_r \hat{v}_r - \lambda \hat{w}_r$$
- **相位对齐机制**（Proposition 2）：对无衰减系统，相位差 $\delta = \arg u_r + \arg v_r - \arg w_r$ 满足 $\dot{\delta} = -C \sin\delta \cdot (\text{正定因子})$，故 $\delta=0$ 是稳定平衡点，训练自然推动相位对齐。
- **噪声估计**（Theorem 2）：对于 80% 数据 + $\alpha$ 比例随机噪声的训练集，梯度主项被缩放因子 $(1-\alpha)$ 修正，采样误差方差上界为 $\mathcal{O}(\frac{[(1-\rho)+\alpha]\eta^2}{\rho P^3}(c_1^2\epsilon^2 + c_2^2\epsilon^4))$。

## 实验与结果
- **设置**：模加任务 $P=113$（素数），两层 MLP 宽度 $D=2000$，输入 one-hot 编码，激活函数 GELU，训练集占 80%，噪声比 $\alpha=0.1$，初始化尺度 $\gamma=0.5$。
- **权重衰减选择效应**：$\lambda=0$ 时模型过拟合（测试精度 0%）；$\lambda=10^{-4}$ 时实现泛化（测试精度接近 100%）；$\lambda=5\times10^{-4}$ 时模型忽略噪声；$\lambda=10^{-3}$ 时无法收敛。
- **频率稀疏性与范数增长**：高稀疏性（$\rho(u)>0.7$）神经元集中了几乎所有范数增长，其 CE-driv e 项主要由干净数据贡献，与噪声无关（Figure 2）。
- **频率匹配与相位对齐**：主导频率一致的神经元占比高，且 $\arg(u_d)+\arg(v_d) \approx \arg(w_d) \pmod{2\pi}$（Figure 3）。
- **梯度近似验证**：理论估计梯度与实际梯度的 cosine similarity $\geq 0.9$（Appendix C.1）。
- **Full-batch GD 对比**：截断模型预测的 dominant frequency 与实际 GELU 网络收敛结果的 agreement rate 为 93%。
- **XOR 实验**：在 $\Omega=\{0,\dots,63\}$ 上训练 XOR 任务，观测到高稀疏性神经元的 dominant frequency 集中在 Hadamard 基方向，验证了理论预测（Figure 5）。
- **噪声加速悖论验证**：不同 $\alpha$ 下噪声样本的早期损失始终下降更快（Figure 4），与定理 3 的理论预测一致。

## 相关工作脉络
- **Grokking 机制**（Power et al., 2022; Liu et al., 2022, 2023）：研究泛化延迟出现的优化轨迹特性；本文从权重衰减阈值角度建立其与 Fourier 结构形成的联系。
- **模加 Fourier 电路**（Nanda et al., 2023; Gromov, 2023）：后验诊断已训练的神经网络的 Fourier 结构；本文提供前向解释，说明为何训练 dynamics 会自然选择这些结构。
- **Fourier 基分解权重解释频率竞争**（He et al., 2026）：需预设 Fourier 基；本文从概率签名算子的共同特征向量**推导**出 Fourier 基，无需先验假设。
- **概率签名框架**（Yao & Xu, 2025）：本文的基础工具，将数据条件统计与嵌入结构关联；本文将其首次系统应用于算法算术任务。
- **带噪声标签学习**（Doshi et al., 2024; Liu & Lu, 2026）：研究泛化与记忆共存的鲁棒性；本文聚焦于早期记忆阶段噪声加速损失下降的具体机制。

## 局限性与未来方向
- **近似适用范围有限**：理论推导基于 small initialization 假设和 Taylor 展开截断，仅覆盖训练的早期/主导梯度 regime，未覆盖整个训练过程的动态演化。
- **有限样本误差**：实际训练使用采样子集（80% 数据），采样误差随 $\rho$ 减小和 $\alpha$ 增大而放大（Theorem 2）。
- **激活函数依赖二阶项**：线性/Tanh 激活因缺少二阶导数项而无法产生所需的频率选择效应（Appendix B）；ReLU 虽非光滑但仍可通过分段线性分析获得类似行为。
- **未来方向**：扩展至更大规模网络架构、更复杂的代数运算（如模乘）、以及连续域上的类似分析。

## 研究启发与可借鉴点
- **概率签名作为数据-表示桥梁**：将训练分布的条件统计量直接映射到梯度主导项的方法，可为其他结构化学习任务（如关系推理、符号运算）提供分析工具。
- **"算子共同特征向量"预测表示基的思路**：不预设基函数，而是从数据分布构造线性算子族，通过求解共同特征向量来**预测**网络将发展的表示坐标系，这一思路可迁移至其他群结构任务。
- **噪声加速现象的分析框架**：通过统计共享坐标的样本对数量（$K_0, K_1$）来量化不同标签分布下的早期损失下降速率，方法简洁且具可计算性，可推广至其他噪声类型分析。
- **实验验证梯度近似的 cosine similarity 方案**：用理论估计梯度与实际梯度的余弦相似度检验近似有效性，可作为后续工作的标准验证手段。

## 关键术语表
- **Probability Signature（概率签名）**：从训练分布推导的条件概率统计量（一阶 $\varphi$、二阶 $\Phi$、回溯 $\psi$），用于刻画梯度主导项的数据依赖结构。
- **Grokking（顿悟/渐进泛化）**：模型在训练损失已降至极低后，测试精度仍长时间停滞，随后突然跃升的现象。
- **Fourier Circuit（Fourier 电路）**：模加任务中 trained 网络发展出的稀疏表示，其输入/输出权重集中在匹配的 Fourier 频率上。
- **Frequency Matching（频率匹配）**：同一神经元的 $U_d, V_d, W^d$ 在 Fourier 域中主导频率趋于一致的现象。
- **Phase Alignment（相位对齐）**：匹配频率下 $\arg(u_d)+\arg(v_d)\approx\arg(w_d)\pmod{2\pi}$ 的相位锁定关系。
- **Weight Decay（权重衰减）**：$L^2$ 正则化项，在本文中充当频率选择的生存阈值，抑制低频/低幅度模式而允许高频主导模式增长。
- **Blow-up Time（爆破时间）**：无衰减系统中某频率模式幅度发散至无穷所需的时间，决定该模式是否能在权重衰减作用下胜出。
- **Hadamard Basis（Hadamard 基）**：XOR 任务中联合概率签名算子的共同特征基，对应 Walsh 字符的离散正交基。

## 可复现要素
- **数据集**：模加任务 $\Omega=\{0,\dots,P-1\}$，$P=113$（素数），完整加法表划分为 clean/noise/test 三部分（比例 80%/10%/10% 中的 80% 用于训练）；XOR 任务 $\Omega=\{0,\dots,63\}$。**论文未明确说明公开情况**。
- **代码/权重**：**论文未提及开源**。
- **关键超参**：网络宽度 $D=2000$，隐藏层宽度隐含于矩阵维度；权重衰减 $\lambda \in \{0, 10^{-4}, 5\times10^{-4}, 10^{-3}\}$；初始化尺度 $\gamma=0.5$（即 $U_{ij},V_{ij},W_{ij}\sim\mathcal{N}(0, D^{-2\gamma})$）；激活函数 GELU；Adam 优化器，初始学习率 $\eta=10^{-3}$，warmup+cosine decay，$\eta_0=10^{-5}$；噪声比 $\alpha=0.1$。
