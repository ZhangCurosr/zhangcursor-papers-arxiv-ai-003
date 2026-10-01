---
title: "RICCATI-STATE-SPACE-MODELS-NON-ITERATIVE-PARALLELIZATION-FOR"
source: https://arxiv.org/pdf/2609.35441v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:08:56"
field: "长序列状态空间模型"
keywords: ["State Space Models", "Parallel Scan", "Riccati Equation", "Möbius Transformation", "Nonlinear Recurrence", "Liquid Dynamics", "Long Sequence Modeling"]
innovations: ["构造输入条件化Riccati动力学使离散流映射为Möbius变换，实现单次并行扫描的非迭代精确求值", "推导约束参数化保证有界不变区间与收缩性，避免极点发散", "建立与LRC动力学的局部Taylor对应并证明精确ZOH积分优于显式Euler离散"]
benchmarks: ["UEA Time Series Classification (6 datasets)", "PPG-DaLiA Regression", "Weather Forecasting"]
---

# 论文速读：RICCATI-STATE-SPACE-MODELS-NON-ITERATIVE-PARALLELIZATION-FOR

## 一句话总结
本文提出 RiccatiSSM，一种非线性状态空间模型，通过将每个状态维度建模为输入条件化的 Riccati 微分方程，使其离散时间流映射恰好为 Möbius 变换；由于 Möbius 变换在 $2 \times 2$ 矩阵乘法下闭合，整个序列可用单次并行前缀扫描精确求值，无需迭代线性化，相比 LrcSSM 端到端运行时间降低 22%–33%。

## 研究问题与动机
- **线性 SSM 的局限**：现有 SSM（如 S4/S6/Mamba）的状态更新是仿射的（$x_t = \Lambda_t x_{t-1} + b_t$），虽支持单次并行扫描，但局部 Jacobian 不依赖当前状态，无法表达状态依赖的非线性动态。
- **一般非线性模型代价高**：DEER、LrcSSM 等通过 Newton 迭代逐次线性化非线性递归来恢复并行性，需 $K$ 次扫描（$K$ 由数据决定），增加计算复杂度和收敛不确定性。
- **核心科学问题**：是否存在一类非线性动态，本身在 composition 下严格闭合，从而允许单次并行扫描精确求值？
- **设计目标**：在保持液体模型（liquid recurrent）核心特性——状态依赖局部收缩率的同时，实现非迭代的精确并行评估。

## 核心贡献（创新点）
1. **引入 RiccatiSSM**：将每个状态维度建模为输入条件化 Riccati 微分方程，离散时间流为 Möbius 变换，通过 $2 \times 2$ 矩阵乘法的结合律实现单次并行扫描，与 DEER/LrcSSM 的本质区别在于"构造可组合的动力学族"而非"对一般非线性做迭代线性化近似"。
2. **推导约束参数化方案**：通过对 Riccati 系数的特定参数化确保动力学存在有界不变区间 $[-B, B]$、区间内严格收缩、以及投影读取无极点，解决了通用 Riccati 方程可能有限时间发散的问题；与无约束 Riccati 的动力学本质的区别在于引入了显式的稳定性保证。
3. **建立与液体动力学的局部对应**：通过将 LRC 向量场在零状态处做二阶 Taylor 展开，证明 Riccati 动力学可在局部匹配 LRC 的动态特性，并用数值实验验证 RiccatiSSM 的精确 ZOH 积分误差比 LrcSSM 的显式 Euler 离散误差低两个数量级。
4. **系统实验验证效率与性能**：在 6 个 UEA 长序列分类数据集及回归/预测任务上验证，RiccatiSSM 与 LrcSSM 预测性能相当，但在匹配架构下将端到端运行时降低 22%–33%。

## 方法详解
- **输入条件化 Riccati 动力学**（逐维度对角结构）：
  $$\dot{x}_i = \varepsilon_i(u)\left[\alpha_i(u) + \beta_i(u)x_i + \gamma_i(u)x_i^2\right]$$
  其中 $\alpha, \beta, \gamma, \varepsilon$ 由输入 $u$ 的独立网络头生成；局部 Jacobian 为 $\varepsilon_i(\beta_i + 2\gamma_i x_i)$，显式依赖当前状态。
- **投影提升与精确离散化**：将标量 Riccati ODE 提升为二维线性系统 $\frac{d}{dt}(p,q)^\top = L_t(p,q)^\top$，其中 $L_t = \varepsilon_t\begin{pmatrix}\beta_t/2 & \alpha_t \\ -\gamma_t & -\beta_t/2\end{pmatrix}$ 为无迹矩阵；在分段常数输入假设下，精确零阶保持（ZOH）解为：
  $$M_t = \exp(\Delta t \cdot L_t) = \cosh(\omega_t \Delta t)I + \frac{\sinh(\omega_t \Delta t)}{\omega_t}L_t, \quad \omega_t = \sqrt{\varepsilon_t^2(\beta_t^2/4 - \alpha_t\gamma_t)}$$
  原始状态恢复为 $x_t = p_t/q_t$，对应 Möbius 更新 $x_t = (a_t x_{t-1} + b_t)/(c_t x_{t-1} + d_t)$。
- **单次并行扫描**：由于 Möbius 变换的复合等价于 $2 \times 2$ 矩阵乘法，整个序列的 prefix product $P_t = M_t \cdots M_1$ 可通过结合律单次并行前缀扫描计算，复杂度为 $\mathcal{O}(TD)$ 工作量与 $\mathcal{O}(\log T)$ 并行深度。
- **稳定参数化**（Appendix A.1）：将网络原始输出 $(\hat{\alpha}, \hat{\beta}, \hat{\gamma}, \hat{\varepsilon})$ 约束为：$\varepsilon = \sigma(\hat{\varepsilon})$，$s = s_{\min} + \mathrm{softplus}(\hat{\beta})$，$\alpha = (sB + |\gamma|B^2)\tanh(\hat{\alpha})$，$\beta = -(s + 2|\gamma|B)$，从而保证 $[−B,B]$ 为前向不变集且 Jacobian 在区间内满足 $\partial \dot{x}/\partial x \leq -\varepsilon s_{\min} < 0$。
- **前向流程三步**：① 并行生成所有时间步的系数矩阵 $M_t$；② 单次并行扫描计算 prefix products $\{P_t\}$；③ 投影读出 $x_t = P_t^{(11)}x_0 + P_t^{(12)})/(P_t^{(21)}x_0 + P_t^{(22)})$。

## 实验与结果
- **数据集与任务**：
  - 分类：6 个 UEA 时间序列数据集（Heartbeat 405、SCP1 896、SCP2 1152、Ethanol 1751、Motor 3000、EigenWorms 17984），覆盖序列长度 405–17984。
  - 回归：PPG-DaLiA 心率估计。
  - 预测：Weather 数据集，720 步历史预测 720 步未来。
- **主要结果（分类）**：
  - RiccatiSSM 在 Heartbeat ($72.7\%$ vs LrcSSM $73.0\%$)、SCP1 ($85.2\%$ vs $87.4\%$)、Ethanol ($36.9\%$ vs $36.6\%$)、Motor ($58.6\%$ vs $59.6\%$) 上与 LrcSSM 相当；在 Worms 上最优达 $90.6\%$（LrcSSM $83.9\%$）。
  - 最强结果：Motor Imagery $59.6\%$（与 LinOSS-IM $60.0\%$ 接近）。
- **回归（PPG-DaLiA）**：MSE $7.15 \pm 1.01 \times 10^{-2}$，优于 LrcSSM ($10.89 \times 10^{-2}$)，接近 LinOSS-IM ($6.40 \times 10^{-2}$)。
- **预测（Weather）**：MAE $0.5681$，优于 LrcSSM ($0.5888$) 和 S4 ($0.5783$)，接近 LinOSS-IMEX ($0.5081$)。
- **运行时间（Figure 2）**：匹配 6 层、hidden=64、state dim=64 架构下，RiccatiSSM 仅需约 1 次扫描，LrcSSM 需 ~3 次 quasi-DEER 迭代；端到端降低 **22%–33%**，最长序列 EigenWorms 上增益最大。
- **消融**：移除输入依赖速度因子 $\varepsilon(u)$ 使 MSE 上升 $1.3 \times 10^{-2}$；仅保留 $\alpha(u)$ 输入依赖时 MSE 升至 $8.71 \times 10^{-2}$；LRC-tied 参数化（Taylor 匹配约束）MSE 为 $8.47 \times 10^{-2}$，说明自由参数化更具灵活性。

## 相关工作脉络
1. **DEER / Quasi-DEER / ELK（Lim et al., 2024; Gonzalez et al., 2024）**：通过 Newton 迭代逐次线性化一般非线性递归来恢复并行性；RiccatiSSM 放弃"先任意后迭代逼近"路线，转而"先构造闭合族再单次精确求值"。
2. **LrcSSM（Farsang & Grosu, 2025）**：针对液体电阻/电容网络的对角 Jacobian 设计，但仍需 ~3 次 quasi-DEER 迭代；RiccatiSSM 与其在任务设置和实验协议上直接可比，但消除了迭代开销。
3. **线性 SSM（S4/S6/Mamba/LinOSS）**：仿射更新支持并行扫描但缺乏状态依赖动态；RiccatiSSM 填补了"非线性状态依赖"与"精确并行"之间的空白。
4. **Liquid Time-Constant Networks（LTC, Hasani et al., 2021）与 LRC**：引入状态依赖时间常数的生物启发电机；RiccatiSSM 保留该核心特性（Equation 9），但用 Riccati 二次项显式表达状态依赖，并实现了精确 ZOH 积分而非显式 Euler。
5. **Möbius Attention（Halacheva et al., 2024）与 Kalman Linear Attention（Shaj et al., 2026）**：前者将 Möbius 变换用于静态特征映射，后者用于线性高斯不确定度统计的递归；RiccatiSSM 首次将 Möbius 复合用于**状态本身的时序递推动力学**。

## 局限性与未来方向
- **动力学形式受限**：仅支持二阶多项式（Riccati）形式，系数 $\alpha, \beta, \gamma$ 只能依赖输入而不能依赖当前状态（否则破坏 composition 闭合性）。
- **稳定性约束引入归纳偏置**：有界不变区间和收缩性保证需要特定参数化，可能限制模型在极端动态场景下的表达能力。
- **Worms 数据集种子敏感性较高**：个体运行精度在 77.8%–91.7% 之间波动，说明在某些数据集上优化更敏感。
- **未来方向**（可合理推断）：探索其他在 composition 下闭合的非线性动态族（如有理函数更高阶形式）；将 projective lift 技术扩展到多维耦合状态；结合选择性机制（如 S6）实现通道级输入门控。

## 研究启发与可借鉴点
1. **"构造闭合族替代迭代逼近"的设计范式**：对于需要并行扫描的递归模型，优先考虑动力学本身是否在固定尺寸表示下 composition 闭合，可为其他非线性结构（如有理/分式动态）提供新思路。
2. **Projective lift + ZOH 精确积分技巧**：将标量非线性 ODE 提升为低维线性系统再取投影，既能获得精确离散化又保持并行性，可迁移至其他可升维的非线性动态设计。
3. **约束参数化保障稳定性的设计模式**：通过 softplus/tanh/sigmoid 复合将无约束网络输出映射到满足数学性质（有界、收缩、无极点）的参数空间，兼顾可学习性与理论保证。
4. **与已有液体模型的局部对应分析**：通过 Taylor 展开建立与新模型和现有模型的局部等价性，并用数值实验分解截断误差与离散化误差，为模型对比提供了严谨的分析框架。
5. **实验设计可借鉴**：使用与基线完全匹配的架构（层数、hidden dim、state dim）进行公平的时间对比，单独报告"最佳配置"与"匹配架构"两类结果，避免混用造成误导。

## 关键术语表
- **Riccati 微分方程**：形如 $\dot{x} = \alpha + \beta x + \gamma x^2$ 的标量非线性 ODE，其解可通过二维线性系统的投影表示。
- **Möbius 变换（分式线性变换）**：形如 $f(x)=(ax+b)/(cx+d)$ 的映射，可由 $2\times 2$ 矩阵表示，且复合等价于矩阵乘法。
- **并行前缀扫描（Parallel Scan）**：利用结合律将序列累积操作并行化的算法，工作量为 $\mathcal{O}(T)$、深度为 $\mathcal{O}(\log T)$。
- **零阶保持（Zero-Order Hold, ZOH）**：假设输入在每个时间步内保持恒定的离散化方法；此处用于对 Riccati 系统做精确矩阵指数积分。
- **投影读取（Projective Readout）**：从提升空间的齐次坐标 $(p_t, q_t)$ 通过除法 $x_t = p_t/q_t$ 恢复原始状态的操作。
- **LrcSSM**：基于液体电阻/电容网络的非线性 SSM，具有对角 Jacobian，但需 quasi-DEER 迭代实现并行评估。
- **DEER（Differentiable Equation Estimation Routine）**：将序列求值建模为不动点问题并用 Newton 迭代逐次线性化的并行化方法。
- **有界不变区间**：在约束参数化下，状态被限制在 $[-B,B]$ 内且向量场在边界处指向区间内部。

## 可复现要素
- **数据集**：UEA 时间序列分类数据集（Heartbeat、SCP1、SCP2、Ethanol、Motor、EigenWorms）、PPG-DaLiA、Weather；均公开可用。
- **代码/权重**：论文未提供公开代码链接，附录 C 说明代码基于 Rusch & Rus (2025) 和 Farsang & Grosu (2025) 的实现；完整超参数在 Table 7 中列出。
- **关键超参**：学习率 $\{10^{-5}, 10^{-4}, 10^{-3}\}$，hidden dim $\{16, 64, 128\}$，state-space dim $\{16, 64, 256\}$，层数 $\{2, 4, 6\}$；按 5 折 seed 网格/随机搜索选取。
- **硬件**：NVIDIA A40/A100 GPU。
- **复现难度**：中等——需实现 $2\times 2$ 矩阵并行扫描及 ZOH 精确积分；稳定参数化的具体公式见 Appendix A.1。
