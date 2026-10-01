---
title: "RICCATI-STATE-SPACE-MODELS-NON-ITERATIVE-PARALLELIZATION-FOR"
source: https://arxiv.org/pdf/2609.35441v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:08:47"
field: "高效序列建模"
keywords: ["State Space Models", "Riccati Dynamics", "Parallel Scan", "Nonlinear Sequence Modeling", "Mobius Transform", "Liquid Dynamics", "Long-sequence"]
innovations: ["通过 Riccati 动态的 Mobius 流映射实现单次并行扫描的非迭代非线性 SSM", "约束参数化保证有界收缩动态且避免极点", "相比迭代 LrcSSM 减少 22-33% 运行时间并保持竞争性性能"]
benchmarks: ["UEA Time Series Classification", "PPG-DaLiA Regression", "Weather Forecasting"]
---

# 论文速读：RICCATI-STATE-SPACE-MODELS-NON-ITERATIVE-PARALLELIZATION-FOR

## 一句话总结
论文提出 RiccatiSSM，一种输入条件化的非线性状态空间模型，其每个状态维遵循 Riccati 微分方程；通过将离散状态更新表示为 Mobius 变换（分式线性映射），利用其组合封闭性（对应 $2 \times 2$ 矩阵乘法）实现单次关联并行扫描的精确评估，无需迭代线性化。在长序列分类、回归和预测任务上，RiccatiSSM 取得与现有基线竞争性的预测性能，同时相比迭代非线性 LrcSSM 在匹配架构下减少 22–33% 的运行时间。

## 研究问题与动机
- **核心问题**：现有状态空间模型（SSMs）因状态更新为仿射形式而支持单次并行扫描，但缺乏状态依赖的非线性动态；一般非线性递归模型虽能提供状态依赖的动态（如局部收缩率变化），但失去组合封闭性，需通过迭代线性化（如 DEER、LrcSSM 的 quasi-DEER 方法）才能并行评估，导致多次扫描和额外计算开销。
- **现有方法不足**：
  1. 线性/仿射 SSMs（如 S5、Mamba、LinOSS）可精确并行扫描，但局部 Jacobian 不依赖于当前状态，动态表达能力受限。
  2. 非线性递归模型（如 LTC、LrcSSM）通过状态依赖的动态提升表达力，但评估需 $K$ 次迭代扫描（$K$ 数据依赖且可能停滞），深度为 $O(K \log T)$，计算成本高。
  3. 现有并行化非线性递归的方法（如 DEER、ELK）通过 Newton 迭代近似，非精确组合，且迭代次数不固定。
- **研究动机**：探索是否存在一类非线性动态，其时间步映射恰好保持组合封闭性，从而允许通过单次关联并行扫描精确评估完整非线性状态轨迹，兼顾状态依赖动态与高效并行计算。

## 核心贡献（创新点）
1. **引入 RiccatiSSM**：提出一种非线性状态依赖序列模型，其离散时间流映射为 Mobius 变换（分式线性映射），通过 $2 \times 2$ 矩阵乘法的组合封闭性实现单次关联并行扫描的精确评估，无需 Newton 或不动点迭代。
   - **与已有工作的本质区别**：不同于 DEER/LrcSSM 通过迭代线性化近似非线性递归，RiccatiSSM 从动态本身设计出发，使非线性流映射恰好属于组合封闭族。
2. **推导约束参数化方案**：提出约束 Riccati 系数参数化，确保动态有界、收缩，并避免分数线性状态更新中的极点，使 Riccati 动态适合序列建模。
   - **与已有工作的本质区别**：一般 Riccati 方程可能在有限时间内发散（分母为零），本工作通过构造不变区间 $[-B, B]$ 和严格收缩条件（Jacobian $\leq -\varepsilon s_{\min} < 0$）保证稳定性，而 LrcSSM 依赖饱和非线性实现有界性。
3. **实证验证效率与性能**：在长序列分类、回归和预测任务上评估，RiccatiSSM 达到竞争性预测性能，同时在匹配架构下相比迭代非线性 LrcSSM 减少 22–33% 端到端运行时间。
   - **与已有工作的本质区别**：首次展示通过动态设计实现非迭代并行化可带来实际计算收益，而不仅限于理论可行性。

## 方法详解
- **输入条件化 Riccati 动态**：对每个状态维度 $i$，连续时间动态为 $\dot{x}_i = \varepsilon_i(u)[\alpha_i(u) + \beta_i(u) x_i + \gamma_i(u) x_i^2]$，其中系数 $\alpha_i, \beta_i, \gamma_i, \varepsilon_i$ 由输入 $u$ 依赖的投影头生成；二次项使局部 Jacobian $\partial \dot{x}_i/\partial x_i = \varepsilon_i(\beta_i + 2\gamma_i x_i)$ 显式依赖于当前状态，实现状态依赖的收缩率。
- **射影提升与精确离散化**：将标量 Riccati 方程视为二维线性系统 $\frac{d}{dt}\begin{pmatrix} p \\ q \end{pmatrix} = L \begin{pmatrix} p \\ q \end{pmatrix}$ 的射影表示（$x = p/q$），其中 $L = \varepsilon \begin{pmatrix} \beta/2 & \alpha \\ -\gamma & -\beta/2 \end{pmatrix}$ 为无迹矩阵；采用零阶保持（ZOH）离散化，每步精确解为 $M_t = \exp(\Delta t L_t)$，其闭式解为 $M_t = \cosh(\omega_t \Delta t) I + \frac{\sinh(\omega_t \Delta t)}{\omega_t} L_t$，$\omega_t = \sqrt{\varepsilon_t^2(\beta_t^2/4 - \alpha_t \gamma_t)}$；由此诱导的状态更新 $x_t = (a_t x_{t-1} + b_t)/(c_t x_{t-1} + d_t)$ 为 Mobius 变换。
- **单次关联并行扫描**：由于 Mobius 变换的组合对应 $2 \times 2$ 矩阵乘法（$M_{f_2 \circ f_1} = M_{f_2} M_{f_1}$），定义前缀积 $P_t = M_t \cdots M_1$，利用矩阵乘法的结合性，通过单次关联并行扫描计算所有 $P_t$（时间复杂度 $O(T)$ 工作、$O(\log T)$ 并行深度）；最终状态通过射影读出 $x_t = (P_t^{(11)} x_0 + P_t^{(12)}) / (P_t^{(21)} x_0 + P_t^{(22)})$ 恢复。
- **稳定参数化**：约束系数确保存在不变区间 $[-B, B]$ 且动态在其内严格收缩：参数化网络输出 $(\hat{\alpha}, \hat{\beta}, \hat{\gamma}, \hat{\varepsilon})$，经变换 $\varepsilon = \sigma(\hat{\varepsilon})$，$s = s_{\min} + \text{softplus}(\hat{\beta})$，$\alpha = (sB + |\gamma|B^2)\tanh(\hat{\alpha})$，$\beta = -(s + 2|\gamma|B)$；该设计保证 $\beta^2/4 - \alpha\gamma > 0$（实双曲分支）、$|\alpha| \leq sB + |\gamma|B^2$、$\beta \leq -s \leq -s_{\min}$，从而 Jacobian $\leq -\varepsilon s_{\min} < 0$，轨迹有界且无极点。

## 实验与结果
- **数据集与基准**：6 个 UEA 时间序列分类数据集（Heartbeat 序列长 405、SCP1 896、SCP2 1152、Ethanol 1751、Motor 3000、Worms 17984）；回归数据集 PPG-DaLiA；预测数据集 Weather（720 步历史预测 720 步未来）。基线包括 Transformer 变体（RFormer）、神经 ODE/控制微分方程（NRDE、NCDE、Log-NCDE）、线性 SSMs（LRU、S5、Mamba、S6、LinOSS-IMEX/IM）及非线性 LrcSSM。
- **主要结果**：
  - **分类**：RiccatiSSM 在 SCP1（$87.4\%$ vs LrcSSM $85.2\%$）和 Motor（$59.6\%$ vs LrcSSM $58.6\%$）上优于 LrcSSM，在 Heart、SCP2、Ethanol 上性能相当；在 Worms 上变异性较大（种子间 77.8%–91.7%）。
  - **回归**：PPG-DaLiA 上 MSE $\times 10^{-2}$ 为 $7.15 \pm 1.13$，优于 LrcSSM（$10.89 \pm 0.96$），接近最强线性基线 LinOSS-IM（$6.40 \pm 0.23$）。
  - **预测**：Weather 上 MAE 为 $0.5681$，优于 LrcSSM（$0.5888$）及多数基线，接近 LinOSS-IMEX（$0.5081$）。
- **效率提升**：匹配 6 层架构、状态维度 64 下，RiccatiSSM 单步评估时间为 LrcSSM 的 67–78%（图 2），端到端运行时间减少 22–33%；最长序列 EigenWorms 上提速最显著。

## 相关工作脉络
1. **并行 SSMs（S5、Mamba、LinOSS）**：仿射状态更新支持单次并行扫描，但动态无状态依赖性；RiccatiSSM 扩展至非线性状态依赖动态，仍保持单次扫描。
2. **非线性递归并行化（DEER、ELK、LrcSSM）**：通过 Newton/准 Newton 迭代线性化递归，需 $K$ 次扫描；RiccatiSSM 通过动态设计避免迭代，实现精确组合。
3. **液体动力学模型（LTC、LRC、LrcSSM）**：状态依赖的时间常数通过神经元内部非线性实现；RiccatiSSM 以二次项显式建模状态依赖，并通过射影提升获得精确离散化。
4. **Mobius 变换在 ML 中的应用（MobiusAttention、Kalman Linear Attention）**：前者用于静态特征映射，后者用于不确定性统计的递归更新；RiccatiSSM 将 Mobius 变换直接作用于状态本身作为时间动态。
5. **精确离散化 ODE 层（CfC、LinOSS-IM）**：通过闭式解避免数值积分误差；RiccatiSSM 对 Riccati 方程采用 ZOH 精确积分，结合并行扫描实现整体无近似误差。

## 局限性与未来方向
- **动态形式限制**：RiccatiSSM 仅限于二次多项式形式，系数只能依赖输入而非状态；更一般的非线性动态（如高次多项式、超越函数）无法保持组合封闭性。
- **优化敏感性**：在 Worms 数据集上表现变异性大（种子间精度波动显著），可能对超参数或初始化更敏感。
- **固定区间约束**：稳定参数化强制状态局限于 $[-B, B]$，可能限制对超出该范围动态任务的表达力。
- **未来方向**：探索其他组合封闭的非线性动态族（如有理函数更高阶形式）；设计自适应区间扩展机制；将 RiccatiSSM 集成至更深层架构或长上下文语言模型中验证泛化性。

## 研究启发与可借鉴点
1. **动态设计驱动并行化**：通过选择组合封闭的动力学族（如 Mobius 变换），可将非线性序列评估转化为单次关联扫描；这一思路可扩展至其他封闭映射族（如线性分式、正交变换）以平衡表达力与效率。
2. **射影提升技巧**：将非线性标量 ODE 嵌入二维线性系统（$x=p/q$）实现精确离散化，可借鉴于其他非线性动态的并行化设计（如广义 Riccati 型方程）。
3. **约束参数化保障稳定性**：通过参数变换（softplus、tanh、sigmoid）将无约束网络输出映射至满足稳定条件的系数空间，避免训练发散；该方法可推广至其他需保证有界性的动态模型。
4. **实验对比严谨性**：在匹配架构（层数、状态维度）下对比运行时性能，分离了动态设计带来的计算收益；这一对照实验设计值得在效率研究中复用。
5. **与液体动力学的局部对应**：通过二阶泰勒展开建立 Riccati 动态与 LRC 向量场的局部等价性，并提供积分误差隔离分析；该分析方法可用于新动态族与现有模型的对比研究。

## 关键术语表
- **State Space Models (SSMs)**：通过仿射状态更新处理序列的模型，其组合封闭性支持高效并行扫描。
- **Mobius 变换（分式线性映射）**：形如 $f(x)=(ax+b)/(cx+d)$ 的映射，对应 $2\times2$ 矩阵表示，组合封闭且可通过矩阵乘法实现。
- **Parallel Prefix Scan（关联扫描）**：利用结合律并行计算序列前缀积的算法，工作复杂度 $O(T)$，深度 $O(\log T)$。
- **Riccati 微分方程**：一阶非线性 ODE，形式为 $\dot{x}=\alpha+\beta x+\gamma x^2$，其解可由二维线性系统射影表示。
- **Zero-Order Hold (ZOH) 离散化**：假设输入在时间步内恒定，通过矩阵指数精确积分连续动态。
- **Liquid Time-Constant (LTC) 网络**：神经元时间常数由当前状态调节的连续时间递归模型，具备状态依赖动态。
- **Contraction Dynamics（收缩动态）**：局部 Jacobian 负定确保状态轨迹收敛，保障模型稳定性。
- **Projective Readout（射影读出）**：从 lifted 坐标 $(p,q)$ 恢复状态 $x=p/q$ 的操作。

## 可复现要素
- **数据集**：UEA 时间序列分类数据集（Heartbeat、SCP1/2、Ethanol、Motor、Worms）、PPG-DaLiA 回归数据集、Weather 预测数据集；论文未明确声明公开链接，但 UEA 和 PPG-DaLiA 为公共数据集。
- **代码/权重**：论文未提供开源代码或预训练权重；实验基于 Rusch & Rus (2025) 和 Farsang & Grosu (2025) 的代码库扩展。
- **关键超参数**：学习率 $\{10^{-5},10^{-4},10^{-3}\}$、隐藏维度 $\{16,64,128\}$、状态空间维度 $\{16,64,256\}$、层数 $\{2,4,6\}$；最优配置见论文 Table 7；稳定参数化中 $s_{\min}>0$（具体值未提及）、区间边界 $B$（未指定）。
- **硬件**：NVIDIA A40 和 A100 GPU。
