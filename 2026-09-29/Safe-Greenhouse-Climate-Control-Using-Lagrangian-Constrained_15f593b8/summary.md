---
title: "Safe-Greenhouse-Climate-Control-Using-Lagrangian-Constrained"
source: https://arxiv.org/pdf/2609.34966v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:10:33"
field: "农业智能控制与强化学习"
keywords: ["Greenhouse Climate Control", "Safe Reinforcement Learning", "Constrained Markov Decision Process", "Kolmogorov-Arnold Networks", "RCPO", "Proximal Policy Optimization"]
innovations: ["将温室气候控制建模为CMDP并以拉格朗日乘子自适应调节累积违反约束", "用KAN替代PPO中MLP以提升强非线性温室动态的表示能力", "结合正弦周期时间编码捕获昼夜气候节律"]
benchmarks: ["Winter lettuce greenhouse dynamic model", "40-day real-weather disturbance from Bleiswijk, Netherlands"]
---

# 论文速读：Safe-Greenhouse-Climate-Control-Using-Lagrangian-Constrained

## 一句话总结
论文将温室气候控制建模为约束马尔可夫决策过程（CMDP），提出基于拉格朗日约束的 RCPO-PPO 安全强化学习框架，并结合 Kolmogorov–Arnold Networks（KAN）与正弦周期时间编码，实现经济收益最大化与长期气候约束违反的显式平衡。

## 研究问题与动机
- 现有 RL 温室控制器依赖固定奖励惩罚间接约束气候偏差，但启发式标量惩罚无法显式控制长期累积违反量，权重设置不当会导致策略过度保守（降低产量）或无法抑制持续气候偏差（抑制光合作用、引发病害）。
- 传统 MPC 等方法依赖准确的温室-作物动力学模型，难以捕捉作物生理、室内微气候与外部气象扰动间的强非线性与时变耦合。
- 安全 RL（如 CBF、Safe Shielding）多依赖精确解析动力学模型，计算开销大，与农业环境中常见的长时域累积约束不匹配。
- 温室气候具有强昼夜周期特性，标准 MLP 基 RL 控制器难以有效捕获此类周期性动态。

## 核心贡献（创新点）
1. **提出面向温室气候控制的 CMDP 建模**：显式分离经济效益优化目标与累积气候风险约束，使种植者可直接指定长期违反预算。
2. **引入 RCPO 拉格朗日自适应约束机制**：通过在线 primal–dual 更新自动调节惩罚强度，避免手动调参，使累积违反量稳定贴近预设阈值。
3. **用 KAN 替代 PPO 中的 MLP 策略/价值网络**：利用 Kolmogorov–Arnold 表示定理，将非线性变换分布于边而非节点，提升对温室强非线性动态的拟合能力。
4. **设计正弦/余弦周期时间编码**：显式嵌入昼夜节律特征，帮助控制器捕捉环境变化的周期性模式。
5. **系统性消融实验验证各组件贡献**：明确区分 RCPO（约束满足）与 KAN/时间编码（经济性能提升）的独立作用。

## 方法详解
- **CMDP 建模**：状态 $s_t$ 包含可测温室输出（干重、CO₂、温度、湿度）、上一时刻控制量、6 小时天气前瞻窗口（$N_p = 13$，Δt=30 min）以及昼夜周期编码；动作 $a_t$ 为 CO₂ 注入速率、通风率、加热功率。
- **经济奖励**：瞬时奖励为作物干重增量收益减去 CO₂ 与加热成本（通风成本忽略），单位归一化为 €/m²。
- **气候约束代价**：定义 $v_{(\cdot)}(t)$ 为各变量偏离允许范围的正向距离，累积代价 $J_c(\pi) = \mathbb{E}[\sum_t \gamma^t c_t]$，其中 $c_t$ 为加权 violation 之和。
- **RCPO 拉格朗日优化**：构造 $\mathcal{L}(\pi, \lambda) = J_r(\pi) - \lambda(J_c(\pi) - d)$，策略通过 PPO 基于修正奖励 $\tilde{r}_t = r_t - \lambda c_t$ 更新；拉格朗日乘子每 $K=4$ 回合更新：$\lambda \leftarrow [\lambda + \alpha_\lambda(J_c(\pi) - d)]_+$。
- **KAN 网络架构**：Actor/Critic 均为 3 层 KAN，维度 [62, 128, 128, 128]，spline order k=3，grid size=3，grid range=[-1,1]，base function=SiLU，Adam 优化器，学习率 $1\times10^{-4}$。
- **周期时间编码**：$t_{sin} = \sin(2\pi t / T_{day})$，$t_{cos} = \cos(2\pi t / T_{day})$，拼接至观测向量以捕获昼夜动态。

## 实验与结果
- **仿真环境**：基于 van Henten 冬季生菜温室动态模型（28 参数非线性 ODE），RK4 离散化，Δt=30 min；参数不确定性 ±2.5%；天气数据取自荷兰 Bleiswijk 2014 年 2 月 9 日起连续 40 天实测序列，训练/评估加入 std=0.3 白噪声扰动。
- **评估基线**：Penalty-based vanilla PPO（固定惩罚权重）。
- **主要结果（Table 7）**：
  - RCPO-PPO（完整方法）：累积利润 3.892 €/m²（↑2.91% vs 基线 3.782），累积违反 0.698（↓18.65% vs 基线 0.858）。
  - 消融显示：RCPO 独立贡献违反降低约 17–19%；KAN 与时间编码主要提升经济性能。
- **变量级分析（Fig. 4）**：湿度违反降低 20.3%，CO₂ 违反降低 15.6%，温度违反降低 9.2%。
- **资源消耗（Fig. 5）**：CO₂ 投入成本降低 22.5%，加热成本降低 5.4%，总运行成本降低 8.8%。
- **训练设置**：总步数 $4\times10^6$，8 并行环境，rollout=1920，γ=0.98，Clipping=0.1，$\alpha_\lambda=0.01$，$\lambda_0=1.0$，约束阈值 $d=0.7$。

## 相关工作脉络
- **传统控制方法**：规则控制、PID、MPC（[8][9][10]）——依赖精确模型，难以处理强非线性与不确定性。
- **RL 温室控制基线**：Penalty-based PPO（[16][17][21][22]）——通过固定惩罚隐式约束，缺乏长期违反显式控制能力。
- **安全 RL 范式**：Control Barrier Functions（CBF）与 Safe Shielding（[26][27][39]）——步级安全过滤，需精确动力学模型；本文采用模型-free 拉格朗日方法，更适合农业长时域累积约束。
- **CMDP 与 RCPO**：Wachi & Sui（[29]）、Tessler et al.（[40]）——拉格朗日 primal–dual 框架，本文将其适配至温室气候控制并验证。
- **KAN 在 RL 中的应用**：Kich et al.（[34]）首次探索 KAN 用于在线 RL；本文进一步验证其在温室强非线性控制中的优势。
- **周期特征编码**：时间正弦/余弦编码在机器人控制中已有应用，本文将其引入温室场景以捕获昼夜节律。

## 局限性与未来方向
- 仅在单一生菜温室模型上验证，未推广至其他作物类型或季节条件。
- 使用"完美天气预报"（即训练与评估使用同一实测序列），未建模预报误差对控制性能的影响。
- 尚未在真实温室环境中部署验证，仿真-现实差距未评估。
- 未考虑作物病害等生物响应指标，仅以气候偏差作为安全代理。
- 未来工作：扩展至多样作物/季节、真实环境验证、集成先进安全 RL 算法、引入不确定性感知决策。

## 研究启发与可借鉴点
- **CMDP 建模范式**可直接迁移至其他农业/能源系统控制场景（如禽舍环境调控、建筑节能），将多目标权衡转化为显式约束优化。
- **KAN + 周期编码组合**对强非线性、强周期驱动的时变系统具有泛化价值，可在其他过程控制任务中复用。
- **RCPO 自适应拉格朗日机制**避免了固定惩罚权重的繁琐调参，适合对安全边界有明确要求的工业场景。
- **消融设计方法**：将安全约束机制（RCPO）与表征增强（KAN/时间编码）解耦评估，清晰归因各组件贡献，值得借鉴。
- **资源成本联合优化**：在奖励函数中同时建模作物收益与 CO₂/加热成本，实现经济-安全-资源三重平衡。

## 关键术语表
- **CMDP（Constrained Markov Decision Process）**：在标准 MDP 基础上引入长期累积代价约束的决策框架，用于显式刻画安全约束。
- **RCPO（Reward Constrained Policy Optimization）**：基于拉格朗日对偶的在线 primal–dual 安全 RL 算法，自适应调节约束惩罚强度。
- **KAN（Kolmogorov–Arnold Networks）**：受 Kolmogorov–Arnold 表示定理启发的神经网络架构，将非线性变换参数化在网络边上（可学习样条函数），而非标准 MLP 的节点激活函数。
- **拉格朗日乘子（Lagrange Multiplier）**：在约束优化中用于衡量约束紧度的标量参数，此处自适应更新以调节气候违反惩罚强度。
- **周期时间编码（Cyclic Time Encoding）**：使用 sin/cos 函数将时间映射到单位圆上，避免 23:59→00:00 的离散跳变，有效捕获昼夜周期模式。
- **累积气候违反（Cumulative Climate Violation）**：整个生长周期内气候变量超出允许范围的加权积分代价，作为 CMDP 中的约束目标。
- **Primal–Dual 优化**：同时更新策略参数（primal）和拉格朗日乘子（dual）的优化方法，用于求解带约束的 RL 问题。
- **GAE（Generalized Advantage Estimation）**：PPO 中用于降低优势函数估计方差的技术，本文使用 λ_GAE=0.95。

## 可复现要素
- **数据集**：荷兰 Bleiswijk 2014 年 2 月 9 日起 40 天实测天气序列（辐射、CO₂、温度、湿度），论文未提供公开链接。
- **代码开源**：论文未明确声明代码是否开源；仿真基于 Stable-Baselines3 框架。
- **关键超参**：总训练步数 $4\times10^6$，并行环境数 8，rollout 长度 1920，批大小 1920，优化 epoch 10，γ=0.98，GAE λ=0.95，Clipping=0.1，熵系数 0.01，价值函数系数 0.5，最大梯度范数 0.5；RCPO 初始乘子 1.0，min=0.05，max=50.0，阈值 d=0.7，更新间隔 K=4，学习率 0.01；KAN 层数 [62,128,128,128]，spline order=3，grid size=3，grid range=[-1,1]，base function=SiLU，Adam 学习率 $1\times10^{-4}$。
- **模型参数**：28 参数温室-作物模型（见 Table 8），含 ±2.5% 参数不确定性。
