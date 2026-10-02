---
title: "Safe-Greenhouse-Climate-Control-Using-Lagrangian-Constrained"
source: https://arxiv.org/pdf/2609.34966v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:12:08"
field: "农业智能控制与安全强化学习"
keywords: ["Greenhouse Climate Control", "Safe Reinforcement Learning", "Constrained MDP", "Kolmogorov-Arnold Networks", "PPO", "RCPO", "Smart Farming"]
innovations: ["将温室气候控制建模为CMDP并通过RCPO自适应拉格朗日对偶更新显式约束长期累计违规", "用KAN替代MLP作为策略与价值网络增强强非线性温室动态的表征能力", "正弦周期时间编码显式嵌入观测以捕捉温室日周期动态"]
benchmarks: ["Winter Lettuce Greenhouse Model", "Cumulative Economic Profit", "Cumulative Climate Violation Cost"]
---

# 论文速读：Safe-Greenhouse-Climate-Control-Using-Lagrangian-Constrained

## 一句话总结
本文针对温室气候控制中经济收益与作物安全约束之间的权衡难题，将问题建模为约束马尔可夫决策过程（CMDP），提出基于拉格朗日自适应惩罚的RCPO-PPO安全强化学习框架，并结合Kolmogorov–Arnold Networks（KAN）与正弦周期时间编码增强策略表达能力，在冬季生菜温室仿真中实现了累计气候违规降低18.65%且经济利润提升2.91%的效果。

## 研究问题与动机
- 现有基于强化学习的温室控制器依赖固定奖励惩罚项来间接约束气候变量偏离，但启发式惩罚权重的选择高度依赖人工调参，难以显式控制长期累计违规量。
- 惩罚权重设置不当会导致两类极端：过重惩罚使策略过度保守、牺牲产量；过轻惩罚无法抑制持续性气候偏离，进而抑制光合作用并诱发作物病害。
- 温室微气候与作物生长之间存在强非线性与时变耦合特性，传统MLP策略网络在表达此类复杂动态时存在表征效率瓶颈。
- 日周期性的环境波动（光照、温湿度昼夜变化）是温室控制的重要先验信息，但未在观测中充分建模。

## 核心贡献（创新点）
1. **温室专用CMDP建模**：将经济回报与累计气候风险约束显式解耦，允许种植者直接指定长期违规预算阈值。
2. **RCPO自适应拉格朗日约束机制**：通过在线对偶更新替代人工调参的固定惩罚系数，使累计违规稳定收敛至预设安全阈值。
3. **KAN替代MLP作为策略与价值近似器**：利用Kolmogorov–Arnold表示定理沿网络边学习一元样条函数，增强对温室强非线性动态的拟合能力。
4. **正弦周期时间特征嵌入观测**：将日内时间编码为sin/cos对偶特征，显式捕捉温室环境的日周期动态。
5. **系统化消融验证**：分别验证了RCPO约束机制、KAN网络结构、时间特征编码三者的独立贡献与协同效应。

## 方法详解
- **CMDP形式化**：定义$\mathcal{M}_c = \langle S, A, P, r, \{c_i\}, \gamma, \{d_i\} \rangle$，目标为$\max_\pi J_r(\pi)$，约束为$J_{c_i}(\pi) = \mathbb{E}[\sum_t \gamma^t c_i(s_t, a_t)] \leq d_i$，其中$d$为累计违规预算。
- **拉格朗日函数**：$\mathcal{L}(\pi, \lambda) = J_r(\pi) - \lambda(J_c(\pi) - d)$，将对偶变量$\lambda$作为自适应惩罚强度。
- **对偶更新规则**：每$K=4$个训练回合更新一次$\lambda \gets [\lambda + \alpha_\lambda(J_c(\pi) - d)]_+$，$\alpha_\lambda=0.01$，$\lambda \in [0.05, 50]$。
- **等效修正奖励**：$\tilde{r}_t = r_t - \lambda c_t$，用于PPO的actor-critic训练，而$\lambda$更新使用原始约束成本评估。
- **瞬时违规代价**：$c_t = w_{CO_2}v_{CO_2}(t) + w_T^{low}v_T^{low}(t) + w_T^{high}v_T^{high}(t) + w_{RH}v_{RH}(t)$，其中$v_{(\cdot)}(t) = \max(0, y_{(\cdot)} - y_{(\cdot)}^{max}) + \max(0, y_{(\cdot)}^{min} - y_{(\cdot)})$。
- **观测向量**：$s_t = [y(t), u(t-1), w(t:t+N_p), \tau(t), \sin(\phi_t), \cos(\phi_t)]$，维度62，含作物干重、$CO_2$浓度、温度、湿度、前一动作、6小时天气前瞻及周期时间特征。
- **KAN网络结构**：Actor/Critic均为3层[KAN]，隐层宽度128，样条阶数$k=3$，网格大小3，范围$[-1,1]$，基础激活函数SiLU。
- **经济奖励**：$r_t = c_{DW}(x_{DW}(t+1)-x_{DW}(t)) - (10^{-6}c_{CO_2}u_{CO_2} + \frac{c_{heat}u_{heat}}{3.6\times10^6})\Delta t$，单位€/m²。
- **实现框架**：基于Stable-Baselines3的PPO，$\gamma=0.98$，总训练步数$4\times10^6$，并行环境数8，裁剪范围0.1。

## 实验与结果
- **仿真环境**：荷兰Bleiswijk冬季生菜温室动态模型（van Henten, 1994），40天真实天气扰动输入，每30分钟采样一次，参数扰动±2.5%均匀分布白噪声（std=0.3）。
- **基线方法**：固定惩罚系数的vanilla PPO。
- **主要结果**（Table 7）：
  - RCPO-PPO+KAN+Time：**累计利润3.892 €/m²**，较基线PPO（3.782）提升**+2.91%**；累计违规**0.698**，较基线（0.858）降低**-18.65%**。
  - 变量级违规减少：相对湿度下降20.3%，CO₂下降15.6%，温度下降9.2%。
  - 能耗降低：CO₂注入成本降低22.5%，加热成本降低5.4%，总运营成本降低8.8%。
- **消融结论**：RCPO是约束满足的主要贡献者；KAN与时间特征主要提升经济性能；三者结合取得最优综合效果。

## 相关工作脉络
1. **Morcego et al. (2023)**：比较RL与MPC在温室控制中的表现，本文在其基础上引入显式安全约束机制解决RL策略的长期违规累积问题。
2. **Van Laatum et al. (2026)**：结合随机MPC与RL处理参数不确定性，本文聚焦于无模型拉格朗日约束优化，避免了对精确动力学模型的依赖。
3. **Wachi & Sui (2020)**：提出CMDP下的Safe RL理论框架，本文将其适配到温室场景并引入KAN增强表征。
4. **Tessler et al. (2018) RCPO**：原始算法针对离散/连续通用环境，本文将其与PPO、KAN及周期特征深度集成于温室控制。
5. **Liu et al. (2024) KAN**：提出Kolmogorov–Arnold Networks架构，本文首次将其引入在线强化学习温室控制任务。

## 局限性与未来方向
- **天气预测假设理想化**：采用"完美预测"（perfect forecast）即直接使用历史实测序列作为前瞻输入，未建模预报误差对策略鲁棒性的影响。
- **仅验证单一作物与季节**：模型基于冬季生菜，尚未推广至其他作物类型或不同季节条件。
- **病害风险未直接建模**：虽指出高湿降低可降低病害风险，但疾病发生率未纳入约束或奖励。
- **仿真验证而非实地部署**：结论基于数值仿真，真实温室环境的传感器噪声、执行器延迟等未考虑。
- 未来方向包括扩展至多作物/多季节、集成不确定性感知决策、在真实温室环境中验证。

## 研究启发与可借鉴点
1. **拉格朗日对偶替代手工调参惩罚权重**：在任意需平衡收益与安全约束的控制系统中，RCPO风格的对偶更新可消除繁琐的超参搜索。
2. **KAN替代MLP提升非线性表征**：对于具有强非线性、多变量耦合的时序控制任务，KAN可作为MLP的可选升级方案，尤其在低数据效率场景下可能更快收敛。
3. **周期特征编码（sin/cos）融入观测**：对具有昼夜/季节性周期的动态系统，直接嵌入循环时间特征比one-hot或线性编码更具表达力，可迁移至能源管理、交通调度等领域。
4. **分离"惩罚强度学习"与"策略性能优化"**：将约束违反的度量与策略优化的目标解耦（即用原始$c_t$评估约束、用$\tilde{r}_t$训练策略），是Safe RL工程落地的实用技巧。

## 关键术语表
**Constrained Markov Decision Process (CMDP)**：在标准MDP基础上引入累计代价约束的决策框架，目标是在满足约束前提下最大化累计奖励。
**Reward Constrained Policy Optimization (RCPO)**：基于拉格朗日对偶的在线原-对偶优化算法，自适应调整惩罚强度以保障长期约束满足。
**Kolmogorov–Arnold Network (KAN)**：基于Kolmogorov–Arnold表示定理的神经网络架构，将非线性变换参数化在边上（可学习样条函数）而非节点（固定激活函数）。
**Lagrangian Multiplier (拉格朗日乘子)**：在约束优化中引入的对偶变量，用于衡量约束违反的严重程度并自适应调节惩罚强度。
**Diurnal Periodic Encoding (日周期编码)**：将一天内的时间点通过sin/cos函数映射为二维循环特征，以显式表达环境的昼夜周期性变化。
**Cumulative Violation Budget (累计违规预算)**：由用户指定的长期约束违反上界$d$，代表可接受的风险总量阈值。

## 可复现要素
- **数据集**：荷兰Bleiswijk地区2014年2月9日起40天实测天气序列（辐射、CO₂、温湿度），未公开代码/权重。
- **仿真模型**：van Henten (1994) 冬季生菜温室动态模型，离散化步长$\Delta t = 30$ min，RK4积分。
- **关键超参**：$\gamma=0.98$，总训练步数$4\times10^6$，并行环境8，裁剪0.1，$\lambda_0=1.0$，$\alpha_\lambda=0.01$，$d=0.7$，$K=4$。
- **代码状态**：论文未提供开源代码与权重链接。
