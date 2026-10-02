---
title: "SCA-Spatial-Credit-Assignment-for-Reinforcement-Learning-of"
source: https://arxiv.org/pdf/2609.36939v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:56:41"
field: "GUI 智能体强化学习"
keywords: ["GUI Agents", "Reinforcement Learning", "Spatial Credit Assignment", "Group-Relative RL", "Visual Grounding", "Policy Gradient"]
innovations: ["提出 SCA 框架利用屏幕坐标改进 group-relative 信用分配", "设计 Residual/Prox 双分支处理混合命中与全失败组", "提供精确梯度对比验证信用估计器的方向对齐性"]
benchmarks: ["ScreenSpot-Pro", "GUI-Act-Web", "OmniAct-Web", "OmniAct-Desktop", "GUI-Odyssey"]
---

# 论文速读：SCA-Spatial-Credit-Assignment-for-Reinforcement-Learning-of-GUI-Agents

## 一句话总结
本文提出空间信用分配（SCA）方法，利用样本点击的屏幕坐标信息改进 GUI 智能体的群体相对强化学习信用分配，解决二元奖励评估中空间不同失败点击被等同对待、全失败组缺乏相对信号的问题。

## 研究问题与动机
- **现有 Group-Relative RL 忽略空间结构**：GRPO 等方法仅基于采样动作的二元奖励比较分配信用，未利用 GUI 动作的坐标空间信息。
- **混合命中组的空间信用损失**：成功与失败点击组成的组中，所有失败点击获得相同组内优势值，忽略了它们距目标位置的远近差异所蕴含的局部坐标-奖励关系。
- **全失败组的零优势困境**：当所有采样点击均失败时，二元奖励方差为零导致组优势为零，无法从空间距离信息中提取有用梯度信号。
- **推理部署无需额外预测器**：空间参考仅用于训练更新构造，部署策略保持不变，无推理开销。

## 核心贡献（创新点）
- **提出 SCA 框架解决二元信用的空间盲区**：与 GRPO/RLOO 等纯奖励比较方法本质不同，首次将屏幕坐标作为信用分配的空间参考引入 GUI 强化学习。
- **设计 SCA-Residual 处理混合命中组**：通过留一法拟合坐标-奖励趋势并计算残差信用，区别于简单距离奖励塑形或固定高斯奖励建模。
- **设计 SCA-Prox 处理全失败组**：使用目标近邻指数排序分配信用，与 SE-GUI 等自演化方法不同，仅在有坐标标注时提供方向信号。
- **提供精确梯度对比验证**：通过合成实验验证空间更新与精确回报梯度的方向对齐性，区别于仅报告基准指标的工作。
- **在多个 GUI 基准上实现最强 RFT 结果**：在 ScreenSpot-Pro 和多个低级别动作预测指标上超越 GUI-R1/UI-R1 等基线。

## 方法详解
- **分组采样与路由机制**：对每个 GUI 状态采样 N=5 个响应，根据二元奖励向量将组分为混合命中（含成功与失败）、全失败、全成功三类，分别路由到不同信用分配分支。
- **SCA-Residual（混合组）**：
  - 将坐标投影到水平、垂直、两条对角线共 4 个方向轴。
  - 对每个响应 i，使用其余 N-1 个坐标-奖励对在每轴上拟合 Theil-Sen 线性趋势，预测该位置期望奖励 $\hat{r}_i$。
  - 内层留一验证计算预测技能分数 $s_{i,d}$，加权聚合得 $\hat{r}_i$。
  - 计算残差 $r_i - \hat{r}_i$ 并标准化得 $A_i^{\text{res}}$，与原始 GRPO 优势按预测质量门控权重 $w_i$ 混合：$A_i^{\text{Residual}} = z(w \odot A^{\text{res}} + (1-w) \odot A^{\text{GRPO}})$。
  - 门控权重 $w_i$ 由验证质量分数 $q_i$ 线性映射：$q_i \in [0.80, 0.99]$ 区间内从 0 到 1。
- **SCA-Prox（全失败组）**：
  - 计算每个点击到标注目标中心 $c_T$ 的欧氏距离。
  - proximity 分数 $p_i = \exp(-\|a_i - c_T\|_2 / \ell_T)$，其中 $\ell_T$ 为尺度参数。
  - 信用 $A_i^{\text{Prox}} = \alpha_T \cdot z_{V_T}(p_{V_T})_i$，$\alpha_T$ 按 proximity 标准差缩放。
- **Actor Loss**：使用 clipped PPO 目标，信用标量 $A_i$ 经 stop-gradient 后分配给响应 token：$\mathcal{L}_{\text{actor}} = -\frac{\sum M_{it} \ell_{it}}{\sum M_{it}} + \lambda_{\text{KL}} \widehat{D}_{\text{KL}}$，clip 比率 $\varepsilon_c=0.2$，$\lambda_{\text{KL}}=10^{-2}$。
- ** Dense Reward 扩展**：响应信号 $y_i = 0.8 R_{\text{Gaussian},i} + 0.2 R_{\text{format},i}$，但路由仍由二元指标 $h_i$ 决定。

## 实验与结果
- **基线模型**：GUI-R1-3B、UI-R1-3B（均为 RFT 方法），以及 SeeClick、OS-Atlas、ShowUI、UGround 等 SFT 方法。
- **Grounding 基准**：
  - **ScreenSpot-Pro**：SCA-3B 均值 26.1%，超越 GUI-R1-3B 的 25.2%（+0.9pp），在全部 12 个子列提升 0.3-1.7pp。
  - **ScreenSpot Web**：Text 90.5±0.30 vs 89.6，Icon 73.5±0.42 vs 72.1。
  - **ScreenSpot Desktop**：Text 94.8±0.28 vs 93.8，Icon 66.5±0.45 vs 64.8。
- **Action Prediction 基准**：
  - **GUI-Act-Web GR**：91.2±0.35 vs 89.86（+1.34pp）。
  - **OmniAct-Web GR**：90.1±0.38 vs 88.58（+1.52pp）。
  - **OmniAct-Desktop GR**：93.0±0.32 vs 91.86（+1.14pp）。
  - **Low-level Overall**：82.0±0.36 vs 80.88（+1.12pp）。
  - **GUI-Odyssey SR**：66.0±0.52 vs 64.41（+1.59pp）。
- **精确梯度分析**：SCA-Residual MSE 0.14825，余弦相似度 0.99985；Full SCA MSE 0.14754，余弦 0.99997；对比 Binary GRPO MSE 0.16201，MSE 降低约 8.5%。

## 相关工作脉络
- **GRPO/RLOO 等 Group-Relative RL**：Shao et al. (2024)、Guo et al. (2025)、Kool et al. (2019)，仅使用奖励比较构造信用，无空间参考。
- **GUI-G²**：Tang et al. (2026) 使用高斯奖励建模进行 GUI 接地，SCA 与之不同在于不依赖预定义高斯核，而是从采样组内学习坐标-奖励趋势。
- **SE-GUI**：Yuan et al. (2025) 自演化 RL，SCA 聚焦于训练阶段的信用分配改进，非策略自演化。
- **RSGround-R1**：Huang et al. (2026) 遥感 grounding 的空间推理，领域不同。
- **GUI-R1/UI-R1**：Luo et al. (2025)、Lu et al. (2026)，本文直接超越的 RFT 基线，SCA 作为信用分配改进模块可与之结合。
- **Action-dependent Control Variates**：Gu et al. (2017)、Liu et al. (2018)、Wu et al. (2018)、Tucker et al. (2018)，通过 critic 或解析控制变量降低方差，SCA 无需额外网络。

## 局限性与未来方向
- **仅评估 3B 规模模型与 N=5 采样数**：未测试更大模型或不同组大小的影响。
- **门控校准依赖合成数据**：最佳 q 阈值通过仿真场景选择，真实分布下泛化性待验证。
- **离线固定状态评估**：未测试 OSWorld、WebArena、MiniWoB 等在线交互环境中的恢复能力。
- **分支贡献归因需匹配实验**：当前结果反映整体 SCA，Residual 与 Prox 各自增益需消融验证。
- **图标接地仍为瓶颈**：SCA 对图标目标提升较小（+0.48pp vs 文字 +1.23pp），感知能力仍是限制因素。

## 研究启发与可借鉴点
- **空间参考用于信用分配的设计范式**：将动作坐标作为附加监督信号融入 RL 信用构造，而非修改策略网络结构，可迁移至其他具身/机器人操作任务。
- **留一法交叉拟合防止奖励泄露**：每个响应的预测参考排除自身奖励，避免 overfitting 到当前采样，可推广至其他 group-relative 方法。
- **Regime-based 路由机制**：根据奖励分布特征（混合/全失败/全成功）切换不同信用分支，兼顾信号有用性与数值稳定性。
- **精确梯度对比验证信用估计器**：合成环境下的 MSE、余弦相似度、范数比三维评估，为信用分配方法提供理论可解释性。
- **零推理开销改进**：空间参考仅在训练时计算并丢弃，部署策略不变，工程友好。

## 关键术语表
- **Group-Relative Reinforcement Learning**：从同一状态采样的多个响应间比较奖励分配信用，如 GRPO、RLOO。
- **Theil-Sen Estimator**：基于中位数成对斜率的鲁棒线性回归估计量，对异常值不敏感。
- **Leave-One-Out Cross-Fitting**：预测每个响应时排除其自身奖励，防止奖励泄露到参考值。
- **Credit Assignment**：将环境返回的标量奖励分配给构成轨迹的个体动作或 token。
- **Clipped Actor Loss**：PPO 风格的裁剪策略梯度损失，限制策略更新步长防止崩溃。
- **Grounding Accuracy (GR)**：GUI 智能体点击目标位置坐标与标注区域匹配的成功率。
- **Stop-Gradient (sg)**：阻止梯度回传的算子，使信用标量不影响参考模型。
- **Proximity Score**：基于指数衰减的距离敏感分数，衡量点击位置与目标的接近程度。

## 可复现要素
- **数据集**：ScreenSpot、ScreenSpot-Pro、GUI-Act-Web、OmniAct-Web、OmniAct-Desktop、GUI-Odyssey（论文未明确是否公开，但引用来源可查）。
- **代码/权重**：论文未提及开源；基线 GUI-R1/UI-R1 可能有开源实现。
- **关键超参**：组大小 N=5，clip 比率 ε_c=0.2，KL 系数 λ_KL=10^-2，门控阈值 q_lo=0.80、q_hi=0.99，ρ_lo=0.3、ρ_hi=0.7，密度奖励权重 0.8/0.2。
- **基座模型**：Qwen2.5-VL-3B-Instruct。
- **实现框架**：EasyR1（clipped PPO）。
