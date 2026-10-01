---
title: "ReSPO-Reshaped-Sequence-Policy-Optimization-for-Gradient-Sta"
source: https://arxiv.org/pdf/2609.35433v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:20:07"
field: "大语言模型强化学习"
keywords: ["RLVR", "off-policy optimization", "gradient starvation", "sequence-level policy optimization", "Reinforcement Learning from Verifiable Rewards", "policy gradient"]
innovations: ["揭示clipping在正负advantage上的不对称梯度饥饿问题", "基于α-散度变分目标推导两分支平滑核函数,正分支保留W→0处梯度,负分支同时抑制两端尾部", "通过KL投影实现指数方差控制倾斜,参数由导数匹配条件解析确定无需网格搜索"]
benchmarks: ["DAPO-MATH-17k", "AIME 2025", "AIME 2024", "AMC 2023", "OlympiadBench", "MinervaMath", "MATH-500"]
---

# 论文速读：ReSPO - Reshaped Sequence Policy Optimization for Gradient Starvation in Off-Policy Learning

## 一句话总结
论文针对 RLVR 中 rollout 复用导致的离策略漂移问题，发现现有 clipping 策略在正负 advantage 上存在不对称的梯度饥饿，提出 ReSPO——一种基于 α-散度变分目标的平滑两分支序列级重要性权重重塑方法，在不依赖价值函数的情况下实现更稳定的离策略策略优化。

## 研究问题与动机
- **Rollout 复用引发严重离策略偏差**：RLVR 训练中为节省计算成本复用旧 rollout，使当前策略 $\pi_\theta$ 与数据生成策略 $\pi_{\text{old}}$ 产生显著漂移，重要性权重 $W$ 偏离 1。
- **Clipping 方法的梯度饥饿（Gradient Starvation）**：GRPO 和 GSPO 的硬截断导致不对称问题：正响应（$\hat{A} > 0$）的低权重尾部梯度被过度压缩，而负响应（$\hat{A} < 0$）的高权重尾部未被抑制，极端情况权重无界放大。
- **长推理轨迹信号丢失**：早期训练中蕴含关键学习信号的长正响应因累积策略漂移落入低权重尾部，clipping 使其梯度贡献几乎为零，阻碍快速收敛。
- **序列级修正的必要性**：token 级重要采样无法精确对齐序列级 reward，GSPO 的长度归一化（$W^{1/|o|}$）引入长度相关的几何畸变。

## 核心贡献（创新点）
1. **形式化离策略梯度饥饿问题**：首次系统分析 clipping 在 sign-dependent 梯度分配上的缺陷，揭示正/负响应对重要性权重尾部的相反失败模式。
2. **提出 ReSPO 平滑两分支核函数**：基于 α-散度变分目标推导序列级重塑核 $\phi^{(\pm)}(W)$，正分支 $\alpha^{(+)}>1$ 保留 $W\to 0$ 处非零梯度，负分支 $\alpha^{(-)}=1$ 同时抑制两端尾部。
3. **设计指数方差控制倾斜**：通过 KL 投影将线性矩约束转化为平滑乘法倾斜，区别于 VESPO 对原始权重施加指数惩罚的方式。
4. **参数解析推导而非超参搜索**：利用导数匹配条件（$W=1$ 处斜率连续）和尾部行为约束解析确定超参，无需大规模网格搜索即可达到良好性能。
5. **在密集模型和 MoE 架构上均验证有效性**：在 Qwen3-1.7B 和 Qwen3-30B-A3B 上，ReSPO 在所有 N∈{8,16,32} 设置下获得最高晚期训练分数，并在 AIME/AMC 等竞赛基准上实现最大提升（30B 模型平均提升 6.3pp/5.8pp/3.4pp）。

## 方法详解
- **问题建模**：定义序列级重要性权重 $W = \prod_t w_t = \pi_\theta(o|q)/\pi_{\text{old}}(o|q)$，将 reshaping 视为测度变换 $\tau(o) = \mu(o)\phi(W(o))$。
- **双支核函数设计**：
  - 正分支（$\hat{A}\geq 0$）：取 $\alpha^{(+)}=2$，得到 $\phi_0^{(+)}(W)=\frac{1+W}{2}$（算术平均），结合指数倾斜 $\phi^{(+)}(W)=\frac{1+W}{2}\exp(1-W)$，满足 $\phi^{(+)}(0)=e/2\approx 1.36>0$，$\phi^{(+)}(\infty)=0$。
  - 负分支（$\hat{A}< 0$）：取 $\alpha^{(-)}\to 1$ 极限，$\phi_0^{(-)}(W)=\sqrt{W}$（几何平均），结合指数倾斜 $\phi^{(-)}(W)=\sqrt{W}\exp(2(1-\sqrt{W}))$，满足 $\phi^{(-)}(0)=\phi^{(-)}(\infty)=0$，峰值在 $W=1/4$。
- **变分推导**：通过 power-mean interpolation 构造 $\tau_0^*=[(1-\beta)\mu^{\alpha-1}+\beta\pi^{\alpha-1}]^{1/(\alpha-1)}$，再利用 KL 投影施加二阶矩约束，得到 $\phi(W)=\phi_0(W)\exp(\lambda(1-\phi_0(W)))$。
- **梯度更新公式**：$\nabla_\theta \mathcal{I}_{\text{ReSPO}}=\mathbb{E}[\frac{1}{G}\sum_i \text{sg}(\phi(W_i;\hat{A}_i))\cdot\hat{A}_i\cdot\sum_t\nabla_\theta\log\pi_\theta(o_{i,t}|q,o_{i,<t})]$，其中 stop-gradient 确保权重仅作为标量系数。
- **数值稳定实现**：log W 截断到 $[-20,20]$，正分支直接计算，负分支在 log-space 计算以避免溢出。

## 实验与结果
- **数据集**：DAPO-MATH-17k（训练），AIME 2025/2024、AMC 2023、OlympiadBench、MinervaMath、MATH-500（评测）。
- **模型**：Qwen3-1.7B-Base（密集）和 Qwen3-30B-A3B-Base（MoE）。
- **基线**：GRPO（token 级 clip）、GSPO（序列级 clip）、VESPO（KL 变分软优化）。
- **关键超参**：rollout 复用比 N∈{8,16,32}，学习率 $10^{-6}$，mini-batch M=32，G=8。
- **训练性能**：ReSPO 在 6 组设置中 5 组取得最高早期峰值，5 组取得最高晚期均值；N=32 时 1.7B 提升 7.9pp，30B 提升 8.4pp。
- **评测性能**：N=16 时 ReSPO 在 12 组模型-基准比较中获 9 次最优；N=32 时获 8 次最优；竞赛题（AIME25/AIME24/AMC23）30B 模型平均超越最强基线 6.3pp/5.8pp/3.4pp。
- **响应长度分析**：在 30 个长度分箱中，ReSPO 在 24 个分箱取得最高准确率；N=32 时短中期响应领先 8.4–24.4pp。
- **鲁棒性**：3 次独立 seed 实验，1.7B 晚期均值 27.07%（σ=0.54pp），30B 均值 57.92%（σ=0.95pp），标准差远小于性能增益。
- **与 Routing Replay 组合**：在 30B+N=16 设置下，ReSPO+R3 晚期分数从 56.32% 提升至 58.16%。

## 相关工作脉络
- **GRPO/GSPO**：本文指出的 clipping 梯度饥饿问题的直接对比对象，前者 token 级后者序列级，均存在正负响应的不对称处理缺陷。
- **VESPO (Shen et al., 2026)**：同为变分软优化，但使用 KL 散度且指数因子作用于原始权重 W，导致正分支在 $W\to 0$ 处梯度饥饿。
- **REAL (Zhai et al., 2026)**：将梯度饥饿重新框架化为分类问题；本文保留重要性采样以显式建模离策略陈旧性。
- **Routing Replay (Ma et al., 2025)**：解决 MoE 训练-推理 router 不匹配；与 ReSPO 正交，可组合使用。
- **TRPO/PPO**：传统 trust region 方法，PPO 的硬 clip 是本文要改进的核心机制；TRPO 的 KL 约束与本文变分思路有概念联系。
- **DAPO (Yu et al., 2025)**：开源 RL 系统，本文在其 DAPO-MATH 数据上评估；DAPO 的动态采样与本工作的 off-policy 场景相关。

## 局限性与未来方向
- **计算资源限制**：受限于预算，30B 实验仅运行 8,192 token 响应上限和固定 1024 步更新；更长训练和更大模型需更多 GPU 资源（估计需 8×H200 数周）。
- **超参数未充分搜索**：当前参数由解析条件确定（$\alpha^{(+)}=2, \beta=0.5, \lambda=2$），未进行网格搜索验证是否存在更优配置。
- **未探索更多 off-policy 设定**：仅测试 N∈{8,16,32}，未验证极端漂移下的行为。
- **单一 reward 类型**：仅在数学推理的 binary reward 场景验证，对 continuous reward 或复杂 verifier 的泛化性待检验。
- **未结合 critic-based 方法**：如 PPO 的 value function 与 ReSPO 的交互效果未研究。

## 研究启发与可借鉴点
- **sign-dependent kernel 设计范式**：正负 advantage 需采用不同 reshaping 策略的思想可迁移至其他 off-policy 场景（如对话 RL、agent planning）。
- **变分框架的几何解释**：用 α-散度族统一刻画不同 tail 行为，为设计软 clipping 提供系统性框架，可推广至其他 divergence 选择。
- **导数匹配确定超参**：利用 $W=1$ 处斜率连续性解析确定 λ，避免超参搜索开销，这一思路可应用于其他 policy gradient 方法的参数设计。
- **序列级 vs token 级权衡分析**：GSPO 长度归一化的几何畸变分析（附录 A.6）提供了理论工具，可用于诊断其他序列级优化方法。
- **与 Routing Replay 的正交性**：证明重要性权重控制与 system-level 稳定性技术可解耦组合，为后续工作提供集成思路。

## 关键术语表
- **RLVR (Reinforcement Learning from Verifiable Rewards)**：利用确定性验证器（如数学答案匹配）计算标量奖励的强化学习方法。
- **Gradient Starvation (梯度饥饿)**：clipping 导致重要性权重低/高尾部梯度信号被过度压缩或无限放大的不对称问题。
- **Sequence-level Importance Ratio (序列级重要性比率)**：整条响应的概率比 $W=\pi_\theta(o|q)/\pi_{\text{old}}(o|q)$，聚合 token 级比率。
- **α-divergence (α-散度)**：广义散度度量族，通过参数 α 控制 interpolating 行为；本文用于推导 power-mean reshaping kernel。
- **Exponential Tilt (指数倾斜)**：通过 KL 投影施加方差约束后得到的乘法修正因子 $\exp(\lambda(1-\phi_0(W)))$。
- **Rollout Reuse (Rollout 复用)**：重复使用同一批采样响应进行多次策略更新以提高样本效率，但加剧 off-policy 漂移。
- **Stop-gradient (Stop-grad)**：在计算图中标记某变量不参与梯度传播，使 $\phi(W)$ 仅作为标量系数而非可微函数。
- **Power-mean Interpolation (幂均值插值)**：形式为 $[(1-\beta)\mu^{\alpha-1}+\beta\pi^{\alpha-1}]^{1/(\alpha-1)}$ 的测度插值，α 控制 tail 行为。

## 可复现要素
- **数据集**：DAPO-MATH-17k（公开，huggingface），评测集 AIME/AMC/OlympiadBench/MinervaMath/MATH-500 均为公开数据集。
- **代码**：论文声明附带 code artifact，包含 ReSPO 实现及密集/MoE 实验 launch 配置；项目页面标注为 "Project Page: ReSPO"。
- **模型权重**：使用 Qwen3-1.7B-Base 和 Qwen3-30B-A3B-Base（需从官方获取）。
- **关键超参**：$\alpha^{(+)}=2, \alpha^{(-)}=1, \beta^{(+)}=\beta^{(-)}=0.5, \lambda^{(+)}=\lambda^{(-)}=2$；学习率 $10^{-6}$，gradient clip norm=1.0，weight decay=0.1。
- **训练配置**：verl 框架 + vLLM 异步 rollout，FSDP（1.7B）/Megatron（30B），mini-batch M=32，G=8，1024 次策略更新。
