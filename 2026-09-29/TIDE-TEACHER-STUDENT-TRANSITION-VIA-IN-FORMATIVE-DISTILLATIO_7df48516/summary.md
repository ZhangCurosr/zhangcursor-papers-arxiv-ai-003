---
title: "TIDE-TEACHER-STUDENT-TRANSITION-VIA-IN-FORMATIVE-DISTILLATIO"
source: https://arxiv.org/pdf/2609.35058v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:12:48"
field: "Agent Reinforcement Learning"
keywords: ["on-policy distillation", "reinforcement learning", "multi-turn agent", "teacher-student transition", "credit assignment", "adaptive scheduling"]
innovations: ["两尺度OPD-RL协调框架：全局分歧触发调度+局部相对价值-分歧调制", "用平滑师生分歧趋势作为自适应切换事件信号替代固定schedule", "相对过程价值与分歧乘积融合实现turn级学习信号分配"]
benchmarks: ["WebShop", "ALFWorld", "SearchQA"]
---

# 论文速读：TIDE-TEACHER-STUDENT-TRANSITION-VIA-IN-FORMATIVE-DISTILLATIO

## 一句话总结
本文提出 TIDE（Teacher-Student Transition via Informative Distillation and Exploration），一种在**全局训练调度**与**局部交互轮次**两个尺度上动态协调 On-Policy Distillation（OPD）与强化学习（GRPO）的混合训练框架，解决固定权重混合导致的小模型早期探索不足和后期能力受限问题。

## 研究问题与动机
1. **稀疏轨迹级奖励限制小模型早期探索**：GRPO 在多轮 Agent 任务中依赖终端奖励，早期探索效率低；OPD 提供 token 级教师指导可缓解此问题，但固定混合比例无法适应训练不同阶段的需求变化。
2. **全局尺度：固定权重假设失效**：随训练推进，持续强蒸馏压力会限制模型超越教师能力；教师监督的边际效用递减，需动态调整 OPD/RL 权重比。
3. **局部尺度：分歧信号歧义**：师生分歧仅标识策略偏离，无法区分"有价值的探索性偏离"与"低质量策略漂移"；需结合动作价值信号进行判别。
4. **混合方法缺乏两尺度协同设计**：现有 ATOD 等方法仅采用预设线性调度，RetireOPD 等采用单一触发机制，缺少全局调度与局部轮次分配的统一框架。

## 核心贡献（创新点）
1. **两尺度 OPD-RL 协调框架**：将混合训练问题形式化为全局教师影响调度与局部更新优先级分配的双重分配问题，区别于单一尺度调度的先前工作。
2. **分歧触发的全局自适应调度**：利用批量级师生分歧的平滑趋势作为事件信号，当正向减少变缓时自动推进 OPD→RL 切换，无需手动指定任务相关切换步数。
3. **相对价值-分歧联合的局部调制机制**：将相对过程价值与响应级分歧结合，生成 OPD turn 优先级分数；相对动作值决定奖励驱动更新方向，分歧归一化后重加权跨轮次信号。
4. **多规模实证验证**：在 WebShop 与 ALFWorld 上，于 1.5B/3B/7B 三个学生尺度均取得最优结果，显著超越 GRPO、OPD、ATOD 等基线。

## 方法详解
**全局控制器（分歧触发切换）**：
- 计算 token 级师生 log-prob 差：$\delta_{t,j} = \log\pi_T(y_{t,j}|h_t, y_{t,<j}) - \log\pi_{old}(y_{t,j}|h_t, y_{t,<j})$
- 聚合为 turn 级差异 $D_t$ 与 batch 级差异 $G_k$，用 EMA 平滑得 $m_k$
- 计算窗口内相对减少量 $I_k = (m_{k-W+1} - m_k) / \max(m_{k-W+1}, \epsilon)$
- 当 $0 < I_k \leq \zeta$ 时，单调递增 handoff 状态 $r_k$，逐步降低 $\lambda_k$（OPD 系数）、提升 $\beta_k$（RL 系数）

**局部调制器（相对价值-分歧联合）**：
- 基于 GiGPO 计算相对过程价值 $Q_t$（匹配历史组内的均值中心化折扣回报）
- 对有效 turn 的 $Q_t$ 做 min-max 归一化得 $\tilde{Q}_t$，对所有 turn 的 $D_t$ 归一化得 $\tilde{D}_t$
- OPD turn 权重：$w_t^{OPD} = z_t = \tilde{Q}_t \times \tilde{D}_t$（高价值+高分歧 turn 获强蒸馏信号）
- RL turn 权重：$w_t^{RL} = \tilde{D}_t$（分歧程度决定奖励更新幅度）
- 组合优势：$A_{t,j}^{TIDE} = \beta_k w_t^{RL} Q_t + \lambda_k w_t^{OPD} \delta_{t,j}$
- 使用 clip PPO 目标优化：$\mathcal{I}(\theta) = \mathbb{E}[\sum_{t,j} \min(\rho_{t,j} A_{t,j}^{TIDE}, \text{clip}(\rho_{t,j}, 1-c, 1+c) A_{t,j}^{TIDE})]$

## 实验与结果
**数据集与设置**：
- WebShop（500任务，30步上限，exact success rate + normalized score）
- ALFWorld（140任务 valid-seen，50步上限，avg success across 6 task types）
- 学生模型：Qwen2.5-Instruct 1.5B/3B/7B，教师为冻结的 GRPO 训练 Qwen2.5-7B
- 超参：lr=1e-6，EMA μ=0.9，窗口 W=10，切换阈值 ζ=0.02，步长 η=0.02，λ∈[0.1,1.0]，β∈[0.1,1.0]，初始 λ₀=1.0，β₀=0.1

**主要结果**（Table 2）：
- **1.5B**：WebShop SR 77.2%（vs ATOD 73.2%，+4.0）；ALFWorld Avg 86.0%（vs ATOD 83.6%）
- **3B**：WebShop SR 79.0%（vs ATOD 74.1%，+4.9）；ALFWorld Avg 89.3%（vs ATOD 86.4%）
- **7B**：WebShop SR 83.2%（vs ATOD 79.0%，SDAR 82.8%）；ALFWorld Avg 92.1%（超越所有对比基线）
- **SearchQA 补充实验**：TIDE 达 39.4% macro-EM，超越 OPD（36.3%）3.1 点

**消融验证**（Table 3）：
- 全局调度：TIDE 超越 Success Handoff、Fixed 1:1、Linear/Cosine 预设调度
- 局部调制：移除 process reward（-5.0点）、移除 disagreement（-2.8点）、Additive Fusion（-9.8点）均显著下降
- 3B 消融复现一致结论

## 相关工作脉络
1. **GRPO（Shao et al., 2024）**：组相对策略优化，轨迹级任务奖励归一化；TIDE 在其基础上引入 OPD 信号并实现两尺度动态协调。
2. **OPD（Agarwal et al., 2024）**：On-policy 蒸馏，用教师 log-prob 辅助学生训练；本文指出其固定权重局限，提出动态调度。
3. **GiGPO（Feng et al., 2026）**：Group-in-Group 策略优化，基于匹配历史的相对过程优势估计；TIDE 借用其相对价值信号作为局部调制基础。
4. **ATOD（Tan et al., 2026）**：预设线性退火调度从 OPD 过渡到 RL；TIDE 改用分歧趋势驱动的自适应切换，性能显著提升。
5. **RetireOPD（Yu et al., 2026）**：分歧 plateau 且学生达到阈值后完全退役教师监督；TIDE 采用渐进式单调过渡而非 abrupt retirement。
6. **SDAR（Lu et al., 2026）**：per-token gating 实现蒸馏与 RL 共享学生-教师策略；TIDE 采用 turn 级权重调制，架构更简洁。

## 局限性与未来方向
1. **局部调制依赖精确历史匹配**：相对价值计算需 exact observation-context grouping，在状态空间极大或连续环境中匹配成功率可能下降。
2. **SearchQA 实验仅测试全局调度**：本地信号分配未在检索增强问答任务中验证，两尺度协同的跨领域泛化性待检验。
3. **教师模型构建成本**：需先用 GRPO 训练 7B 教师模型（一次性开销），虽可复用但限制了快速迭代场景。
4. **超参敏感性**：虽在 ζ, η 五倍范围内表现稳健，但 η 过小可能导致切换过慢，需根据任务调整。
5. **未来方向**：探索语义匹配替代精确字符串匹配以扩大适用场景；将局部调制扩展至多模态 Agent；研究无教师设定下的自蒸馏变体。

## 研究启发与可借鉴点
1. **分歧趋势作为调度信号**：用平滑后的 batch 级师生 log-prob 差异变化率替代固定 schedule 或成功率为触发条件，是一种简洁有效的自适应切换机制，可迁移至其他混合训练场景。
2. **相对价值-分歧乘积融合**：OPD 权重采用 $\tilde{Q}_t \times \tilde{D}_t$ 的乘法融合，强调"高价值且高分歧"的 turn，避免仅靠单一信号导致的误导向，值得借鉴于多信号协调问题。
3. **归一化 fallback 设计**：对无匹配组的 turn 赋予 $\tilde{Q}_t=1$ 作为分歧-only OPD 兜底，保证所有 turn 均有学习信号，鲁棒性强。
4. **实验设计严谨性**：保持 teacher、rollout budget、training budget 完全一致下对比 ATOD vs TIDE，控制变量清晰，消融覆盖全局/局部独立与联合移除。
5. **可结合本团队方向**：若团队关注推理模型训练，可将 TIDE 的两尺度思想应用于 Math/Solver 任务的过程监督与终端奖励混合训练；若关注小模型蒸馏，可尝试将分歧触发机制与知识蒸馏的 curriculum 设计结合。

## 关键术语表
**On-Policy Distillation (OPD)**：从当前学生策略采样的轨迹上，用教师模型的 log-probability 分布提供 token 级蒸馏信号的训练方法。
**GRPO（Group Relative Policy Optimization）**：组相对策略优化，通过组内轨迹级奖励归一化计算优势函数，用于多轮 Agent 训练的 RL 算法。
**相对过程价值（Relative Process Value）**：在匹配历史状态组内，以均值中心化的折扣回报衡量某 turn 动作相对于替代动作的价值差异。
**师生分歧（Teacher-Student Disagreement）**：学生采样 token 上与教师 log-prob 的绝对差值聚合，反映策略偏离程度。
**Global Handoff**：TIDE 全局控制器基于分歧趋势自动调节 OPD/RL 权重比的渐进切换机制。
**Local Modulation**：TIDE 局部调制器基于相对价值与分歧的 turn 级权重分配，决定各交互轮次的学习信号强度。
**Discrepancy-triggered Schedule**：以 EMA 平滑后的分歧减少率 $I_k$ 作为触发条件的事件驱动调度，替代预设线性/余弦 schedule。
**GiGPO（Group-in-Group Policy Optimization）**：基于精确历史匹配构建组内相对优势的 credit assignment 方法，TIDE 借用其价值信号。

## 可复现要素
- **数据集**：WebShop（500任务）、ALFWorld（140任务 valid-seen）、SearchQA（7子集）——均为公开基准
- **代码/权重**：论文未提供开源链接；教师模型为 GRPO 训练的 Qwen2.5-7B checkpoint
- **关键超参**：lr=1e-6，weight_decay=0.01，clip半径 c=0.2，entropy系数=0.001，EMA μ=0.9，窗口 W=10，ζ=η=0.02，λ∈[0.1,1.0]，β∈[0.1,1.0]，γ=0.95，ν=1，history_length=2
- **训练配置**：每 prompt 8 rollouts，160 updates，WebShop minibatch=64，ALFWorld minibatch=256，PPO 每 update 1 epoch
- **评估协议**：WebShop temperature=1.0 top-p=1.0 max 30 steps；ALFWorld temperature=0.4 top-p=1.0 max 50 steps；3 seeds 均值±标准差
