---
title: "VASPO-Self-Distillation-with-Privileged-Value-Estimation"
source: https://arxiv.org/pdf/2610.20516v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 08:56:37"
field: "LLM 推理强化学习"
keywords: ["reinforcement learning", "self-distillation", "value estimation", "mathematical reasoning", "PPO", "privileged information", "token-level credit assignment"]
innovations: ["共享特权信息同步驱动特权 critic 价值估计与自教师 token 调制", "critic 预更新对齐当前策略后再计算 GAE 优势", "有界教师比率在正优势门控下融合，保持优化方向"]
benchmarks: ["AIME24/25/26", "HMMT25", "Minerva Math", "OlympiadBench", "Knights and Knaves"]
---

# 论文速读：VASPO-Self-Distillation-with-Privileged-Value-Estimation

## 一句话总结
论文提出 **VASPO**（Value-Augmented Self-Distillation Policy Optimization），一种在 PPO 框架内共享**特权信息**（如参考答案与解题步骤）同时服务**特权价值估计器（critic）**与**自教师（self-teacher）**的强化学习方法。实验表明，该设计在数学与逻辑推理基准上显著优于基线。

## 研究问题与动机
- **RLVR 的 credit assignment 稀疏性**：基于可验证奖励的强化学习（如 PPO、GRPO）只在回答末尾得到标量回报，难以识别中间 token 的贡献，而组相对策略优化（GRPO）将同一回答的优势广播给所有 token。
- **OPSD 的 token 级指导并非验证器 grounded**：On-policy 自蒸馏虽能在学生生成前缀处提供密集的 token 级信号，但特权信息下的教师偏好容易偏向“风格”而非任务内容，且无法可靠反映学生**无指导续写**的最终成功率。
- **两种信号的位置效应互补**：消融与分析发现，特权信息对 critic 预测精度的提升在回答末尾更显著；而 self-teacher 与学生的分布差异在前端大、中段小、末端回升，提示两者应分设职责。
- **现有工作未将同一特权信息同时用于价值估计**：如 RLSD 等方法仅用特权信息做 teacher–student 信号调节，未探索用它训练一个**PPO 风格的续写价值估计器**，从而割裂了“局部 token 指导”与“全局轨迹回报”的联合建模。

## 核心贡献（创新点）
- 刻画特权信息在不同响应位置的作用曲线：**critic 收益随位置向后增长，self-teacher 分歧呈“高→低→回升”的非单调形态**。
- 提出 **VASPO**：共享固定特权信息给特权 critic 与自教师；critic 预测学生无指导续写的预期回报以计算 token-level PPO 优势，自教师通过有界概率比调制选中正优势，actor 仍仅以原始 prompt 为输入。
- 与已有工作的本质区别：不同于 GRPO/RLSD 等先用组相对优势再调节的方式，VASPO 在 **PPO 路径**中将特权信息直接融入价值函数，使优势随前缀变化并在教师调制前保持验证器 grounding。
- 在 6 个数学推理基准与 Knights and Knaves 逻辑推理上验证：相较 PPO，**Qwen3-4B 均值提升 +2.59pp，OLMo3-7B 提升 +1.81pp**，并在 KK 上取得最高均值准确率。

## 方法详解
- **共享特权信息与 rollouts**：采样前固定特权信息 $z$（解法与答案），actor 仅以 prompt $x$ 采样响应 $y$；在每一前缀 $h_t=(x,y_{<t})$，特权 critic 与 detached 自教师同时看到 $(h_t, z)$。
- **特权价值估计（critic）**：
  - 目标：$V_\phi^z(h_t,z)\approx\mathbb{E}_{\pi_{\theta_{\mathrm{old}}}}[G_t\mid h_t,z]$，即学生无指导续写的成功概率/期望回报。
  - 在每个 rollout batch 内**先更新 critic**：以 verifier 终端回报 $G_i$ 为监督（$\gamma=1$，所有有效 token 同目标），采用 PPO 裁剪价值损失 $\mathcal{L}_V$。
  - 更新后重新计算 GAE 优势；因终端回报且 $\gamma=\lambda_{\mathrm{GAE}}=1$，优势退化为 $A_{i,t}^z=G_i-v_{i,t}^{z,+}$，并 detach 后用于 actor。
- **教师引导的 PPO 更新**：
  - 在 student 行为分布 $p_{\mathrm{old},t}$ 上计算 detach 的教师–学生 log-ratio 差距 $d_t=\mathrm{sg}[\log q_{\bar\theta,t}^z(y_t)-\log p_{\mathrm{old},t}(y_t)]$，裁剪得权重 $w_t=\mathrm{clip}(e^{d_t},1-\epsilon_w,1+\epsilon_w)$。
  - Token 级门控掩码 $M_t=\mathbf{1}[A_t^z>0]\cdot\mathbf{1}[|\tanh(d_t/\tau)|>\delta]$，只对**正优势且师生差距足够大**的 token 启用调制。
  - 融合优势 $A_t^{z,\mathrm{f}}=\mathrm{sg}[A_t^z(1+M_t(w_t-1))]$：选中标量放大/衰减，其余不变，保证符号不被翻转。
  - Actor 使用标准 PPO 裁剪代理损失，critic 仅在本 batch 执行一次 pre-update，不做第二次价值更新。
- **关键实现**：教师用学生参数的 EMA（$\beta=0.95$）；advantage 不做全局 whitening（消融表明会翻转符号）；教师评分一次性缓存。

## 实验与结果
- **数据集与模型**：数学训练 DeepMath-103K；逻辑训练 Knights and Knaves（4–8 roles）。推理模型 Qwen3-4B-Instruct-2507 与 OLMo3-7B-Instruct-DPO。
- **基线**：Vanilla、DAPO、PPO、OPSD、RLSD。
- **评估指标**：数学 avg@32（AIME24/25/26、HMMT25）与 avg@8（Minerva Math、OlympiadBench）；KK avg@16 的 ID 与 OOD（9/10/11 roles）。
- **主要结果**：
  - **数学均值**：Qwen3 上 VASPO **53.17%** vs PPO **50.58%**（+2.59pp）；OLMo3 上 **33.42%** vs PPO **31.61%**（+1.81pp），在多数子任务上取得最佳/次佳。
  - **逻辑推理**：KK 上 VASPO 均值 **60.81%**，超过所有对比方法；OOD 11-role 上亦领先。
  - **训练动态**：VASPO 与基线训练奖励相近，但**held-out 准确率与 critic 解释方差（EV）均更高**。
- **消融要点**：完整优于仅特权 critic/仅教师调制；token 掩码、完整解法输入、critic pre-update 与合理 $\epsilon_w=0.2$ 均必要；后融合 whitening 有害。

## 相关工作脉络
- **RLVR/PPO/GRPO**：以终端验证奖励为主，区别在于 VASPO 将特权信息引入 PPO 价值函数而非仅作教师分支。
- **OPSD/自蒸馏**：传统 OPSD 仅用特权信息做 token 分布匹配；VASPO 进一步让同信息服务价值估计，并用门控防止风格漂移。
- **RLSD**：保留 outcome 优势符号并用 teacher–student 证据比调节幅度；VASPO 改用**learned critic 的 prefix-dependent 优势**，再经有界教师调制。
- **GenAC/VC-PPO/VAPO/ReDiPPO**：改进 critic 结构或长度自适应；VASPO 关注 critic 的**输入侧**——显式接入特权信息并以预更新对齐当前 policy。
- **SRPO/RLCSD/SSPO**：基于样本路由或步骤级权重的混合；VASPO 不在样本级路由，而在**token 级 fuse**，且保持统一 PPO 更新。
- **MiMo-V2-Flash/SDPG/SAF-OPD**：多为 additive 目标或外部教师；VASPO 用**单一特权 bank**同步驱动两路读者，降低多信号耦合复杂性。

## 局限性与未来方向
- **交互/多步环境不适配**：当前不允许用本条响应结束时的环境反馈回溯作为同轨迹早期前缀的 privileged context。
- **特权信息质量依赖**：即使最终答案正确，中间步骤仍可能含噪声，污染价值估计或教师信号。
- **上下文与计算开销**：solution+answer 拉长 reader 输入，增加训练成本；rollout 是最大瓶颈。
- **单轮生成**：尚未扩展到多轮对话中历史与证据动态演化的场景。
- **校准与点估计的权衡**：特权条件提升点估计（EV 更高）但未必改善校准（ECE），可能影响不确定性感知。

## 研究启发与可借鉴点
- **位置效应分析可复用**：先绘制“特权信息对不同读者贡献的位置曲线”，再决定信号投放策略，避免盲目融合。
- **共享特权 bank 的双路径范式**：同一 $z$ 同时服务 value function 与 teacher，可作为后续多信号 LLM-RL 的通用模板。
- **Critic pre-update 机制**：在 actor 计算优势前先对齐最新策略轨迹，能缓解 distribution shift；适合所有 critic-based RLVR 流程。
- **符号保持的有界调制**：通过 $A^z>0$ 门控与 clipped ratio 融合，可在不翻转优化方向的前提下引入先验偏好，工程稳定。
- **警惕后融合 whitening**：正权重不能保证中心化的符号安全；若需归一化应在融合前或采用分组归一化。

## 关键术语表
- **RLVR**：Reinforcement Learning with Verifiable Rewards，用可自动核查的终态奖励进行策略优化的训练范式。
- **GRPO**：Group Relative Policy Optimization，组内相对优势估计，无需 learned critic。
- **PPO**：Proximal Policy Optimization，通过 clipped surrogate loss 限制策略更新步长的 actor-critic 方法。
- **OPSD**：On-Policy Self-Distillation，学生在自身轨迹上向带特权信息的自教师做 token 级蒸馏。
- **VASPO**：Value-Augmented Self-Distillation Policy Optimization，本文提出的共享特权信息同时服务价值估计与自教师的 PPO 框架。
- **Privileged Information**：训练时额外提供的解法/答案等上下文，部署时不可用。
- **GAE**：Generalized Advantage Estimation，以 $\lambda$ 控制偏差-方差的时序差分优势估计。
- **Explained Variance (EV)**：critic 预测对回报方差的解释比例，越高说明价值估计越准确。

## 可复现要素
- **代码**：已开源，见 https://github.com/cii030/VASPO。
- **数据集**：DeepMath-103K、Knights and Knaves、AIME/HMMT/Minerva/OlympiadBench 等均有公开链接（见附录 Table 4）。
- **模型**：Qwen3-4B-Instruct-2507、OLMo3-7B-Instruct-DPO、CompassVerifier-3B 均公开可用。
- **关键超参**：$\gamma=\lambda_{\mathrm{GAE}}=1$，$\epsilon_{\mathrm{clip}}=0.2$，$\epsilon_w=0.2$，$\tau=\delta=0.02$，EMA $\beta=0.95$，lr 1e-6/5e-6，batch 128/32，responses/prompt=8，max len 8192，200 步更新。
