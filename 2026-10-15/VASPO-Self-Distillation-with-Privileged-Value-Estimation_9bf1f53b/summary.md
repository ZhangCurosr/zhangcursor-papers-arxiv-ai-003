---
title: "VASPO-Self-Distillation-with-Privileged-Value-Estimation"
source: https://arxiv.org/pdf/2610.20516v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 08:56:41"
field: "推理模型强化学习训练"
keywords: ["reinforcement learning", "self-distillation", "privileged information", "value estimation", "mathematical reasoning", "PPO", "token-level supervision"]
innovations: ["共享特权信息同时用于privileged critic延续值估计与self-teacher token调制，actor保持prompt-only", "揭示critic收益随响应位置递增、self-teacher KL早期高期末回升的互补位置动态，驱动双读者分工设计"]
benchmarks: ["AIME 2024/2025/2026", "HMMT 2025", "Minerva Math", "OlympiadBench", "Knights and Knaves", "DeepMath-103K"]
---

# 论文速读：VASPO-Self-Distillation-with-Privileged-Value-Estimation

## 一句话总结
VASPO提出了一种基于PPO的自蒸馏框架，通过共享特权信息（参考解法）同时驱动critic的延续值估计与self-teacher的token级指导，在数学与逻辑推理任务上显著提升了推理模型的准确率。

## 研究问题与动机
- RLVR（可验证奖励强化学习）仅能提供最终答案的稀疏奖励，无法为中间推理步骤分配信用；GRPO广播组内优势，PPO学习critic但两者都依赖后置奖励。
- 在机自我蒸馏（OPSD）通过特权信息提供密集token级监督，但其信号反映的是局部token偏好，而非学生独立继续生成的预期成功率，且可能偏向风格而非内容。
- 近期融合方法（RLSD、SRPO等）仅在teacher分支使用特权信息，未探索用相同信息训练PPO风格的prefix-value估计器，导致"延续值估计"这一用途未被挖掘。
- 实验观察到两种读者的位置动态互补：critic预测收益随响应位置靠后而增长，self-teacher分歧早期高、中期低、期末回升，提示应将特权信息同时用于两类任务。

## 核心贡献（创新点）
- **刻画特权信息的位置动态**：首次系统展示critic受益于参考解法的位置 profile（后期提升明显）与self-teacher KL divergence profile（前期高、中期低、期末回升）的互补特征，为双读者分工提供实证依据。
- **提出VASPO框架**：首次让同一份特权信息同时服务于privileged critic（估计学生无引导续写的预期回报）与privileged self-teacher（调制token优势），actor始终保持prompt-only，实现token引导与延续值估计的协同。
- **empirical验证提升**：在六个数学推理基准上，VASPO相较PPO在Qwen3-4B上平均准确率提升2.59pp、OLMo3-7B上提升1.81pp，并在Knights and Knaves逻辑推理任务上取得最高平均分。

## 方法详解
- **整体架构**：VASPO基于PPO，保留clipped actor update，引入两个privileged reader（critic与self-teacher），二者在相同学生生成前缀处访问固定的特权信息z（如参考解法+答案），actor仅接收标准问题提示。
- **Privileged Value Estimation**：Critic输入为(prefix, z)，预测目标为学生old policy续写的期望回报E[G_t | h_t, z]；每轮rollout前对critic执行一次pre-update（PPO clipped value loss，L_V = max((v^z - G)^2, (v̄^z - G)^2)），之后用刷新后的V_φ⁺^z重新计算GAE优势A^z_t = G_i - v^z,+_{i,t}（γ=λ_GAE=1时退化为单一term）。
- **Teacher-Guided Reweighting**：Self-teacher（EMA参数θ̄，decay=0.95）计算detach后的student-teacher log-prob gap d_t = sg[log q^z_θ̄(y_t) - log p_θ_old(y_t)]，经clip到[1-ε_w, 1+ε_w]得到权重w_t。
- **Token-level Mask**：仅对A^z_t > 0且|tanh(d_t/τ)| > δ的token应用teacher调制，避免对负优势或teacher-student接近的token施加噪声。
- **Fused Advantage**：A^z,f_t = sg[A^z_t(1 + M_t(w_t - 1))]，保留原始优势符号，selected tokens被缩放，其余保持不变；最终用detached A^z,f_t进行标准PPO actor更新。

## 实验与结果
- **数据集**：数学推理使用DeepMath-103K训练，测试集为AIME 2024/2025/2026、HMMT 2025、Minerva Math、OlympiadBench；逻辑推理使用Knights and Knaves（4–8角色训练，9/10/11角色OOD测试）。
- **基线**：Vanilla、DAPO、PPO、OPSD、RLSD。
- **主要结果**：Qwen3-4B上VASPO平均准确率53.17%，较PPO（50.58%）提升2.59pp，较DAPO（51.59%）提升1.58pp；OLMo3-7B上VASPO平均33.42%，较PPO（31.61%）提升1.81pp；Knights and Knaves上VASPO平均60.81%，为所有方法最高。
- **消融结论**：Privileged critic alone优于PPO；去除token-level mask或pre-update均导致全面下降；优势whitening显著损害性能（因中心化处理会翻转优势符号）；仅使用答案而不用完整解法会降低准确率。

## 相关工作脉络
- **Vapnik & Vashist (2009) SVM+**：最早提出"Learning using Privileged Information"范式，将训练时可用但推理时不可用的额外信息用于增强学习器；VASPO沿袭此思想但应用于RLVR+自蒸馏融合场景。
- **Schulman et al. (2017) PPO / Shao et al. (2024) GRPO**：两类主流RLVR基线；VASPO选择PPO而非GRPO作为骨架，因critic价值估计更能利用prefix-level特权信息。
- **Zhao et al. (2026) OPSD**：在机自我蒸馏开创者，仅提供token-level divergence loss；VASPO在其基础上引入critic路径，使特权信息同时服务价值估计。
- **Yang et al. (2026a) RLSD**：将teacher-student ratio调制到GRPO优势上，但优势仍为group-relative且无prefix-value学习；VASPO的advantage来自learned privileged critic，粒度更细。
- **Zhang et al. (2026c) ReDiPPO**：用reference模型校准value estimate，但侧重于discrepancy-aware reweighting；VASPO区分了critic（延续值估计）与teacher（token评分）的双重角色分工。
- **Li et al. (2026) SRPO / Pan et al. (2026) RLCSD**：通过sample routing或contrastive机制在GRPO与自蒸馏间切换；VASPO选择joint fusion而非routing，保持两者同时参与每次更新。

## 局限性与未来方向
- 当前协议下，响应完成后获得的环境反馈无法回溯作为同一条轨迹早期prefix的privileged context，限制了交互式场景的应用。
- 特权参考解法若含错误中间步骤会引入噪声监督；完整解法扩展了teacher和critic的context window，增加训练开销。
- 仅支持single-turn生成；扩展至multi-turn dialogue（对话历史与可用特权证据逐轮动态演化）仍是开放方向。

## 研究启发与可借鉴点
- **双读者分工设计**：同一份特权信息可同时用于" outcome-grounded value estimation"和" token-level guidance"两个正交任务，此思路可迁移至其他需要区分局部/全局信号的RL设置。
- **Critic pre-update机制**：在actor更新前先冻结rollout并单独更新critic，可缓解policy shift导致的value estimate滞后；适用于任何actor-critic架构中对critic时效性敏感的场景。
- **Token-level mask with raw advantage sign**：用未标准化的A_t^z符号和tanh门控进行选择，避免whitening等后处理意外翻转优势符号；这一安全约束对任何fusion式方法均有参考价值。
- **位置profile分析驱动设计**：通过可视化不同位置上的信号动态（如图2、图8）来指导模块分工，而非凭直觉叠加组件；可推广到其他序列生成任务的训练诊断。

## 关键术语表
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：利用可自动验证的最终答案作为二值奖励信号的强化学习训练范式。
- **OPSD (On-Policy Self-Distillation)**：使用特权信息（如参考解法）在同一模型上构建detach的self-teacher，沿学生生成轨迹提供token级监督。
- **Privileged Information**：训练时可用但推理时不可用的额外上下文（如题目完整解法与答案），源自Vapnik的"learning using privileged information"框架。
- **GAE (Generalized Advantage Estimation)**：通过调节λ在偏差-方差之间权衡的advantage估计方法，此处γ=λ=1时退化为G_t - V(h_t)。
- **Explained Variance (EV)**：衡量critic预测解释返回变化比例的诊断指标，越高表示价值估计越准确。
- **Token-level Mask**：基于A_t^z > 0且|tanh(d_t/τ)| > δ的布尔条件，控制哪些token接受teacher调制。
- **Privileged Critic**：同时接收学生prefix和特权信息的critic，预测学生无引导续写的期望回报。
- **Self-teacher (EMA)**：通过指数移动平均同步student参数的detach teacher，用于计算token级概率gap与调制权重。

## 可复现要素
- **代码**：已开源，https://github.com/cii030/VASPO
- **数据集**：DeepMath-103K（MIT license, 103,022题）、Knights and Knaves（CC BY-NC-SA 4.0, 6900题）、AIME 2024/2025/2026（Apache-2.0）、HMMT 2025（CC BY-NC-SA 4.0）、Minerva Math（未注明）、OlympiadBench（未注明）——均可公开获取。
- **模型**：Qwen3-4B-Instruct-2507（Apache-2.0）、OLMo3-7B-Instruct-DPO（Apache-2.0）
- **关键超参**：global batch=128，mini-batch=32，responses/prompt=8，lr_actor=1e-6，lr_critic=5e-6，ε_clip=0.2，ε_w=0.2，τ=δ=0.02，β_EMA=0.95，γ=λ_GAE=1，value-loss coeff=0.5，rollout updates=200，max length=8192，warmup 50 steps。
