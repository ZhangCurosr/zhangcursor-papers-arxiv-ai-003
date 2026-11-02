---
title: "TEV-SCALING-LONG-HORIZON-AGENTIC-ENVIRONMENTS-WITH-DENSE-PRO"
source: https://arxiv.org/pdf/2610.18868v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 09:05:55"
field: "长程 Agent 训练"
keywords: ["长程智能体", "强化学习", "奖励塑造", "合成数据", "Terminal-Bench", "GRPO", "潜在世界"]
innovations: ["latent-world-first 合成管线：以单一潜在世界 W 统一生成任务、验证器和进展信号，从根本上消除 artifact drift", "PPRS 稠密奖励塑造：基于 DAG 节点进展潜力 Φ 将每步奖励转为非零方差信号，修复 GRPO 在长程全失败组中的梯度枯竭问题", "TEV-GYM 首个长程渐进奖励训练数据集：1,170 个跨 7 领域任务，median 12.4 min/rollout，带 8 项自动生成资格门验证"]
benchmarks: ["ALE-CM", "Terminal-Bench 3.0-CM", "Terminal-Bench 4.0-CM", "Terminal-Bench 2.1"]
---

# 论文速读：TEV: SCALING LONG-HORIZON AGENTIC ENVIRONMENTS WITH DENSE PROGRESS REWARDS

## 一句话总结
本文提出了**潜在世界优先（latent-world-first）**的任务合成方法与**进展潜力奖励塑造（PPRS）**的训练配方，构建了首个含可验证稠密渐进奖励的长程智能体训练数据集 TEV-GYM（1,170 个跨 7 领域任务环境），使得在 ALE-CM、Terminal-Bench 等极长时距任务上训练开源模型成为可能，TEV-27B 在 Pass@5 上追平 GPT-6 Astra 和 Claude Opus 5。

## 研究问题与动机
- **开源长程智能体缺乏训练数据**：现有 agentic 训练语料库（SWE-Smith、TMAX、SETA 等）规模达数万，但每个任务仅涉及 PR 级修复或短会话；真实长程任务仅存在于人工评测集（ALE、TB），规模数十至数百条，不可用于训练。
- **长程任务难以构造与验证**：任务需足够长且结构复杂才能提供有效训练信号，同时必须可精确验证（verifiable），二者存在内在张力。
- **稀疏奖励导致群体相对优化（GRPO）梯度枯竭**：当一组 $G$ 次 rollout 全部失败时，终端奖励方差为零，advantage 全为 0，梯度完全消失；长程任务初始训练阶段这种全失败组占比极高。
- **数据合成存在 artifact drift 风险**：分别用模型生成任务、输入、checker 的流水线容易出现三者描述不一致的问题，需要统一锚点。

## 核心贡献（创新点）
- **提出 latent-world-first 合成管线**：先生成一个完整正确的"潜在世界" $W$，再反向渲染为公开任务文件并施加可控损坏，使得任务、验证器、进展信号均为 $W$ 的函数，从根源上消除 artifact drift。与已有工作本质区别在于所有组件共享单一 ground truth 而非分别生成后再对齐。
- **构建首个含稠密渐进奖励的长程训练数据集 TEV-GYM**：1,170 个经过 8 项资格门（qualification gates）自动验证的环境，跨 7 个领域，每任务中位时长约 12.4 分钟（前端 agent），为现有语料库长度的约 6 倍，且在 PASS@1 上最低（29% by Gemini 3.6 Flash），体现更高难度与真实性。
- **提出 PPRS（Progress-Potential Reward Shaping）**：基于潜在世界提供的进展潜力 $\Phi(s) \in [0,1]$，将每步奖励设为 $\lambda r_t = \lambda [\Phi(s_t) - \Phi(s_{t-1})] + \mathbb{1}[\text{pass}]$，使得即便整组 rollout 全部失败的子群也能在第一轮就产生非零 advantage，从根本上修复 GRPO 的梯度枯竭问题。
- **展示跨规模泛化与迁移能力**：TEV-27B 在 ALE-CM 上 Pass@5=0.28 追平 GPT-6 Astra 与 Claude Opus 5；在 TB 4.0-CM 上 Pass@1 提升 150% 超越 GLM 5.2 和 Gemini 3.7 Flash；同一配方应用于 GLM-5.3-Flash 得到 TEV-320B，进一步拉开差距。

## 方法详解
**1. 潜在世界优先任务合成（§2.1）**
- **Stage 0 任务源**：以 ALE-CM 的 7 个子领域（软件工程、数据工程、AI/CS 研究、数学与运筹学、安全与取证、基础设施管理、量子计算）为覆盖目标，随机采样规范并编写 seedable generator。
- **Stage 1 世界构造**：给定 seed 和 scale tier，程序化生成完整的潜在世界 $W$（如仓库客户、产品、交易记录），并校验不变量（唯一 ID、有效 join、有限度量）。同时生成 defect plan，规定哪些记录将被损坏。
- **Stage 2 DAG 工作流图**：将环境工作流编译为 DAG，关键路径定义 horizon，节点构成后续进展潜力的骨架；纯代码生成，无模型参与。
- **Stage 3 产物渲染**：将 $W$ 投影为公开文件（CSV/TSV/JSON/SQLite/safetensors 等），并断言 round-trip 一致性（从文件可精确恢复 $W$）。
- **Stage 4 可控损坏**：按 defect plan 注入 schema drift、重复/缺失记录、shard 索引混乱等，得到 agent 收到的初始状态 $s_0$。
- **Stage 5–6 公开规范与隐藏测试套件**：生成 agent 可见的 task prompt $x$，以及包含 metamorphic transforms 和 stale-output counterfactuals 的隐藏验证套件。

公式 (1)：$E = (x, s_0, \text{ver}_W, \Phi_W)$，所有组件均为 $W$ 的函数。

**2. 进展潜力（Progress Potential）$\Phi$（§3.2）**
- 由 DAG 中每个节点 $v$（可检查的输出状态）构成进度契约，每个节点有权重 $w_v$ 和部分评分器 $c_v(s) \in [0,1]$。
- 信用前置条件门控：$\tilde{c}_v(s)$ 取 $v$ 及其所有前置节点中最小评分。
- 公式 (3)：$\Phi(s) = \sum_{v \in V} w_v \tilde{c}_v(s) \in [0,1]$。
- 三个安全保障（通过 qualification gates 实现）：**锚定性**（exploit resistance：14 种 gaming probe 得分均为 0）、**单调性**（progress validity：沿参考轨迹单调升至 1，回退时下降）、**独立性**（role independence：评分器代码与验证器、求解器不共享闭包）。

**3. PPRS 奖励塑造（§3.3）**
- 公式 (4)：$R(\tau) = \mathbb{1}[\tau \text{ passes}] + \lambda \sum_{t=1}^T r_t = \mathbb{1}[\tau \text{ passes}] + \lambda \Phi(s_T)$，其中 $\lambda=0.25$。
- 与经典 PBRS 的区别：保留终端潜在值，故意修改目标函数——通过让所有失败组在第一轮就有非零方差，修复 GRPO 梯度枯竭。
- 每轮归一化：优势按同 turn index 的组内均值归一化，保留后期步骤的信用；return-to-go 设置 $\gamma=1$，使进展可回传早期动作。
- 通过率保证：任何通过 rollout 得 $1+\lambda$，任何失败 rollout 至多得 $\lambda$，不存在用完成度换取部分进展的套利。

**4. 训练设置（§4.1）**
- 使用 DPPO（group-relative policy gradient），TV divergence bound=0.1，无 KL 锚（$\beta=0$）。
- 每步 8 个任务，每任务 24 rollouts，响应预算 65,536 tokens/rollout（每 turn 16,384），最多 64 次 tool call；学习率 $10^{-6}$，AdamW。
- TEV-9B 在 unit tier（264 任务）上训练；TEV-27B 在完整语料（1,170 任务）上训练；TEV-320B 以 GLM-5.3-Flash 为 base 应用相同配方。

## 实验与结果
**评估基准**：
- **ALE-CM**：Computing & Math 子集，18 任务，分三类（Math&Science 8、ML 5、Coding&System 5）。
- **Terminal-Bench 3.0/4.0 CM 子集**：分别为 22/18 任务（通过 Claude 筛选的 CM 相关任务）。
- **TB 2.1**：中程基准，89 任务，用于验证 zero-shot 泛化。

**主要结果（Pass@k，$k$ 次 rollout 中最优）**：

| 模型 | ALE-CM P@5 | TB 3.0 CM P@1 | TB 4.0 CM P@1 | TB 2.1 P@1 |
|---|---|---|---|---|
| GPT-6 Astra | 0.28 | 0.41 | 0.46 | — |
| Claude Opus 5 | 0.28 | 0.53 | 0.52 | — |
| **TEV-27B** | **0.28** | **0.14（+56% vs base）** | **0.10（+150% vs base）** | **0.794（≈Opus 4.8）** |
| GLM 5.2 | — | 0.04 | 0.06 | — |
| Gemini 3.7 Flash | — | 0.07 | 0.09 | — |
| **TEV-320B** | — | **0.27（+35% vs GLM-5.3）** | **0.37（+28% vs GLM-5.3）** | — |

- TEV-9B 较 Qwen3.5-9B 在 ALE-CM 上 Pass@5 提升 83%（0.06→0.11）。
- TEV-27B 在 ALE-CM 上以 0.28 Pass@5 追平 GPT-6 Astra 和 Claude Opus 5，达到前沿的 85%。
- TB 2.1 上 TEV-9B 零样本 Pass@1 提升 28%，TEV-27B 达 0.794，与 Claude Opus 4.8 持平。
- Qualitative 分析显示 TEV-27B 在 k8s 根因分析任务上能定位真实根因（green 决策）而非仅引用症状。

**消融（Table 5，9B 设置）**：
- 数据消融：替换潜在世界为 LLM 生成（Public-view verifier、Prompt-first）导致 88% 组被丢弃，S@1 从 0.29 降至 0.24/0.23。
- 奖励消融：仅用终端稀疏奖励（Sparse）无法完成一个优化器步骤（27B 尺度）；PPRS per-turn 归一化 + $\gamma=1$ 效果最佳。

## 相关工作脉络
- **SWE-Smith / TMAX / SETA / Endless Terminals / CLI-Gym**：同为 agentic 训练语料库，但规模为数千至数万短程任务，reward 仅为 pass/fail 或 test fraction，无渐进信号，且 difficulty 不可控。TEV-GYM 专注于长时距（median 12.4min），提供 staged $\Phi$。
- **Terminal-Universe / CLI-Gym**：从 agent trajectory 重建环境，无 latent world 锚点，易发生 artifact drift；TEV 从确定性的 $W$ 反向渲染，彻底消除漂移。
- **AutoForge / EnvScaler / ScaleEnv / EnvFactory / LOGIGEN**：同为程序化环境合成路线，但面向短 tool-use  episode；TEV 面向多阶段、跨领域的长时距工作流，且每个环境内嵌进展潜力 $\Phi_W$。
- **Process Reward Models (PRMs) / Learned step rewards**：依赖训练好的 critic 或 LM judge 提供逐步信号；TEV 的 $\Phi$ 完全由 latent world 决定，无需额外模型，且保证因果性和 exploit resistance。
- **Backward construction 家族**（Denoising/BART、bug injection、instruction back-translation）：均在单一 artifact 上操作；TEV 对完整 latent world 的多格式渲染施加损坏，使中间状态全程可验证。

## 局限性与未来方向
- **当前仅覆盖合成终端环境**：所有任务运行在 Docker sandbox 中，不涉及真实线上系统或物理世界交互（§6）。
- **前沿性能仍存在差距**：TB 4.0 上 TEV-27B P@1=0.10，而 Claude Opus 5 达 0.52，说明更强 base model 是充分利用高质量长程数据的瓶颈。
- **领域扩展需人工编写 generator**：支持新工作流需要重新设计和验证环境组件，当前仅覆盖 ALE-CM 的 7 个 CM 子领域（但 Appendix A.10 已演示可扩展至能源工程、金融等）。
- **$\lambda=0.25$ 固定未调参**：对 $\lambda$ 的敏感性未在系统中探索（Appendix B）。
- **评估子集的领域偏差**：TB CM 子集由 Claude 判定是否属于 CM 领域，可能存在主观偏差。

## 研究启发与可借鉴点
- **潜在世界优先的"单一锚点"设计**：将任务、输入、验证器、进度信号全部声明为同一 latent world 的函数，从根本上消除了 artifact drift，这一设计原则可迁移到任何需要可验证合成数据的场景（如科学发现、自动化实验）。
- **Progress Potential $\Phi$ 的结构化设计**：利用 DAG 前驱门控信用，使进展信号严格单调且防 gaming，这一思路可推广到任何具有明确阶段依赖的多步任务（代码生成、论文复现、系统部署）。
- **PPRS 修复 GRPO 梯度枯竭的策略**：在 full-fail 组中用 $\Phi(s_T)$ 代替终端二元 reward，使每组都有非零方差；这一模式可与任何 group-relative 优化器（GRPO、PPO-GROUP）结合使用，建议在其他长程 RL 场景中复现验证。
- **资格门（Qualification Gates）体系**：8 项 benchmark-independent 的自动生成时验证（exploit resistance、progress validity、role independence 等），为合成数据质量提供了可复用的质检范式，可直接借鉴到团队的数据合成管线中。
- **规模阶梯（scale tier）× DAG 深度 的双维难度控制**：将 operational horizon 和 dependency depth 作为独立超参暴露给 generator，便于 curriculum learning 式的课程训练设计。

## 关键术语表
- **Latent World ($W$)**：任务的完整、正确、隐藏的规范状态（包含所有正确中间态），是任务、输入文件、验证器和进展信号的共同锚点。
- **Progress Potential ($\Phi_W(s)$)**：基于 DAG 节点和前置条件门控的加权评分函数，衡量 agent 当前工作空间 $s$ 相对于正确潜在世界的完成程度，取值 $[0,1]$。
- **PPRS (Progress-Potential Reward Shaping)**：将 $\Phi$ 的增量作为稠密逐步奖励的 RL 训练配方，$R(\tau)=\mathbb{1}[\text{pass}]+\lambda\Phi(s_T)$，修复 GRPO 在长程任务上的梯度枯竭。
- **TEV-GYM**：首个含可验证稠密渐进奖励的长程智能体训练数据集，1,170 个环境跨 7 领域，median 12.4 分钟/rollout。
- **ALE-CM**：Agents' Last Exam 的计算与数学科学子集，18 个专家编写长程任务，涵盖数学/ML/编程与系统三类。
- **Terminal-Bench (TB)**：面向真实 CLI 终端任务的长程评测基准系列（3.0/4.0/2.1），每任务需约 8 小时专家工作，CM 子集为本文选取的 evaluating subset。
- **Qualification Gates**：生成时自动执行的 8 项基准无关质量门，包括 exploit resistance、progress validity、role independence 等，确保合成环境无漏洞且进度信号有效。
- **Artifact Drift**：任务、环境和验证器分别生成时出现的描述不一致问题；latent-world-first 通过单一 $W$ 锚点从根本上规避此问题。

## 可复现要素
- **数据集**：TEV-GYM 完全合成，每个环境为 generator+seed+scale tier 的确定性函数；生成中位耗时 30s CPU，无需模型调用；附录 A.10 给出 3 个完整例子的 end-to-end  walkthrough。论文未提供公开下载链接，但给出了精确的生成流程和公式，可复现。
- **代码/权重**：TEV-9B、TEV-27B、TEV-320B 的模型权重论文未声明开源；训练使用公开基座模型（Qwen3.5-9B、Qwen3.8-27B、GLM-5.3-Flash）和 DPPO 算法，超参见 Table 11。
- **关键超参**：$G=24$ rollouts/task，每步 8 任务（192 rollouts），响应预算 65,536 tokens/rollout（每 turn 16,384 tokens），最多 64 tool call，$\lambda=0.25$，TV divergence bound=0.1，$\gamma=1$，学习率 $10^{-6}$，temperature=1.0，零 KL 锚。
