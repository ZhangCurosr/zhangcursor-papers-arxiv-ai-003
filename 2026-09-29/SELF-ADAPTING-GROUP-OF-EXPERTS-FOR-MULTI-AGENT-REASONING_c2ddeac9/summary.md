---
title: "SELF-ADAPTING-GROUP-OF-EXPERTS-FOR-MULTI-AGENT-REASONING"
source: https://arxiv.org/pdf/2609.35412v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:09:32"
field: "多智能体推理与协作"
keywords: ["multi-agent reasoning", "prompt adaptation", "strategy transfer", "self-consistency", "sparse DAG", "training-free", "peer review"]
innovations: ["基于响应质量与互惠同行评审的无需训练策略捐赠者选择机制", "仅查看原始 role prompt 的角色保留提示改写以实现通用推理策略迁移", "得分导向稀疏 DAG 的动态路由与跨阶段加权池投票聚合"]
benchmarks: ["MATH", "GSM8K", "GSM-Hard", "AQuA-RAT", "MMLU", "GPQA-Diamond", "MMMU-Pro", "MathVista"]
---

# 论文速读：SELF-ADAPTING-GROUP-OF-EXPERTS-FOR-MULTI-AGENT-REASONING

## 一句话总结
SAGE 是一个无需训练的多智能体推理框架，通过答案一致性、前缀一致性和互惠同行评审选出"策略捐赠者"，将其原始系统提示中的推理策略迁移给其他智能体（同时保留各自角色），再沿动态稀疏有向无环图进行多轮协作，最终以跨阶段加权投票选出最优答案。

## 研究问题与动机
- 现有 MAS 框架大多仅通过改变上下文来调整通信结构，而固定智能体的角色提示，即使当前问题需要不同的推理策略。
- 智能体响应由三大因素共同决定：底层模型能力、系统提示定义的角色/推理策略、输入上下文。仅改变通信无法修订角色的推理指导。
- 更多讨论并不必然提升答案质量（Smit et al., 2024），需要使团队协作与每个问题的推理需求动态匹配。
- 小模型在特定任务上表现优异但面对更广需求时吃力（Magister et al., 2023），可利用当前问题的响应信号在角色间迁移可复用的推理策略，而非泄露具体问题答案。

## 核心贡献（创新点）
1. **无需训练的三步自适应框架**：先按响应质量选出策略捐赠者，再做一次性的提示改写，最后在动态稀疏 DAG 上多轮协作；与需训练/外部裁判的方法（如 G-Designer）本质不同。
2. **基于响应的策略捐赠者选择**：综合答案一致性（self-consistency 思想）、前缀一致性（Iwase et al., 2026）和互惠同行评审来选 donor；与 SelfOrg 直接用嵌入相似度选边不同。
3. **角色保留的提示改写（Strategy Rewriting）**：用 donor 的原始 system prompt 改写其他 agent 的 prompt，但重写器只能看到两条原始 prompt、看不到问题或任何响应，实现通用推理习惯的可迁移迁移而非答案泄露。
4. **得分导向的稀疏 DAG 路由**：每轮按严格降序构建稀疏入度有界 DAG，理论保证每轮无环且每条有向路径长度 ≤ K；与 MOC 的固定随机 DAG 或 G-Designer 的学习图形成对比。
5. **跨阶段加权池投票**：将初始答案、保留评审和各协作轮答案一起纳入加权投票，领导者 votes 权重更高，避免最终轮正确答案被覆写后丢失。

## 方法详解
- **整体流程**：给定问题 $x$，$N$ 个 agent（模型 $\mathcal{M}_i$、系统提示 $s_i$）独立生成初始响应 $y_i^0=\mathcal{M}_i(x;s_i)$。
- **阶段一：策略捐赠者选择**
  - **答案一致性**：$q(y,B)=\frac{1}{|B|}\sum_{b\in B}\mathbb{I}[\kappa(b)=\kappa(y)\neq\emptyset]$，度量与其他响应的答案重合度。
  - **前缀一致性**：截取响应前 $\tau$  fraction（本文 $\tau=0.60$），令同 agent 在原提示下继续补全，检查是否得相同答案：$z_\tau(y)=\mathbb{I}[\kappa(\bar{y})=\kappa(y)\neq\emptyset]$。
  - **综合得分**：$\rho(y,B)=q(y,B)+\lambda z_\tau(y)$，$\lambda=0.50$。
  - **互惠同行评审**：按 $\rho^0$ 将 agent 分为高分组 $H_m$ 和低分组 $L_m$（各采样 $m=2$），组间两两互评，各自决定是否 KEEP 或 EDIT 自己的答案，得到候选响应集 $\mathcal{R}$；保留每组内得分最高的候选 $y_i^\dagger$，再在其中取 $\arg\max\rho(y_i^\dagger,\mathcal{R}^\dagger)$ 为捐赠者 $\ell$。
- **阶段二：角色保留的策略改写**
  - 用目标 agent 自身的 backbone（$T=0.20$）执行重写：$\tilde{s}_i=R_{\mathcal{M}_i}(s_i,s_\ell)$，仅输入两条原始 prompt，输出只增补通用推理/检查习惯，不引用问题或答案。捐赠者 $\tilde{s}_\ell=s_\ell$ 不变。
- **阶段三：稀疏 DAG 上的多轮协作**
  - 每轮 $t$ 开始前，每个 agent $i$ 选取至多 $K$ 个严格更高分的父节点：$P_i^t=\text{TopK}_K\{j\neq i:\rho_j^{t-1}>\rho_i^{t-1}\}$。理论证明 $|E^t|\le KN-K(K+1)/2$，每条有向路径 ≤ K 条边。
  - 沿拓扑序（降分序）依次修订：领先者只自查，其余 agent 将父节点当轮已更新答案作为额外上下文，按 $\tilde{s}_i$ 修订：$y_i^t=\mathcal{M}_i(x;\tilde{s}_i,(y_i^{t-1},\mathbf{y}_i^t))$。
  - 每轮重算 $\rho_i^t$，直到 $T$ 轮结束或全部答案收敛。
- **阶段四：加权池投票**
  - 池 $\mathcal{P}$ 包含所有阶段的答案样本 $(t,i,y)$；对每个答案键 $a$，$W(a)=\sum(1+\beta\mathbb{I}[i=i_t^\star])\mathbb{I}[\kappa(y)=a]$，$\beta=0.5$；返回权重最大的 $\hat{a}$，同分按领导者票数、后出现的阶段、低 index 依次破 tie。

## 实验与结果
- **数据集**：MATH、GSM8K、AQuA-RAT、GSM-Hard、MMLU、GPQA-Diamond；视觉语言额外在 MMMU-Pro、MathVista 评测。
- **主干模型**：Qwen2.5-1.5B-Instruct、Ministral-3-3B-Instruct-2512；扩展实验用 Qwen2.5-Instruct 0.5B–72B、Qwen2.5-VL-3B-Instruct。
- **基线**：Single、CoT、SelfOrg、MOC、MAD-M²、G-Designer。
- **主要结果（Table 1）**：
  - Qwen2.5-1.5B：SAGE **AVG=58.31%**，对比 MAD-M²（53.40%） **+4.9 点**，在所有 6 个基准上均超越全部多智能体基线；较 SelfOrg（53.12%）**+5.2 点**。
  - Ministral-3-3B：SAGE **AVG=75.08%**，对比 MAD-M²（73.79%）**+1.3 点**；较 SelfOrg（73.07%）**+2.0 点**。
- **智能体缩放（Fig. 2）**：4→9 个 agent，Qwen2.5-1.5B AVG +1.5 点，Ministral-3-3B AVG +1.8 点。
- **模型缩放（Fig. 3）**：SAGE 在 0.5B–72B 所有规模上 GSM-Hard 均优于 Single，提升 1.2–5.0 点；GPQA-Diamond 提升较弱，3B/32B 甚至略低于 Single。
- **异构团队（Fig. 4）**：9 agent（Qwen/Ministral/Phi 各 3）在 GSM8K=93.6%、MMLU=71.4% 上接近纯 Ministral（94.2/76.0），超过三种同质团队均值约 +5 点；Ministral 贡献 72%/65% 的捐赠者。
- **抗对抗（Fig. 5）**：9 agent 中 3 个被注入错误答案，SAGE 在全部 5 个基准上优于 SelfOrg，AQuA-RAT +5.8 点、MATH +4.3 点。
- **消融**：去掉 prompt 改写（SAGE-NOREWRITE）AVG 下降 2.10–2.18 点；仅保留最终轮投票（Final-round WPV）AVG 下降 0.36–0.79 点；随机选 donor（SAGE-RANDOMDONOR）AVG 下降约 1.6 点。

## 相关工作脉络
- **SelfOrg（Tastan et al., 2026）**：基于响应嵌入相似度重建每轮图；本文相比之下用响应得分 + 提示改写，路由更稀疏且有理论深度界。
- **MOC（Guan et al., 2026）**：固定随机 DAG + 两跳上下文记忆；本文每次按当前响应质量重构图。
- **MAD-M²（Tian et al., 2026）**：用 objective token-confidence 做 memory masking 过滤历史；本文不用记忆掩码而靠结构化提示迁移和得分路由。
- **G-Designer（Zhang et al., 2025b）**：用 GNN 学习每问题 topology；本文无需训练、零参数更新。
- **PRomPTed / SPRIG（Srivastava et al., 2024; Zhang et al., 2026a）**：对 instance 或 reusable prompt 做优化；本文的改写器仅看两条原始 role prompt，不与问题/响应交互，避免泄露。
- **MAPRO / HiveMind（Zhang et al., 2026b; Xia et al., 2026）**：训练/优化多 agent 的 prompt 与拓扑；本文是完全训练无关（training-free）的一次性 per-problem 策略转移。

## 局限性与未来方向
- **GPQA-Diamond 上捐赠者信号弱**：最难基准上几乎所有 agent 答错，一致性和评审给出随机化选择，导致策略迁移增益有限。
- **改写质量因题而异**：重写在部分问题上对特定错误类型针对性不强，改写器无法看到响应本身只能抽取通用习惯。
- **鲁棒性仅检验了固定非自适应错误注入**，未覆盖更具适应性的对抗策略。
- **未涉及代码生成、工具调用等场景**，仅覆盖文本推理与视觉语言推理；异构团队仅含 3 种 backbone。
- **角色池固定为 9 类**，未见对更大角色库或跨领域组合的扩展实验。

## 研究启发与可借鉴点
- **无需训练的 per-problem 自适应**：用响应信号（而非梯度/外部训练）驱动策略转移和通信重构，可作为弱模型/资源受限场景下的通用增强范式。
- **重写的"信息隔离"设计**：让重写器仅看原始 prompt 不看问题/响应，是兼顾策略迁移与防答案泄露的工程技巧，可直接复用至其他提示优化流程。
- **前缀一致性作为轻量自我校准信号**：只要求模型对自身输出的前半段续写一次，即可衡量答案稳定性，成本低且可与 self-consistency 互补。
- **跨阶段加权池投票**：保留早期正确但可能在迭代中被覆盖的答案，作为防"过度对齐"的安全网；这一思想可泛化到 debate / multi-round 架构的答案聚合。
- **得分导向稀疏 DAG 的理论保证**：每轮严格降序路由 + 入度上限 K 既控制通信复杂度，又提供无环与路径长度上界，适合部署在需要低延迟的多 agent 服务中。

## 关键术语表
- **SAGE（Self-Adapting Group of Experts）**：本文提出的无需训练的多智能体推理框架，按响应自适应地选择捐赠者、改写提示并沿稀疏图协作。
- **策略捐赠者（Strategy Donor）**：初始响应经一致性/评审评分最高的 agent，其原始 system prompt 被用作策略迁移源。
- **答案一致性（Answer Agreement）**：以 self-consistency 为启发，统计多个响应最终答案相同的比例，作为群体支撑信号。
- **前缀一致性（Prefix Consistency）**：用原响应前 τ  fraction 续写，检查是否复现相同答案，衡量单 agent 内答案稳定性。
- **互惠同行评审（Reciprocal Peer Review）**：高分与低分 agent 两两互评，各自决定是否 KEEP/EDIT 自己的回答。
- **稀疏有向无环图（Sparse DAG）**：每轮按严格得分降序构建的通信拓扑，入度上限 K，保证无环与路径深度有界。
- **加权池投票（Weighted Pool Voting）**：把所有阶段（初始、评审、每轮）出现的答案合并，按阶段 leader 权重汇总选 final answer。
- **Prompt Rewriting**：以 donor 的原始 role prompt 为参考、在隔离条件下生成其他 agent 新 role prompt 的过程。

## 可复现要素
- **数据集**：MATH、GSM8K、AQuA-RAT、GSM-Hard、MMLU、GPQA-Diamond、MMMU-Pro、MathVista；公开基准。
- **代码/权重**：代码开源（论文 Project Page: https://www.atifquamar.com/sage-page）；主干使用开源 Qwen2.5-1.5B-Instruct、Ministral-3-3B-Instruct-2512、Qwen2.5-VL-3B-Instruct。
- **关键超参**：$N=4$（主实验），$K=2$，$T=3$，$m=2$；$\tau=0.60$，$\lambda=0.50$，$\beta=0.5$；温度 initial/revision/review=0.50、prefix=0.60、rewrite=0.20；top-p=0.80，top-k=20，repetition penalty=1.10；最大生成 token=2048，上下文窗口=32768。
- **评估工具**：xFinder-qwen1505 提取答案，xVerify-0.5B-I 判对错；实验重复 3 次报告 mean ± SD。
