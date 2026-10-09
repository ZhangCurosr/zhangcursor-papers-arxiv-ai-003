---
title: "SynCo-Data-Synthesis-Co-Training-for-Self-Evolving-LLMs-via"
source: https://arxiv.org/pdf/2610.11345v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:13:00"
field: "自进化大语言模型与多智能体强化学习"
keywords: ["self-evolving LLM", "multi-agent reinforcement learning", "data synthesis", "GRPO", "synthetic data", "mathematical reasoning", "agent-data co-training", "curriculum evolution"]
innovations: ["将数据合成与任务求解建模为耦合的多智能体 RL 问题，在同一在线批次中联合优化 Synthesizer 和 Reasoner", "提出 outcome-grounded gated reward，以 Reasoner 实际 rollout 结果（teachability）动态评估合成任务的学习价值", "建立 agent-data 闭环自进化框架，Synthesizer 随 Reasoner 能力演化自适应调整训练分布"]
benchmarks: ["GSM8K", "SVAMP", "ASDiv", "GSM-Hard", "MATH-500", "AIME24", "AIME25", "Minerva"]
---

# 论文速读：SynCo - Data Synthesis Co-Training for Self-Evolving LLMs via Multi-Agent Reinforcement Learning

## 一句话总结
SYNCO 提出一种基于多智能体强化学习的代理数据合成协同训练框架，将数据合成策略（Synthesizer）与任务求解策略（Reasoner）在同一在线交互过程中联合优化，使训练数据分布能随学习者的能力变化而持续自适应演化，在八项数学推理基准上取得了最强的平均准确率（78.72%）。

## 研究问题与动机
- **静态数据的错配问题**：现有 Agent 训练管线依赖静态或外部分批构造的数据，无法随着训练过程中 Agent 能力的提升而更新——原本有信息量的任务变得过于简单，而真正有挑战的任务仍无法被触及，导致学习者与训练体验之间的分布失配。
- **合成与学习的割裂**：现有自适应数据合成方法（如基于失败重生成、难度估计、弱项驱动）往往在上一轮学习者 checkpoint 上生成任务，再单独优化学习者，合成策略与任务策略并非在同一在线交互中联合优化，数据生成仍是从属于 Agent 训练的辅助过程。
- **非平稳耦合学习难题**：Synthesizer 通过生成任务塑造 Reasoner 的优化景观，而 Reasoner 的每次更新又重新定义了哪些任务仍有学习价值；任务的价值不能仅凭静态属性（有效性、新颖性、名义难度）判断，而必须 grounded 于 Reasoner 当前的实际交互行为。
- **自我进化 LLM 的可持续性瓶颈**：真正自我进化的 Agent 不仅需要一个更聪明的求解器，更需要一个能随求解器能力演进而持续自适应的"训练数据来源"，现有方法尚未提供闭环的 agent-data 联合演化机制。

## 核心贡献（创新点）
- **统一的多智能体 RL 协同训练框架**：将代理数据合成与任务求解建模为耦合的多智能体 RL 问题，在同一在线批次中对 Synthesizer 和 Reasoner 进行组相对策略优化（GRPO），两者参数独立但目标耦合。
- **能力条件化的闭环自进化机制**：Synthesizer 观测由近期交互内存构建的上下文 $c_t$（含 Reasoner 的实证能力画像与已合成任务覆盖），动态生成结构化可验证任务；Reasoner 的 rollout 结果作为 outcome-grounded reward 直接反馈监督 Synthesizer，形成 agent-data 双向闭环。
- **Outcome-grounded gated reward 设计**：Synthesizer 的奖励由三部分组成——task quality gate、answer reliability gate、以及 teachability 函数 $T(\mu_t)$（在 Reasoner 混合成功率区间最大化），将合成约束在可信任务空间的同时，以当前学习者的实际交互行为判定任务的学习价值。
- **实证验证与机制分析**：在八项数学推理基准上取得最强平均准确率（78.72%），超出最强现有合成数据方法 3.76pp；归因分析表明增益主要来自原本未解决的难题（Learn 子集贡献 80.4% 的额外正确案例），而 Frozen Synthesizer 变体在相同 Reasoner 下表现明显落后；课程演化与梯度利用的深入分析揭示了 teacehability 与梯度携带率的高度对齐（$r=0.934$）。

## 方法详解
**问题形式化**：设 Synthesizer 策略 $\pi_S(\cdot;\theta_S)$ 和 Reasoner 策略 $\pi_R(\cdot;\theta_R)$，两者独立参数化。在迭代 $t$，由上下文 $c_t$ 生成结构化任务 $x_t=(q_t,\mathbf{m}_t,s_t^\star,a_t^\star)$，Reasoner 对每个 $x_t$ 采样 $K=4$ 条独立 rollout $y_t^{(k)}\sim\pi_R(\cdot|q_t;\theta_R^t)$，经 checker 得到 $z_t^{(k)}=\text{Check}(\hat{a}_t^{(k)},a_t^\star)\in\{0,1\}$，并计算经验成功率 $\mu_t=\frac{1}{K}\sum_k z_t^{(k)}$。两者目标分别为：

$$J_R(\theta_R)=\mathbb{E}\!\left[\frac{1}{K}\sum_k r_{R,t}^{(k)}\right],\qquad J_S(\theta_S)=\mathbb{E}[r_{S,t}]$$

其中 $r_{R,t}^{(k)}$ 为 rollout 级正确性奖励（1.0/0.9/0），$r_{S,t}$ 为任务级 gated 奖励：

$$r_{S,t}=G_q(r_t^{\text{quality}})\cdot G_r(r_t^{\text{reliability}})\cdot T(\mu_t)$$

**Teachability 函数**：$T(\mu_t)=2(1-\mu_t)$（$\mu_t\leq0.5$）或 $2\mu_t$（$\mu_t>0.5$），在 Reasoner 恰好一半 rollout 成功时取最大值，uniformly correct/incorrect 趋近于 0。Gate 函数 $G(v)=\text{clip}((v-0.5)/0.5,0,1)$。

**能力条件化上下文**：$c_t=G(\mathcal{M}_t)=(p_t,d_t,e_t)$，分别刻画合成池覆盖、Reasoner 实证能力与薄弱点、以及合成失败与冗余记录。Synthesizer 基于此生成结构化任务卡，经 Verifier $V(x_t)$ 校验 schema 完整、问题自洽、答案一致后进入 Reasoner 训练。

**联合更新**：每步收集批次 $\mathcal{B}_t$，对两个策略分别做组相对 GRPO 更新 $\theta_R^{t+1}=\theta_R^t-\eta_R\nabla_{\theta_R}\mathcal{L}_R^t$，$\theta_S^{t+1}=\theta_S^t-\eta_S\nabla_{\theta_S}\mathcal{L}_S^t$；更新后的 Reasoner 改变能力状态，更新后的 Synthesizer 改变数据分布，形成闭环演化。

## 实验与结果
- **基线**：静态合成数据方法（AgenticQwen、EnvScaler、AgentSkiller、Klear-AgentForge、MetaMathQA、ScaleQuest-Math、OpenMathInstruct-2、R-Zero）+ 控制基线（Static-Pool GRPO、Offline-Augment GRPO、Reject-Sample SFT、SYNCO Frozen Synth）。
- **主要结果（Qwen3-8B）**：SYNCO 平均准确率 **78.72%**，超越最强现有合成数据方法 ScaleQuest-Math（74.96%，+3.76pp）与 Frozen Synth 变体（75.85%，+2.87pp）。在八个基准中六个达到最佳或并列最佳（GSM8K 95.75%、SVAMP 93.00%、ASDiv 97.95%、GSM-Hard 63.31%、MATH-500 87.80%、AIME24 83.33%、AIME25 73.33%、Minerva 35.29%）。
- **跨尺度验证**：Qwen3-4B 上 SYNCO 平均 75.20%，Frozen 为 73.72%，差距一致再现。
- **增益归因（表2）**：在 Learn（初始错误）子集上 SYNCO 较 Frozen 多解 45 例（+5.12pp），在 Retain（初始正确）子集仅多解 11 例（+0.43pp），80.4% 的增益来自新能力获取而非已有知识保留。
- **课程演化（图2）**：梯度携带任务比例从 31.3% 降至 9.5%，全对比例从 34.4% 升至 59.9%；teachability 从 0.206 下降至 0.060，与梯度携带率高度对齐（$r=0.934$）。
- **效率（表3）**：SYNCO 每步更新耗时 45.1s vs Frozen 45.8s（-1.5%），有效梯度利用率 14.5% vs 13.7%（+0.8pp），边际开销可忽略。
- **信号对比（图3）**：按 gated reward 选 top quartile 使梯度携带率从 16.2% 升至 62.3%（3.9×富集），teachability 单独可达 64.6%（4.0×）；质量/新颖性单独排名分别仅 12.8% 和 12.0%，均低于全池基准。
- **ex-ante 难度预测（图4）**：Synthesizer 自行标注的"learnable"与梯度携带率几乎不相关（$r=0.009$，精度 16.8%，召回 22.3%），名义难度无法替代真实交互反馈。

## 相关工作脉络
- **静态合成数据管线**（Self-Instruct、WizardLM、MetaMath、DeepSeekMath 等）：离线构建数据集后单独用于下游 SFT/RL；本文差异在于合成策略是联合在线优化的可学习组件，而非固定模板或单次生成的辅助过程。
- **弱项驱动合成方法**（SwS [Liang et al., 2026]、FAC Synthesis [Li et al., 2026a]）：识别学习薄弱点后重生成任务，但合成策略与求解策略仍分阶段交替优化；本文在同一 batch 内用同一交互记录同时更新两者。
- **自进化 agent 框架**（R-Zero、Socratic-Zero、Agent0、Dr. Zero、SCOPE、CoEvolve）：建立提议者/教师/求解者的交替进化循环；本文的区别是将合成与学习建模为严格耦合的 multi-agent RL 问题，并使用 outcome-grounded gated reward 而非简单的 difficulty frontier 调节。
- **Multi-agent post-co-training**（MAPoRL [Park et al., 2025]）：联合优化多个协作响应生成 agent；本文聚焦"一个 agent 为另一个 agent 构造未来训练经验"的非对称耦合关系，目标函数与优化机制均有本质不同。
- **Agent-World、SEAD、AgentEvolver、EvoFSM**：探索环境、课程、记忆、工作流的自主演化；本文更聚焦于数学推理场景下的"数据合成-任务求解"这一核心 agent-data 耦合对，强调梯度携带任务的量化与课程演化的可观测分析。

## 局限性与未来方向
- **课程维持困难**：随着 Reasoner 能力持续提升，能够产生梯度携带信号的任务比例从 31.3% 骤降至 9.5%（50% 变为全对），如何在学习者接近能力边界后继续维持有效课程是一个开放问题。
- **ex-ante 难度估计失效**：Synthesizer 的前瞻性难度标签与梯度携带率几乎不相关（$r=0.009$），说明仅靠任务生成意图无法预测真实学习价值，需依赖 outcome-grounded feedback 才能识别有效的中间地带。
- **领域局限**：实验集中在数学推理（8 个基准），对于工具使用、open-ended 文本生成、多模态等场景的泛化能力尚待验证。
- **理论分析不足**：多智能体耦合 RL 的收敛性、稳定性等理论保证未在本工作中给出。
- **扩展方向**：将 framework 推广至更通用的 agent 任务（工具调用、web 交互、search），探索多轮交互场景下的 teachability 扩展，以及引入 memory 机制以维持长期课程多样性。

## 研究启发与可借鉴点
- **Outcome-grounded gated reward 范式**：将 task quality/reliability 作为 gate，teachability 作为 utility 信号的分离设计，为"合成数据如何评估自身学习价值"提供了可复用的 reward 构造模板，可迁移至其他需要动态数据合成的 RL 场景。
- **Gradient-bearing 任务量化分析**：定义 SR∈(0,1) 为梯度携带条件并统计其占比、与各种合成信号的相关性，这一分析框架可用于诊断任何 self-play 或课程生成方法的效率瓶颈，具有较强的方法论价值。
- **Retain/Learn 子集归因**：按初始模型预测对错划分评估样本以区分"知识保留"和"能力获取"的贡献比例，这一实验设计简洁而有力，可作为评估任何数据适应方法增益来源的标准分析流程。
- **与团队方向的结合机会**：团队若在工具使用 agent 或多模态推理方向有探索需求，可将 SYNCO 的 agent-data 联合演化框架迁移至 tool-use 任务合成场景，用 outcome-grounded teachability 替代现有的 success/failure 二元信号，探索跨域自进化路径。

## 关键术语表
**Synthesizer**：独立参数化的数据合成策略 $\pi_S$，根据当前 Reasoner 能力上下文生成结构化、可验证的训练任务。
**Reasoner**：独立参数化的任务求解策略 $\pi_R$，接收 Synthesizer 生成的问题并执行 K 条独立 rollout 进行求解。
**Teachability $T(\mu_t)$**：基于 Reasoner 经验成功率 $\mu_t$ 计算的 outcome-grounded 奖励函数，在混合成功/失败（$\mu_t\approx0.5$）时最大，全对或全错时趋零。
**Gated Reward**：Synthesizer 的复合奖励 $r_{S,t}=G_q\cdot G_r\cdot T$，由 quality gate、reliability gate 和 teachability 三项相乘构成，约束搜索空间并引导合成方向。
**Group-Relative Policy Optimization (GRPO)**：组相对策略优化，同一任务的多条 rollout 之间计算组内相对优势，用于更新策略而不依赖绝对价值估计。
**Gradient-bearing Task**：Reasoner 在该任务上的 K 条 rollout 结果不完全相同（$0<\text{SR}<1$），从而能产生非零组相对优势的任务。
**Capability-Conditioned Context $c_t$**：由 Synthesizer 观测的动态上下文，编码近期合成池状态、Reasoner 实证能力画像及历史合成失败反馈。
**Frozen Synthesizer Baseline**：保留在线 learner-conditioned 任务生成但固定 Synthesizer 参数的对照实验，用于隔离协同训练本身的增益。

## 可复现要素
- **数据集**：八项数学推理基准（GSM8K、SVAMP、ASDiv、GSM-Hard、MATH-500、AIME24、AIME25、Minerva），均为标准公开 benchmark，使用标准 test split。
- **代码/权重**：代码开源于 GitHub（https://github.com/weiyang930/SynCo.git）；数据集发布在 Hugging Face（https://huggingface.co/datasets/weiyang930/SynCo）。
- **关键超参**：Backbone Qwen3-8B（另附 Qwen3-4B 验证），actor lr $1\times10^{-6}$，warmup 10 steps，weight decay 0.1，无 KL penalty；每步 32 个合成任务，K=4 rollouts/task；Synthesizer temperature=1.0、p=0.95；Reasoner temperature=0.9、p=0.95；训练框架 UnityMAS-O + GRPO，vLLM rollout 后端。
