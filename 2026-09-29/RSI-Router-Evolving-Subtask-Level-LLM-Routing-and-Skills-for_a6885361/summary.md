---
title: "RSI-Router-Evolving-Subtask-Level-LLM-Routing-and-Skills-for"
source: https://arxiv.org/pdf/2609.34712v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:19:07"
field: "多模型协同与高效推理"
keywords: ["LLM routing", "cost-efficient inference", "agentic tasks", "recursive self-improvement", "subtask-level routing", "model-specific skills"]
innovations: ["提出递归自我改进的子任务级路由框架，联合演化路由策略与模型专属执行技能", "设计帕累托前沿与负样本档案双队列机制，实现路由系统的持续经验累积", "在五个 Agent 基准上以平均 51.7% 成本降低实现与超越大模型性能"]
benchmarks: ["ALFWorld", "ScienceWorld", "WebShop", "SWE-bench Verified", "Terminal-Bench 2.0"]
---

# 论文速读：RSI-Router: Evolving Subtask-Level LLM Routing and Skills for Cost-Efficient Agents

## 一句话总结
RSI-router 提出了一种基于递归自我改进的子任务级别 LLM 路由框架，通过迭代挖掘可复用子任务、演化路由策略与模型专属执行技能，在五个 Agent 基准上以约一半推理成本（平均降低 51.7%）实现与或超越大模型基线的性能表现。

## 研究问题与动机
- **长程 Agent 任务的性能-成本权衡难题**：复杂 Agent 任务（如软件开发、科研辅助）需要多次 LLM 调用，依赖单一大型模型成本高昂，而小模型单独执行时性能不足。
- **任务级路由的局限**：已有任务级路由方法将整个请求分配给单一模型，无法利用小模型完成任务中某些子阶段的能力，错失任务内协作的降本机会。
- **步骤级路由的学习挑战**：现有步骤级方法依赖预收集轨迹，覆盖范围受限；基于强化学习的方法缺乏任务结构指导，探索成本高且信用分配困难。
- **路由决策与执行技能的相互依赖性**：模型分配与执行技能需联合优化——小模型在获得合适执行技能后可能从"无法完成任务"变为"可靠完成子任务"。

## 核心贡献（创新点）
1. **递归自我改进的路由范式**：通过执行结果与累积经验四阶段迭代进化路由系统与执行技能，与静态路由或单轮优化方法形成本质区别。
2. **子任务级路由与模型专属执行技能联合演化**：将长程任务分解为可复用子任务并针对每个"子任务-模型"对演化执行技能，而非仅做任务级或步级路由选择。
3. **帕累托最优路由器选择与负样本归档机制**：同时保留非支配路由器与支配路由器（作为后续演化的反面教材），形成持续改进的经验循环，区别于仅保留 Pareto 前沿的传统多目标优化。
4. **跨五项 Agent 基准的性能-成本前沿扩展**：在 ALFWorld、ScienceWorld、WebShop、SWE-bench Verified 和 Terminal-Bench 2.0 上均建立更优的 Pareto 前沿，显著优于 9 种现有路由基线。

## 方法详解
RSI-router 的核心是一个四阶段递归进化循环，由大型 Proposal Agent（使用 Codex）驱动：

1. **子任务挖掘（Subtask Mining）**：
   - 分析训练轨迹，按执行目标切分 segments，聚类并定义子任务集合 $Z_t$。
   - 为每个子任务生成定义与识别规则；从第 2 次迭代起对已有子任务进行合并/拆分/修订。
   - 用小模型回放轨迹验证子任务识别准确率， refine 歧义定义后固定 $Z_t$ 用于本轮迭代。

2. **路由策略演化（Routing Strategy Evolution）**：
   - Proposal Agent 并行生成 $N=4$ 种多样化路由策略。
   - 基于当前子任务定义从共享技能库有选择地继承历史技能 $\mathcal{K}_{t-1}^i$。
   - 在每个候选路由器 $r_i$ 上评估训练集与验证集的 aggregate 性能与成本，生成训练轨迹。

3. **模型专属技能演化（Model-Specific Skill Evolution）**：
   - 对比候选路由器 $r_i$ 与大模型-only 路由器 $r_L$ 在相同任务上的轨迹，诊断模型专属失败模式与冗余动作。
   - 针对性地精炼或生成执行技能，更新 $\mathcal{K}_{t-1}^i \rightarrow \mathcal{K}_t^i$，得到 $\tilde{r}_i$ 并重新评估。

4. **帕累托最优路由器选择（Pareto-Optimal Router Selection）**：
   - 维护非支配路由器集合 $\mathcal{P}_t$ 与被支配路由器档案 $\mathcal{F}_t$（作为负面样本）。
   - 以验证集性能与成本比较历史与新路由器，更新 $\mathcal{P}_{t+1}$ 与 $\mathcal{F}_{t+1}$。

**推理阶段**：小模型 Qwen3.5-9B 作为 router，基于上下文 $\mathcal{C}$（含用户查询、前 3 步交互历史、上一子任务预测）预测当前步骤子任务，选择对应模型并附加专属技能文本后调用。

**初始设定**：$\mathcal{P}_0 = \{r_L, r_S\}$（分别全程使用大/小模型，无技能），$\mathcal{F}_0 = \emptyset$；迭代次数 $G=6$，每轮候选数 $N=4$。

## 实验与结果
- **数据集/基准**：ALFWorld（家务交互）、ScienceWorld（科学实验）、WebShop（在线购物）、SWE-bench Verified（代码修复）、Terminal-Bench 2.0（终端操作），每基准训练/验证/测试集规模分别为 64/64/64（前三者）、32/32/32（SWE-bench）、32/32/25（Terminal-Bench 2.0）。
- **模型配置**：大模型 DeepSeek-V4.1-Flash，小模型 Qwen3.5-9B；成本估算比率 1:30（基于 H200 GPU-hour 实测）。
- **主要结果（vs DeepSeek-V4.1-Flash 单模型）**：
  - ALFWorld：成功率 98.44%（+8.6pp），成本降低 75.3%（$1.50 → $0.37）
  - ScienceWorld：成功率 35.94%（+4.5pp），成本降低 82.2%（$1.91 → $0.34）
  - WebShop：reward 0.61（+3.4%），成本降低 74.7%（$0.75 → $0.19）
  - SWE-bench Verified：成功率 84.38%（+1.3pp），成本降低 8.1%（$3.46 → $3.18）
  - Terminal-Bench 2.0：成功率 46.67%（+16.7%相对提升），成本降低 18.0%（$17.10 → $14.02）
  - **平均成本降低 51.7%**
- **对比基线**：优于全部 9 种任务级（HybridLLM、FrugalGPT、RouteLLM、GraphRouter、Avengers-Pro）与步骤级（Router-R1、MTRouter）路由方法，占据更优性能-成本 Pareto 前沿。
- **迭代分析**：所有基准在第一轮后持续降成本；技能跨迭代复用率高达 45.8%（ScienceWorld 最终轮 48 个技能源中 22 个来自早期迭代）。
- **消融**：移除执行技能后 ALFWorld/ScienceWorld/WebShop 性能分别下降 12.4/11.5/7.4pp；交互历史长度>3 步无显著提升；直接难度判断路由劣于随机步级路由。

## 相关工作脉络
- **任务级路由（HybridLLM、FrugalGPT、RouteLLM、GraphRouter、Avengers-Pro）**：以整个请求为单位分配单一模型，无法捕捉任务内子阶段的异构能力需求；RSI-router 通过子任务级路由打破此限制。
- **步骤级路由（Router-R1、MTRouter）**：逐 step 选择模型，但缺乏任务结构先验、探索成本高；RSI-router 以子任务为决策单元降低搜索空间并利用任务先验。
- **SkillOrchestra**：从预收集轨迹提取技能辅助路由，但技能与路由策略是分离优化的；RSI-router 联合演化路由与模型专属技能，形成闭环。
- **Self-evolving Agents（GEPA、ACE、Darwin Gödel Machine、Self-Harness、HarnessX 等）**：聚焦 prompt/代码/harness 自我改进；RSI-router 将反馈驱动迭代范式迁移至多模型路由场景。
- **Agent Skills（Trace2Skill、SkillRL、SKT 等）**：关注技能蒸馏与内部化；RSI-router 的独特性在于将技能与路由策略绑定到"子任务-模型"对，并随路由演化持续累积。

## 局限性与未来方向
- **双模型设定限制**：当前仅探索大型/小型两模型协作，未扩展到多模型（如 3+ 模型族）路由场景。
- ** Proposal Agent 依赖强模型**：演化过程由 Codex（大模型）驱动，本身消耗推理成本，规模化应用时可能成为瓶颈。
- **技能泛化性待验证**：当前技能针对特定子任务-模型对演化，跨领域/跨任务迁移能力未充分评估。
- **验证集规模有限**：每基准仅 64（或 32）个验证任务，可能不足以全面捕捉路由策略泛化边界。
- **未探索动态子任务重划分**：子任务集合 $Z_t$ 在单轮迭代内固定，实际长程任务中动态重组织子任务结构可能是潜在改进点。

## 研究启发与可借鉴点
1. **子任务作为路由决策粒度**：将长程 Agent 轨迹按执行目标切分为可复用子任务，比步级/任务级路由更契合 Agent 任务的结构化特征，可迁移至其他多阶段决策场景。
2. **帕累托前沿 + 负样本档案的双队列设计**：既保留最优策略又归档被支配策略作为反面教材，形成经验闭环，适用于多目标优化型 Agent 系统设计。
3. **模型专属执行技能与路由联合演化**：技能不应仅在路由确定后静态注入，而应与路由策略共同迭代——小模型获得适配技能后能力边界可扩展，这一反馈环值得推广。
4. **用大模型 Proposal Agent 自动化演化小模型路由系统**：以强模型驱动演化、弱模型部署推理的分层架构，可在保证演化质量的同时控制在线成本，是 Agent 系统部署的有效范式。
5. **消融发现"技能对简单任务有害"**：SWE-bench 和 Terminal-Bench 移除技能后性能反而提升，提示技能引入需考虑任务复杂度阈值，避免过度引导导致灵活性下降。

## 关键术语表
- **RSI-router**：Recursive Self-Improvement router，通过四阶段递归进化循环迭代优化子任务定义、路由策略与执行技能的 LLM 路由框架。
- **Subtask Mining**：从 Agent 执行轨迹中按执行目标切分并聚类子任务，生成可识别的子任务定义与识别规则的过程。
- **Model-Specific Skill**：针对特定"子任务-模型"对定制的执行指导文本，用于增强小模型在分配子任务上的可靠性并减少冗余动作。
- **Pareto-Optimal Router**：在性能-成本多维空间中不被其他路由器支配的路由策略；RSI-router 同时保留支配与被支配路由器以支持持续演化。
- **Proposal Agent**：驱动整个路由演化流程的大型 LLM（本文使用 Codex），负责生成子任务定义、路由策略与执行技能的智能提议模块。
- **Routing Context $\mathcal{C}$**：用于小模型预测当前子任务的输入上下文，包含用户查询、前若干步交互历史与上一子任务预测结果。
- **Performance-Cost Pareto Frontier**：在所有评估方法中，由性能-成本权衡最优策略构成的前沿曲线；RSI-router 在此前沿上占据更大比例。

## 可复现要素
- **数据集**：ALFWorld、ScienceWorld、WebShop、SWE-bench Verified、Terminal-Bench 2.0 均为公开基准；训练/验证/测试集划分由论文固定提供（Appendix B 含完整 prompt）。
- **代码/权重开源情况**：论文未明确声明代码仓库链接，模型（DeepSeek-V4.1-Flash、Qwen3.5-9B）为公开可用模型。
- **关键超参**：迭代次数 $G=6$，每轮候选策略数 $N=4$，交互历史长度 3 步，成本估算比率 1:30（DeepSeek:Qwen）。
- **评估设置**：三个随机种子平均，测试集无反馈独立评估；成本按官方 DeepSeek 定价与 H200 GPU-hour 实测比率估算。
