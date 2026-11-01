---
title: "USE-AND-DISUSE-INTENT-STRUCTURED-EXPERIENCE-CONSOLIDATION-FO"
source: https://arxiv.org/pdf/2610.12124v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:30:48"
field: "LLM Agent 长期记忆与持续学习"
keywords: ["LLM Agent", "持续学习", "记忆管理", "意图结构化", "经验巩固", "上下文压缩"]
innovations: ["提出基于意图堆栈的嵌套工作上下文，通过 JANUS 动态管理意图开/关/深化", "设计目标条件性 Intent Consolidation（PRECIP）与递归前缀 Consolidation（PALIM），实现叙事粗化与知识保留的分离", "构建使用与不使用驱动的统一记忆-学习循环，无需参数更新即可从交互历史中持续进化"]
benchmarks: ["ALFWorld", "ScienceWorld", "StreamBench", "GoodAI LTM", "LoCoMo", "MemoryAgentBench", "FEVER"]
---

# 论文速读：USE-AND-DISUSE-INTENT-STRUCTURED-EXPERIENCE-CONSOLIDATION-FO

## 一句话总结
论文提出 **Hippocam**，一种受人类记忆"使用与不使用"（use-and-disuse）启发的 LLM Agent 架构，通过分层意图堆栈驱动的目标条件性 Consolidation（PRECIP）与递归前缀 Consolidation（PALIM），在不更新模型参数的情况下实现记忆、知识积累与持续学习的统一。

## 研究问题与动机
- **长期自主 Agent 的知识转化难题**：LLM Agent 从单次任务执行走向长期自主运行时，如何将连续交互经验转化为可复用知识是一个核心挑战，现有方法往往依赖独立的外部记忆库或参数更新机制。
- **现有记忆管理方法的局限**：RAG-based 记忆（如 mem0、A-Mem）需要独立的事实存储与检索；时间衰减/重要性评分规则（如 MemoryBank、Generative Agents）是外部指定的，缺乏与任务进程的有机联系；ACE/ReasoningBank 等学习导向方法将经验蒸馏为 playbook，但丢失了可追溯的细粒度信息。
- **使用与不使用的自然动态**：人类短期记忆聚焦当前目标、选择性维持环境状态，长期记忆则通过重复巩固逐渐形成；长期未使用的经验应逐渐抽象化而非硬性遗忘，已提取的知识可在后续使用中不断被强化和更新。
- **统一架构的缺失**：记忆管理、知识使用与技能学习通常由不同模块分别处理，缺乏一个由 Agent 自身工作历程驱动的连续统一机制。

## 核心贡献（创新点）
- **意图结构化工作上下文**：将 Agent 的交互历史组织为嵌套意图堆栈（intent stack），通过 JANUS 动态管理意图的开/关/深化，使当前上下文始终围绕最相关的目的展开。与现有方法（如 HiAgent 的子目标摘要）的本质区别在于，本文以显式意图目的而非推断的未来需求来指导保留粒度和合并范围。
- **目标条件性 Consolidation（PRECIP）**：当一个意图结束时，PRECIP 将其工作消息合并为自包含的一人称记忆，保留与当前意图链相关的事实、结论、支撑证据及有意义的失败，同时剪除对意图链无贡献的内容。与简单摘要或 RAG 检索的本质区别在于，它保留了具体支撑证据的原文引证和"印象"（材料结构/位置），支持后续渐进式召回。
- **递归前缀 Consolidation（PALIM）**：随着新意图不断打开，更早的历史区域被划分为前缀并执行递归折叠，叙事部分逐次变粗化（失去细节），但从中提取的知识（事实/教训）被分离并持续保留，与后续新增工作一起再次合并，实现知识的互补、更新与强化。与 ACE playbook 生成或简单截断的本质区别在于，它不依赖独立的重要性评分或时间衰减，而是通过"使用频率"自然驱动知识的抽象与消退。
- **消息树与渐进式召回（Progressive Recall）**：Consolidation 保留原始消息作为被替换消息的子节点，构成消息树；主 Agent 可通过 recall 工具沿树向下检索所需细节，按需恢复细粒度信息并在足够时停止，无需一次性加载全量历史。
- **无需参数更新的持续学习**：整个机制在推理层运行，通过"使用"（重新融入的工作触发进一步巩固）与"不使用"（递归前缀巩固使无用经验渐趋抽象）的自然循环，使 Agent 在不更新任何参数的前提下从自身经验中不断学习。

## 方法详解
**整体架构**（§2.1）：Hippocam 包含一个负责推理/工具调用/环境交互的 **主 Agent**，以及三个辅助 LLM 组件：**JANUS**（意图堆栈维护）、**PRECIP**（意图 Consolidation）、**PALIM**（前缀 Consolidation）。

**意图堆栈**（§2.2）：交互上下文 $\mathcal{C} = \langle m_1, \ldots, m_n \rangle$ 由意图动态组织，每个意图表示为 $i = \langle p, \chi, b \rangle$（目的 $p$、退出条件 $\chi$、起始消息索引 $b$）。嵌套意图构成堆栈 $\mathcal{T} = \langle i_1, \ldots, i_d \rangle$。JANUS 在每个主 Agent 推理前执行四种操作：**Continue**（保持不变）、**Deepen**（在现有链中打开更具体的子意图）、**Close**（关闭已完成的后缀）、**Shift**（关闭后打开新方向的意图）。

**Intent Consolidation（PRECIP，§2.3）**：当 JANUS 关闭意图时，PRECIP 将对应消息段合并为一个自包含的一人称记忆，替换原消息。保留内容包括：已确立的状态变化、结论及其具体证据来源、未处理的 enclosing intent 所需材料、材料结构和位置的"印象"（以便后续定向检查）、有信息的失败（informative failures）。剪除：发现事实和建立事实相关的过程、对意图链无贡献的内容。多条事实共享同一教训时合并为一条综合洞察。

**Prefix Consolidation（PALIM，§2.4）**：意图堆栈的开边界将上下文划分为区域 $\mathcal{C}_j$。保护一个**全局工作尾部**（至少 $K$ 轮交互 + 预算 $B$ tokens），不被合并。当上下文压力超过阈值时执行 SWEEP。PALIM 将每条已合并消息拆分为**叙事**（narrative，记录做了什么及结果）和**知识**（knowledge，从工作中提取的事实与教训，不含基于更广归纳产生的内容）。更早的合并消息与新工作一起递归合并，叙事逐次变粗，知识被合并/补充/更新。

**消息树与渐进召回**（§2.5）：Consolidation 将被替换的原始消息及其已有子树存为被替换消息的子节点，形成消息树。主 Agent 的 `recall(uuid)` 工具返回某节点的子消息层，按需逐级下降，每次召回后判断是否需要继续深入。

**使用与不使用的学习循环**（§2.6）：
- **Use**：主 Agent 将合并消息视为自身先前工作，将其中的事实和教训用于当前任务，将新的推论视为假设，通过成功/失败获得反馈。后续 Intent Consolidation 中与这些观察结果一起被巩固，Prefix Consolidation 中被进一步整合、强化或更新。
- **Disuse**：当前未使用的偶然细节和知识通过递归前缀 Consolidation 逐步丢失细节，减少对当前推理的干扰。无需独立的重要性分数或时间衰减规则。

**关键超参**（Table 5）：$K=2$（最少工作轮数）、$B_{\min}=4096$、$\alpha=0.20$、$q=0.95$（工作轮数分位数）、$E_{\min}=2$（稳定前缀中最少意图合并数）、$T_{\min}=16384$（稳定前缀 token 数）、$W=128000$（总上下文容量）、压力阈值 $0.80W$、最小压缩量 $8192$ tokens、最多重试 2 次。

## 实验与结果
**实验设置**：基线包括 mem0、A-Mem、Letta、ReasoningBank、ACE、HiAgent，全部使用 DeepSeek-V4.1-Flash（1M token 上下文窗口）作为任务主干。评估涵盖四项研究问题（RQ1–RQ4）。

**RQ1 — 从积累经验中学习**（Table 1）：
- **ALFWorld**：Hippocam **97.8%**（与 ACE 并列最佳），相比 Plain（78.4%）提升 **+19.4pp**。
- **ScienceWorld**：Hippocam **81.9%**，超越 HiAgent（56.7%）、mem0（58.6%）、Letta（71.2%）、ReasoningBank（66.9%），仅次于 ACE（91.9%）。
- **StreamBench（前150任务）**：Hippocam **81.7%**（DDXPlus 87.3% + ToolBench 76.0% 均值），超越 Letta（81.3%）、A-Mem（79.0%）。
- **扩展至300任务**：DDXPlus 进一步提升至 **93.0%**，ToolBench 提升至 **78.7%**，后期每任务成本降低约 **10%**（Table 12）。

**RQ2 — 跨域泛化**（Table 2）：
- **ALFWorld → ScienceWorld Frozen**：Hippocam **78.9%**，较 Plain 提升 +30.1pp，较最强基线 ACE 提升 +16.2pp。
- **跨域整合增益 $\Delta_{stream}$**：ScienceWorld **+19.2pp**（优于 ACE 的 +14.0pp）；FEVER **+6.0pp**。是唯一在两个域间均实现正向增益的方法（ACE 在 FEVER 上仅 +2.0pp）。

**RQ3 — 长期交互中的记忆可靠性**：
- **GoodAI LTM（32k 交错条件）**：Hippocam **88.7%**（交错）vs **85.1%**（隔离），反而有所提升；而 LTMAgent、Letta、HiAgent 均出现性能下降。
- **LoCoMo 对话推理**：Hippocam 总体 **77.6%**，超越 A-Mem（67.5%）+10.1pp；多跳（40.8%）、时间（86.6%）、开放域（77.1%）均领先；不可答问题准确率达 87.2%。
- **MemoryAgentBench 冲突解决**：Hippocam 平均 **71.7%**，远超 Letta（44.3%）；64k 输入下 **41.0%**（无其他方法超过 19%）。

**RQ4 — 统一机制的普适性**（Figure 1）：Hippocam 在八项评估指标中**七项第一或并列第一**，展现了跨记忆、知识使用与学习的均衡能力。

**消融实验**（Table 4）：
- 移除意图条件（w/o Intent conditioning）：GoodAI 交错得分从 88.7% 降至 **65.6%**（↓23.1pp），隔离从 85.1% 降至 77.8%（↓7.3pp）；填充内容从 14% 升至 **63%**。
- 移除前缀 Consolidation（w/o Prefix）：ToolBench 从 78.7% 降至 **65.7%**；ALFWorld 起始上下文从 19.7k 膨胀至 **50.7k tokens**，平均步骤从 9.62 增至 11.74。
- 禁用召回：MAB-64k 从 41.0% 降至 **27.0%**。

**成本分析**（§A.9）：Hippocam 后期任务（151–300）在更高准确率的同时，每任务成本下降约 **10%**；ToolBench 每正确回答成本为 **0.0159 USD**，显著低于 A-Mem 的 0.0260 USD 和 mem0 的 0.0681 USD。前缀 Consolidation 自身开销约占 ToolBench 成本的 **34%**。

## 相关工作脉络
- **Context Management**：LongLLMLingua/SUPO 按查询相关性压缩上下文；HiAgent 总结已完成子目标；AgentFold 训练模型折叠历史后缀。Hippocam 的不同在于以显式意图目的而非推断需求指导合并粒度，并将更早前缀纳入上下文管理，而不仅是当前任务的后缀。
- **Agent Memory**：RAG-based 系统（mem0/A-Mem/MemoryBank）维护独立记忆库，依赖重要性评分或时间衰减；MemTree 按语义相似度组织树状记忆。Hippocam 的树直接源于正在进行的交互历史和 Consolidation 操作，无需独立记忆库或外部评分机制。
- **Learning from Experience**：Reflexion 使用语言反馈改进尝试；Agent Workflow Memory 累积可复用工作流；ACE/ReasoningBank 通过反馈更新推理记忆/playbook；ETO/EMPO² 通过参数更新从成功/失败中学习。Hippocam 的统一之处在于：不依赖参数更新，也不需要先将经验蒸馏为可重用策略（playbook 变体在 ALFWorld 上并无明显优势，Table 9），而是通过"使用与不使用"的自然循环在推理层完成学习与知识演进。

## 局限性与未来方向
- **成本权衡**：Prefix Consolidation 自身调用占总成本的约 34%（ToolBench），频繁的重组带来可测量的额外开销；且成本优势高度依赖缓存折扣（当缓存命中价格高于未缓存价格的约 15% 时优势可能消失）。
- **评估规模**：所有结果基于单一运行（single run），未报告方差；部分跨域实验（如 mem0/ReasoningBank 在 ScienceWorld 上匹配任务较少）可能存在评估覆盖不均。
- **意图分解的粒度**：现有实验主要展示意图在任务级或子目标级的嵌套，对于更细粒度（如单步工具调用是否需要意图）或更大规模的长期交互，意图堆栈的管理能力尚待验证。
- **递归 Consolidation 的极限**：叙事部分逐次变粗化，但过度递归可能导致关键细节丢失；"足够细节"的判定目前由主 Agent 通过渐进召回自行决定，缺乏显式的信息保真度保障。
- **潜在未来方向**：探索多 Agent 协作场景下的意图协调与知识共享；将"使用与不使用"的动态机制与轻量级参数微调（如 LoRA）结合以进一步降低推理层开销；在更长 horizon（>300 任务）和更复杂真实世界任务中验证可扩展性。

## 研究启发与可借鉴点
- **意图驱动的状态管理机制**：将交互历史按嵌套意图组织并据此触发 Consolidation，而非简单地按时间或相似度截断，这一设计可迁移到任何需要长期上下文管理的 Agent 系统，尤其适合多阶段、多子目标的任务场景。
- **叙事/知识分离的递归抽象**：PALIM 将已合并消息拆分为叙事（逐步粗化）和知识（保留细节并持续整合），这一思路可直接应用于构建不需要独立向量数据库的轻量级长期记忆系统，避免语义检索的开销与误差。
- **渐进式召回（Progressive Recall）替代一次性检索**：通过消息树结构按需逐级展开细节，而非在启动时检索全部相关记忆，可减少上下文噪音并节省 token 消耗；这一机制可与现有的 RAG 系统结合，作为第二级细化检索的补充。
- **使用/不使用驱动的知识演化**：用"是否被后续工作使用"而非人工设定的重要性分数来决定知识保留粒度，这一原则可用于设计更自然的经验过滤机制，特别适用于开放域、长周期的 Agent 应用（如个人助理、持续研发助手）。
- **跨域知识整合的实验设计**：Frozen + Sequential 双条件的跨域评估范式（$\Delta_{stream}$ 衡量源域知识对目标域学习的增益）可作为评测 Agent 泛化能力的标准模板，值得后续研究参考。

## 关键术语表
- **Intent Stack（意图堆栈）**：LIFO 结构的嵌套目的层次，最外层为全局目标，最内层为当前子目的，用于结构化 Agent 的交互上下文。
- **Intent Consolidation（意图 Consolidation）**：当 JANUS 关闭某个意图时，PRECIP 将该意图期间的工作消息合并为自包含的一人称记忆并替换原文，保留结论、证据及材料结构印象。
- **Prefix Consolidation（前缀 Consolidation）**：PALIM 对更早的历史前缀执行递归折叠，使叙事逐次变粗、提取的知识持续整合，实现"不使用即抽象"的效果。
- **Progressive Recall（渐进式召回）**：主 Agent 通过 `recall(uuid)` 工具沿消息树逐级展开已合并消息的子节点，按需恢复细粒度细节，直至满足当前任务需求。
- **Use-and-Disuse（使用与不使用）**：核心认知动态——被后续工作引用的知识被强化/更新，长期未被使用的细节通过递归 Consolidation 逐渐抽象，无需外部评分或时间衰减规则。
- **Narrative vs. Knowledge（叙事 vs. 知识）**：PALIM 将合并消息分为两部分：叙事记录"做了什么及结果"，逐次变粗；知识记录"关于某主体的真实陈述"，保留识别细节并持续合并更新。
- **Working Tail（工作尾部）**：当前上下文中最短的后缀（至少 $K$ 轮交互、$B$ tokens），受保护不被前缀 Consolidation 触及，确保最新工作细节始终可见。
- **Informative Failure（信息性失败）**：在 Intent Consolidation 中被特意保留的失败及其原因，区别于可忽略的日常小错，用于后续工作中的经验积累和模式学习。

## 可复现要素
- **数据集**：ALFWorld、ScienceWorld、StreamBench（DDXPlus + ToolBench）、HotpotQA、FEVER、GoodAI LTM、LoCoMo、MemoryAgentBench；均为公开基准。
- **代码/权重开源**：论文未明确声明代码开源，仅提供了详细的算法描述（Algorithm 1–2）和完整的 prompt 规范（Appendix B），可据此复现。
- **关键超参**：$K=2$、$B_{\min}=4096$、$\alpha=0.20$、$q=0.95$、$E_{\min}=2$、$T_{\min}=16384$、$W=128000$、压力阈值 $0.80W$、最小压缩量 $\Delta_{\min}=8192$ tokens、重试次数 $J=2$（见 Table 5）。
- **骨干模型**：DeepSeek-V4.1-Flash（任务主干）、DeepSeek-V4-Pro（辅助评分）；辅助组件（JANUS/PRECIP/PALIM）使用同系列模型。
- **上下文窗口**：1M token。
